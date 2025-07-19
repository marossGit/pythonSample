# BBandVWAPStrategyGA - Genetic Algorithmによるボリンジャーバンド＋VWAP戦略の最適化

import numpy as np
import pandas as pd
from stockLib.technical import calc_vwap, calc_rsi, calc_bollinger_band
from geneticalgorithm import geneticalgorithm as ga

class BBandVWAPStrategyGA:
    """
    遺伝的アルゴリズムによって最適化されるボリンジャーバンド＋VWAP＋RSI戦略クラス

    - 利確条件：tp_ratio または RSI または BB上限
    - ロスカット条件：sl_ratio または RSI または BB下限
    """

    def __init__(self, tp_ratio=1.03, sl_ratio=0.97, rsi_buy_thresh=30, rsi_sell_thresh=70, bb_n=2):
        self.tp_ratio = tp_ratio                  # 利確倍率（例：1.03で3%上昇）
        self.sl_ratio = sl_ratio                  # 損切倍率（例：0.97で3%下落）
        self.rsi_buy_thresh = rsi_buy_thresh      # RSIがこの値未満なら買いシグナル
        self.rsi_sell_thresh = rsi_sell_thresh    # RSIがこの値超なら売りシグナル
        self.bb_n = bb_n                          # ボリンジャーバンドのσ係数

    def run(self, df):
        """
        戦略を実行し、取引履歴と最終資産を返す
        """
        df = df.copy()

        # テクニカル指標の計算
        df["VWAP"] = calc_vwap(df)
        df["RSI"] = calc_rsi(df)
        df["BB_upper"], df["BB_lower"] = calc_bollinger_band(df["Close"], n=20, n_sigma=self.bb_n)

        capital = 1000000  # 初期資金
        position = 0       # 保有株数
        entry_price = 0
        trade_log = []

        for i in range(20, len(df)):
            price = df["Close"].iloc[i]
            rsi = df["RSI"].iloc[i]
            bb_upper = df["BB_upper"].iloc[i]
            bb_lower = df["BB_lower"].iloc[i]

            # 買い条件
            if position == 0:
                if rsi < self.rsi_buy_thresh and price <= bb_lower:
                    position = capital // price
                    entry_price = price
                    capital -= position * price
                    trade_log.append((df.index[i], "BUY", price))

            # 売り条件（OR判定）
            elif position > 0:
                exit = False

                if price >= entry_price * self.tp_ratio:
                    exit = True
                elif rsi > self.rsi_sell_thresh:
                    exit = True
                elif price >= bb_upper:
                    exit = True
                elif price <= entry_price * self.sl_ratio:
                    exit = True
                elif rsi < self.rsi_buy_thresh:
                    exit = True
                elif price <= bb_lower:
                    exit = True

                if exit:
                    capital += position * price
                    trade_log.append((df.index[i], "SELL", price))
                    position = 0

        if position > 0:
            capital += position * price
            trade_log.append((df.index[-1], "FINAL SELL", price))

        return trade_log, capital


def run_genetic_BBandVWAP_strategy(df):
    """
    遺伝的アルゴリズムでBBandVWAPStrategyGAのパラメータ最適化を行う
    """
    def fitness(params):
        tp, sl, rsi_buy, rsi_sell, bb_n = params
        strat = BBandVWAPStrategyGA(tp_ratio=tp, sl_ratio=sl,
                                     rsi_buy_thresh=int(rsi_buy), rsi_sell_thresh=int(rsi_sell),
                                     bb_n=int(bb_n))
        _, final_capital = strat.run(df)
        return -final_capital  # 最大化ではなく最小化されるのでマイナスを返す

    # パラメータ範囲の定義
    varbound = np.array([
        [1.01, 1.10],   # tp_ratio
        [0.90, 0.99],   # sl_ratio
        [10, 40],       # rsi_buy_thresh
        [60, 90],       # rsi_sell_thresh
        [1, 3]          # bb_n (σの係数)
    ])

    algorithm_param = {'max_num_iteration': 30,
                       'population_size': 20,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.1,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': 10}

    model = ga(function=fitness,
               dimension=5,
               variable_type='real',
               variable_boundaries=varbound,
               algorithm_parameters=algorithm_param)

    model.run()
    return model
