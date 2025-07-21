# stock_forecast_portfolio.py
# 予想2：日経平均補正付き理論株価予測モデル
# 財務データ（自己資本・利益）＋日経平均連動補正を用いた理論株価推定
# 参考：独自設計

import datetime
import calendar
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class StockForecastDisp:
    def __init__(self, stock, monthly_data, qData):
        self.stock = stock
        self.monthly_data = monthly_data  # 月次終値（株価、NIKKEI）
        self.qData = qData                # 四半期財務データ（Equity, Profit, etc.）

    def get_average_nikkei_last_3months(self, idx):
        """
        指定インデックス以前の3ヶ月のNIKKEI平均を算出
        """
        start = max(0, idx - 2)
        return self.monthly_data.iloc[start:idx+1]["NIKKEI_Close"].mean()

    def solveB(self, j, r=1.0):
        """
        過去2期分の四半期決算データから、理論株価モデルの係数a, bを最小二乗法で推定

        p = (a × 自己資本 + b × 利益) × 日経平均補正係数
        
        r：補正強度（0.0=無補正〜1.0=完全補正）
        """
        data = []
        if j - 1 < 0:
            return None, None

        for k in range(2):
            try:
                row = self.qData.iloc[j-k]
                date_obj = datetime.datetime.strptime(row["DisclosedDate"], "%Y-%m-%d").date()
                last_day = calendar.monthrange(date_obj.year, date_obj.month)[1]
                month_end_datetime = datetime.datetime(date_obj.year, date_obj.month, last_day)
                
                # 対象月の株価と日経平均
                stock_price = float(self.monthly_data.loc[month_end_datetime]["Close"])
                nikkei_price = float(self.monthly_data.loc[month_end_datetime]["NIKKEI_Close"])
                
                # 補正係数：NIKKEI / 過去3ヶ月平均
                average_nikkei = self.get_average_nikkei_last_3months(self.monthly_data.index.get_loc(month_end_datetime))
                adjusted_n = (1 - r) + r * (nikkei_price / average_nikkei)

                equity = float(row["Equity"])
                profit = float(row.get("OrdinaryProfit", row.get("Profit", 0)))
                data.append({"p": stock_price, "v": equity / 1e8, "i": profit / 1e8, "adjusted_n": adjusted_n})
            except:
                continue

        if len(data) < 2:
            return None, None

        def objective(params):
            a, b = params
            return sum((d['p'] - ((d['v'] * a + d['i'] * b) * d['adjusted_n'])) ** 2 for d in data)

        result = minimize(objective, [0.1, 0.1], bounds=[(0, None), (0, None)], method='L-BFGS-B')
        return result.x if result.success else (None, None)

    def getForecastPrice2(self, r=1.0):
        """
        現在の財務データとsolveB()で得たa, bから理論株価を算出
        - 日経平均による補正付きモデル
        """
        if len(self.qData) < 2:
            return None

        a, b = self.solveB(j=len(self.qData)-1, r=r)
        if a is None or b is None:
            return None

        row = self.qData.iloc[-1]
        equity = float(row["Equity"])
        profit = float(row.get("OrdinaryProfit", row.get("Profit", 0)))
        date_obj = datetime.datetime.strptime(row["DisclosedDate"], "%Y-%m-%d").date()
        last_day = calendar.monthrange(date_obj.year, date_obj.month)[1]
        month_end_datetime = datetime.datetime(date_obj.year, date_obj.month, last_day)

        nikkei_price = float(self.monthly_data.loc[month_end_datetime]["NIKKEI_Close"])
        average_nikkei = self.get_average_nikkei_last_3months(self.monthly_data.index.get_loc(month_end_datetime))
        adjusted_n = (1 - r) + r * (nikkei_price / average_nikkei)

        fair_price = (equity / 1e8 * a + profit / 1e8 * b) * adjusted_n
        return fair_price
