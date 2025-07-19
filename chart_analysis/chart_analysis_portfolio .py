# Technical Chart Evaluation with Regression & Human-Readable Descriptions
# ポートフォリオ公開用に成形された、回帰分析を使ったテクニカルチャート評価モジュール

import numpy as np
import pandas as pd
import statsmodels.api as sm

class ChartEvaluator:
    """
    株価チャートに対して統計的に回帰評価を行い、
    傾向・位置・ボラティリティ・RSI・ベータ特性などを
    人間にわかりやすい形でスコア化・言語化するクラス。
    """

    def __init__(self, price_df, nikkei_df=None):
        """
        Parameters:
        - price_df: pandas.DataFrame（日次株価データ、'Open', 'High', 'Low', 'Close', 'ATR', 'RSI'等を含む）
        - nikkei_df: pandas.DataFrame（日経平均データ、ベータ分析に使用）
        """
        self.df = price_df
        self.nikkei = nikkei_df
        self.eval = {}

    def calc_trend(self, series):
        """一次回帰を用いたトレンド傾きとノイズ（残差）評価"""
        x = np.arange(len(series))
        y = series.values

        x_scaled = (x - np.mean(x)) / np.std(x)
        y_scaled = (y - np.mean(y)) / np.std(y) if np.std(y) > 0 else np.zeros_like(y)

        coef = np.polyfit(x_scaled, y_scaled, 1)
        y_pred = np.polyval(coef, x_scaled)

        slope = round(coef[0], 3)
        residual_std = round(np.std(y_scaled - y_pred), 3)

        return slope, residual_std

    def calc_quadratic_trend(self, series):
        """二次回帰を使い、曲線的なトレンド・反転傾向を分析"""
        x = np.arange(len(series))
        y = series.values

        x_scaled = (x - np.mean(x)) / np.std(x)
        y_scaled = (y - np.mean(y)) / np.std(y) if np.std(y) > 0 else np.zeros_like(y)

        coeffs = np.polyfit(x_scaled, y_scaled, 2)
        y_pred = np.polyval(coeffs, x_scaled)
        residual_std = round(np.std(y_scaled - y_pred), 3)

        a, b, c = coeffs
        slope_start = round(2*a*x_scaled[0] + b, 3)
        slope_end = round(2*a*x_scaled[-1] + b, 3)

        return {
            "a": round(a, 3),
            "b": round(b, 3),
            "c": round(c, 3),
            "residual_std": residual_std,
            "slope_start": slope_start,
            "slope_end": slope_end
        }

    def describe_trend(self, slope):
        """一次傾きに対する自然言語による説明"""
        if slope < -0.5:
            return "下降トレンドです。", 1
        elif slope < 0:
            return "やや下降傾向です。", 2
        elif slope < 0.3:
            return "横ばい傾向です。", 3
        elif slope < 0.6:
            return "やや上昇傾向です。", 4
        else:
            return "上昇トレンドです。", 5

    def describe_quadratic(self, result):
        """2次トレンドの結果を自然言語で要約"""
        desc = []

        if result['residual_std'] > 1.0:
            desc.append("ノイズが多く、トレンド把握は困難です。")
            score = 1
        elif result['residual_std'] > 0.7:
            desc.append("ばらつきがやや大きいです。")
            score = 3
        elif result['residual_std'] > 0.3:
            desc.append("ややフィットしています。")
            score = 4
        else:
            desc.append("滑らかにフィットしています。")
            score = 5

        if result['a'] > 0:
            if result['b'] > 0:
                desc.append("初期から上昇基調です。")
            else:
                desc.append("一度下がってからの回復傾向です。")
        else:
            desc.append("山型で下落傾向です。")
            score = min(score, 2)

        if result['slope_end'] < 0:
            desc.append("最後の傾きが下向きで注意が必要です。")
            score = min(score, 2)

        return " ".join(desc), score

    def evaluate_price_position(self):
        """現在価格が直近1ヶ月のどの水準かを評価"""
        now = self.df['Close'].iloc[-1]
        high = self.df['High'].iloc[-20:].max()
        low = self.df['Low'].iloc[-20:].min()
        rate = (now - low) / (high - low + 1e-6)

        if rate < 0.1:
            return "ほぼ最安値圏です。割安。", 5
        elif rate < 0.3:
            return "安値圏です。", 4
        elif rate < 0.5:
            return "中間的な水準です。", 3
        elif rate < 0.7:
            return "やや高値圏です。", 2
        else:
            return "高値圏にあります。", 1

    def evaluate_rsi_atr(self):
        """RSIとATRによるエントリー適性評価"""
        rsi = self.df['RSI'].iloc[-1]
        atr = self.df['ATR'].iloc[-1]
        close = self.df['Close'].iloc[-1]

        atr_ratio = atr / close if close else 0

        if np.isnan(rsi) or np.isnan(atr):
            return "RSIまたはATRが取得できません。", 1

        if atr_ratio < 0.005:
            return "ボラが非常に低く、動意に乏しいです。", 1
        elif atr_ratio > 0.04:
            return "ボラが高すぎて不安定です。", 2

        if rsi < 30:
            return "売られすぎ水準で反発に期待。", 5
        elif rsi > 70:
            return "買われすぎで反落に注意。", 2
        elif 45 <= rsi <= 55:
            return "RSIは中立圏で安定的です。", 4
        else:
            return "RSIはやや過熱/弱含みです。", 3

    def evaluate_asymmetric_beta(self):
        """日経平均との連動性（非対称ベータ）を分析"""
        if self.nikkei is None:
            return "日経平均データ未設定のため分析不可。", 1

        returns = pd.DataFrame({
            'stock': self.df['Close'].pct_change(),
            'nikkei': self.nikkei['Close'].pct_change()
        }).dropna()

        up = returns[returns['nikkei'] > 0]
        down = returns[returns['nikkei'] < 0]

        def beta(x, y):
            x = sm.add_constant(x)
            return sm.OLS(y, x).fit().params[1]

        beta_up = beta(up['nikkei'], up['stock'])
        beta_down = beta(down['nikkei'], down['stock'])
        diff = beta_up - beta_down

        if diff < -0.3:
            return "下落時に敏感、上昇時に弱い傾向です。", 1
        elif diff < -0.1:
            return "やや下落時に反応しやすいです。", 2
        elif diff < 0.1:
            return "上昇・下落の反応差は小さいです。", 3
        elif diff < 0.3:
            return "上昇にやや強い安定型です。", 4
        else:
            return "上昇に強く、下落に鈍い優秀な反応です。", 5

    def evaluate_chart(self):
        """総合的にチャートの形状・各指標を判定し、文章とスコアを返す"""
        summary = []
        total_score = 0

        slope, _ = self.calc_trend(self.df['Open'])
        desc1, s1 = self.describe_trend(slope)
        summary.append("[一次回帰] " + desc1)
        total_score += s1

        quad = self.calc_quadratic_trend(self.df['Open'])
        desc2, s2 = self.describe_quadratic(quad)
        summary.append("[二次回帰] " + desc2)
        total_score += s2

        desc3, s3 = self.evaluate_price_position()
        summary.append("[現在値評価] " + desc3)
        total_score += s3

        desc4, s4 = self.evaluate_rsi_atr()
        summary.append("[RSI / ATR] " + desc4)
        total_score += s4

        desc5, s5 = self.evaluate_asymmetric_beta()
        summary.append("[ベータ分析] " + desc5)
        total_score += s5

        return "\n".join(summary), total_score
