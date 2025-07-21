# stockLightGBMForeCast.py
# -------------------------
# 本コードは LightGBM による「株価が上昇する確率」を予測する分類機械学系のモデルです
# 予測目標は "5ヶ月後に株価が上がっているか"

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, fbeta_score, precision_score
)
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import shap

class stockLightGBMClassifier:
    def __init__(self, baseDeck, futureDeck, nikkeiRate):
        """
        モデルの初期化。
        baseDeck: 現在の株データリスト（特徴量抽出元）
        futureDeck: 約5ヶ月後の株価リスト（正解ラベル算出用）
        nikkeiRate: 期間中の日経平均リターン（株価変動補正に使用）
        """
        self.stockList = baseDeck["stockList"]
        self.futureStockList = futureDeck["stockList"]
        self.nikkeiRate = nikkeiRate

    def create_training_data(self):
        """
        銘柄ごとに特徴量とラベルを作成して、学習データを構築。
        特徴量は9項目、ラベルは「5ヶ月後に上昇したかどうか」
        """
        features = []
        labels = []

        for st in self.stockList:
            df = st.dayData
            if df is None or len(df) < 251:
                continue  # データ不足

            # 対応する将来データを検索
            match = next((f for f in self.futureStockList if f.companyId == st.companyId), None)
            if not match or match.dayData is None or len(match.dayData) < 251:
                continue

            try:
                # 現在終値と5ヶ月後終値の取得
                close_now = df["Close"].iloc[-1]
                close_future = match.dayData["Close"].iloc[-140]

                if close_now <= 1 or np.isnan(close_now) or np.isnan(close_future):
                    continue

                # 日経平均変動率で正規化したリターン計算
                price_return = (close_future - close_now) / close_now * 100 / self.nikkeiRate
                if abs(price_return) > 500:
                    continue  # 異常値除外

                if st.evalAll < 50:
                    continue  # 評価が低い銘柄は除外

                # 9つの特徴量を抽出
                row = [
                    st.evalPerform["score"][1],  # 資産と資本
                    st.evalPerform["score"][2],  # 売上と利益
                    st.evalPerform["score"][3],  # キャッシュフロー
                    st.evalPerform["score"][4],  # 理論株価との乖離
                    st.evalPerform["score"][5],  # ROEの詳細分析
                    st.evalPerform["score"][6],  # 四半期評価
                    st.evalChart["score"][5],    # チャートベースの非対称ベータ
                    st.evalSupply["score"][1],   # 出来高
                    st.evalSupply["score"][2]    # 需給
                ]

                if any([np.isnan(x) for x in row]):
                    continue

                label = 1 if price_return > 0 else 0  # 上昇:1 / 非上昇:0
                features.append(row)
                labels.append(label)

            except Exception:
                continue  # 予期せぬデータエラーはスキップ

        # DataFrame へ変換
        self.X = pd.DataFrame(features, columns=[
            "資産と資本", "売上と営業利益", "キャッシュフロー", "理論株価", "ROEの詳細", "四十四分析",
            "非常値ベータ", "出来高", "需給"
        ])
        self.y = pd.Series(labels)

    def train_model(self):
        """
        LightGBM分類器を学習し、基本的な評価スコアと特徴量重要度を表示。
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=100,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        probas = self.model.predict_proba(X_test)[:, 1]

        print("\n✅ モデル評価")
        print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
        print(f"AUC: {roc_auc_score(y_test, probas):.3f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print("\nClassification Report:\n", classification_report(y_test, preds))

        print("\n--- 特徴量重要度 ---")
        for name, score in zip(self.X.columns, self.model.feature_importances_):
            print(f"{name}: {score}")

    def predict_proba(self, x):
        """
        任意の9特徴量ベクトルを入力とし、
        クラス1（株価上昇）の確率を返す。
        """
        return self.model.predict_proba([x])[0][1]

    def evaluate_model_performance(self):
        """
        予測確率に対してF0.5スコアを最大化するしきい値を探索し、
        その評価を可視化（しきい値プロット・ヒストグラム・SHAP）
        """
        if not hasattr(self, "model"):
            print("❌ モデルが訓練されていません。train_model() を実行してください。")
            return

        X_test = self.X
        y_true = self.y
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # しきい値を0.3〜0.9まで試し、F0.5スコアを計算
        thresholds = np.linspace(0.3, 0.9, 100)
        best_thresh, best_f05 = 0.5, 0
        f05_scores = []

        for t in thresholds:
            preds = [1 if p >= t else 0 for p in y_prob]
            f05 = fbeta_score(y_true, preds, beta=0.5)
            f05_scores.append(f05)
            if f05 > best_f05:
                best_f05 = f05
                best_thresh = t

        print(f"\n🎯 最適なしきい値 (F0.5最大): {best_thresh:.2f} (F0.5: {best_f05:.3f})")
        final_preds = [1 if p >= best_thresh else 0 for p in y_prob]
        final_precision = precision_score(y_true, final_preds)
        print(f"Precision: {final_precision:.3f}")

        # --- しきい値 vs F0.5スコア プロット ---
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, f05_scores, label='F0.5 Score')
        plt.axvline(x=best_thresh, color='r', linestyle='--', label=f'Threshold = {best_thresh:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('F0.5 Score')
        plt.title('F0.5 Score vs Threshold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- クラス別予測確率のヒストグラム ---
        y_prob_0 = [p for p, t in zip(y_prob, y_true) if t == 0]
        y_prob_1 = [p for p, t in zip(y_prob, y_true) if t == 1]

        plt.figure(figsize=(8, 5))
        plt.hist(y_prob_0, bins=20, alpha=0.6, label='Class 0 (Down)', color='orange')
        plt.hist(y_prob_1, bins=20, alpha=0.6, label='Class 1 (Up)', color='skyblue')
        plt.axvline(x=best_thresh, color='red', linestyle='--', label=f'Threshold = {best_thresh:.2f}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Prediction Probability Distribution by True Class')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- SHAP値による特徴量の寄与度可視化 ---
        explainer = shap.Explainer(self.model.predict_proba, self.X)
        shap_values = explainer(self.X)

        font_path = "C:/Windows/Fonts/meiryo.ttc"  # 日本語ラベル対応フォント
        jp_font = fm.FontProperties(fname=font_path)

        shap.summary_plot(shap_values[:, :, 1], self.X, feature_names=self.X.columns, show=False)
        ax = plt.gca()
        for label in ax.get_yticklabels():
            label.set_fontproperties(jp_font)
            label.set_fontsize(12)
        plt.tight_layout()
        plt.show()
