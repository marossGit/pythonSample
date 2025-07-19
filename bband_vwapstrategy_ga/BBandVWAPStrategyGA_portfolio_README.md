# BBandVWAPStrategyGA - Genetic Algorithmによる最適化戦略

この戦略は、以下の3つのテクニカル指標を組み合わせて売買判断を行います：

- ボリンジャーバンド（BB）
- VWAP（出来高加重平均価格）
- RSI（相対力指数）

---

## 🚀 売買ルール

### エントリー条件（買い）：
- RSI < `rsi_buy_thresh`（例：30）
- `Close`価格 ≦ BB下限

### エグジット条件（売り）：
以下のいずれかを満たした場合に利確／損切（**OR判定**）：

#### 利確条件（いずれか）：
- `Close`価格 ≧ エントリー価格 × `tp_ratio`
- RSI > `rsi_sell_thresh`
- `Close`価格 ≧ BB上限

#### ロスカット条件（いずれか）：
- `Close`価格 ≦ エントリー価格 × `sl_ratio`
- RSI < `rsi_buy_thresh`
- `Close`価格 ≦ BB下限

---

## 🧬 最適化方法（Genetic Algorithm）
5つのパラメータをGAで最適化：

| パラメータ        | 範囲       | 説明                     |
|-------------------|------------|--------------------------|
| `tp_ratio`        | 1.01 - 1.10 | 利確倍率                |
| `sl_ratio`        | 0.90 - 0.99 | ロスカット倍率          |
| `rsi_buy_thresh`  | 10 - 40     | RSIの買い閾値           |
| `rsi_sell_thresh` | 60 - 90     | RSIの売り閾値           |
| `bb_n`            | 1 - 3       | ボリンジャーバンドσ係数 |

---

## 📈 初期資金
- 1,000,000円からスタート

## 🧪 使用ライブラリ
- `numpy`
- `pandas`
- `geneticalgorithm`
- 自作モジュール `stockLib.technical` より `calc_vwap`, `calc_rsi`, `calc_bollinger_band`

## ⚠ 注意点
- 入力DataFrameには `Open`, `High`, `Low`, `Close`, `Volume` の列が必要です。
- 最低20日分のデータが必要です（BB計算のため）

---


