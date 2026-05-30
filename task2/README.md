# 📈 Task 2: Apple (AAPL) Stock Price Prediction

**DevelopersHub Corporation — AI/ML Engineering Internship**

---

## 📌 Task Objective

Use historical Apple stock data (Open, High, Low, Volume) along with engineered time-series features to predict the **next day's closing price** using regression models.

---

## 📂 Dataset Used

| Property | Detail |
|---|---|
| **Stock** | Apple Inc. (AAPL) |
| **Source** | Yahoo Finance via `yfinance` library |
| **Period** | January 2022 — December 2024 |
| **Frequency** | Daily (business days only) |
| **Raw Features** | Open, High, Low, Close, Volume |

---

## 🛠️ Libraries Used

- `yfinance` — fetch stock data from Yahoo Finance
- `pandas`, `numpy` — data handling and feature engineering
- `scikit-learn` — model training and evaluation
- `matplotlib`, `seaborn` — visualization

---

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| `MA_5`, `MA_20` | 5-day and 20-day moving average of Close |
| `Return_1d` | Previous day's percentage return |
| `Lag_1`, `Lag_2`, `Lag_3` | Close prices from 1, 2, 3 days ago |
| `HL_spread` | High minus Low (daily volatility) |
| `OC_spread` | Open minus Close (intraday direction) |
| `Volume_MA5` | 5-day average trading volume |
| `Target` | Next day's closing price (what we predict) |

---

## 🤖 Models Applied

### 1. Linear Regression
- Baseline model
- Assumes linear relationship between features and next-day close
- Features scaled with `StandardScaler`

### 2. Random Forest Regressor
- Ensemble of 200 decision trees
- Captures non-linear feature interactions
- Provides feature importance ranking

---

## 📊 Visualizations Created

| Plot | File |
|---|---|
| AAPL price history + volume | `aapl_history.png` |
| Actual vs predicted (both models) | `aapl_predictions.png` |
| Predicted vs actual scatter | `scatter_pred_actual.png` |
| Feature importance (RF) | `feature_importance.png` |
| Price with moving averages | `moving_averages.png` |

---

## 🔍 Key Results and Findings

1. **Lag features dominate** — yesterday's closing price is the single strongest predictor of tomorrow's price (high auto-correlation in stock prices).
2. **Random Forest outperforms Linear Regression** — captures non-linear relationships and feature interactions.
3. **Moving averages add trend context** — MA_5 and MA_20 crossovers are classic buy/sell signals in technical analysis.
4. **Model limitation:** Sudden price movements due to earnings or news events are unpredictable from technical features alone.

---

## 🚀 How to Run

```bash
# Install dependencies
pip install yfinance scikit-learn pandas numpy matplotlib seaborn notebook

# Launch notebook
jupyter notebook task2_stock_prediction.ipynb
```

---

## 📁 File Structure

```
task2/
├── task2_stock_prediction.ipynb   # Main Jupyter Notebook
├── AAPL_data.csv                  # Fallback dataset (if yfinance unavailable)
├── aapl_history.png
├── aapl_predictions.png
├── scatter_pred_actual.png
├── feature_importance.png
├── moving_averages.png
└── README.md
```

---

*Submitted as part of the DevelopersHub Corporation AI/ML Engineering Internship — Task 2*
