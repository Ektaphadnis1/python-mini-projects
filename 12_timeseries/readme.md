# 📈 Mini Project 12 — Time Series Case Study
### Apple Stock Price Analysis & ARIMA Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA-purple)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Problem Statement

Stock prices are one of the most challenging real-world time series datasets due to their volatility and non-stationarity. This project performs a complete time series analysis on **Apple Inc. (AAPL)** closing stock prices from 2018–2023, covering trend analysis, decomposition, stationarity testing, and 12-month price forecasting using the **ARIMA model**.

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| **Stock** | Apple Inc. — Ticker: `AAPL` |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data) |
| **File** | `AAPL.csv` |
| **Period** | January 2018 — December 2023 |
| **Frequency** | Daily (resampled to Monthly for ARIMA) |
| **Target Column** | `Close` — Closing Price (USD) |

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Time series & forecast plots |
| `seaborn` | Distribution plots |
| `statsmodels` | Decomposition, ADF test, ARIMA model |
| `scikit-learn` | Evaluation metrics (MAE, RMSE) |

---

## ⚙️ How to Run

### 1. Download dataset from Kaggle
Go to: https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data
Download `AAPL.csv` and place it in this folder.

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

### 3. Run the script
```bash
python timeseries.py
```

---

## 🔍 Project Workflow

```
Load CSV → EDA → Visualizations → Decomposition → ADF Test → ACF/PACF → ARIMA → Forecast → Evaluation
```

| Step | Description |
|------|-------------|
| Data Loading | Read `AAPL.csv`, parse dates, filter 2018–2023 |
| EDA | Statistics, missing values, daily return calculation |
| Decomposition | Separate Trend, Seasonality, Residuals (weekly data) |
| Stationarity | ADF Test → apply 1st differencing if needed |
| ACF & PACF | Determine optimal p, d, q parameters for ARIMA |
| ARIMA | Train on monthly data, forecast last 12 months |
| Evaluation | MAE, RMSE, MAPE on test set |

---

## 📊 Visualizations Generated

| File | Description |
|------|-------------|
| `plot1_raw_price.png` | AAPL daily closing price 2018–2023 |
| `plot2_moving_averages.png` | 30-day & 90-day moving averages |
| `plot3_daily_returns.png` | Daily percentage returns (volatility) |
| `plot4_return_distribution.png` | Distribution of daily returns |
| `plot5_decomposition.png` | Trend, Seasonality & Residual decomposition |
| `plot6_acf_pacf.png` | ACF & PACF plots for ARIMA parameter selection |
| `plot7_arima_forecast.png` | Forecast vs actual with 95% confidence interval |
| `plot8_actual_vs_predicted.png` | Actual vs predicted for test period |

---

## 🤖 ARIMA Model — Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **p** | 2 | Autoregressive terms (from PACF plot) |
| **d** | 1 | Differencing order (from ADF test) |
| **q** | 2 | Moving average terms (from ACF plot) |

---

## 📈 Model Evaluation

| Metric | Meaning |
|--------|---------|
| **MAE** | Average dollar error per prediction |
| **RMSE** | Penalises large errors more heavily |
| **MAPE** | Percentage error relative to actual price |

> Exact values are printed when you run `timeseries.py`

---

## 💡 Key Findings

- AAPL shows a **strong long-term upward trend** from 2018 to 2023.
- Raw stock price is **non-stationary** — confirmed by ADF test. 1st differencing makes it stationary.
- **Daily returns** follow a near-normal distribution around 0%, consistent with random walk theory.
- ARIMA captures **overall trend direction** but struggles with sharp market movements — a known limitation of linear models on stock data.
- The **90-day moving average** clearly highlights bull and bear market phases.

---

## ⚠️ Limitations

- ARIMA assumes **linear relationships** and may not capture complex market dynamics.
- Stock prices are influenced by unpredictable external factors (news, policy, earnings).
- For higher accuracy, advanced models like **LSTM** or **Facebook Prophet** could be explored.

---

## 📚 References
- [Kaggle Dataset](https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data)
- [Statsmodels ARIMA Docs](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)

---

## 👤 Author
**Ekta Phadnis** — https://github.com/Ektaphadnis1/python-mini-projects/tree/main/12_timeseries

> *Mini Project 12 — Python Coursework Assignment*