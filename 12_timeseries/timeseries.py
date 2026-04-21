# ============================================================
#   MINI PROJECT 12 - TIME SERIES CASE STUDY
#   Dataset : Apple Inc. (AAPL) Stock Price (Kaggle CSV)
#   Goal    : Analyze stock trends & forecast using ARIMA
# ============================================================

# ── SECTION 1 : IMPORT LIBRARIES ────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, mean_absolute_error

print("=" * 60)
print("   MINI PROJECT 12 : STOCK PRICE TIME SERIES ANALYSIS")
print("=" * 60)


# ── SECTION 2 : LOAD DATASET ────────────────────────────────
# Make sure AAPL.csv is in the same folder as this script
print("\n[1] LOADING STOCK DATA FROM KAGGLE CSV...")

df_raw = pd.read_csv('AAPL.csv')

print("    Columns found:", list(df_raw.columns))
print(f"    Rows : {len(df_raw)}")
print("\n    First 5 rows:")
print(df_raw.head())

# ── Flexible column detection ────────────────────────────────
# Kaggle AAPL CSVs can have slightly different column names
# This handles both cases automatically

df_raw.columns = df_raw.columns.str.strip()   # remove accidental spaces

# Find the date column
date_col = next((c for c in df_raw.columns
                 if 'date' in c.lower()), df_raw.columns[0])

# Find the close price column
close_col = next((c for c in df_raw.columns
                  if 'close' in c.lower()), None)

if close_col is None:
    raise ValueError("Could not find a 'Close' column. "
                     f"Available columns: {list(df_raw.columns)}")

print(f"\n    Using date column  : '{date_col}'")
print(f"    Using close column : '{close_col}'")

# Build clean DataFrame
df = df_raw[[date_col, close_col]].copy()
df.columns = ['Date', 'Close']
df['Date']  = pd.to_datetime(df['Date'])
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Keep 2018–2023 to match project scope
df = df[(df.index >= '2018-01-01') & (df.index <= '2023-12-31')]
df.fillna(method='ffill', inplace=True)

print(f"\n    Date Range : {df.index.min().date()} → {df.index.max().date()}")
print(f"    Total rows : {len(df)}")


# ── SECTION 3 : EDA ─────────────────────────────────────────
print("\n[2] BASIC STATISTICS:")
print(df.describe())

print("\n[3] MISSING VALUES:")
print(df.isnull().sum())


# ── SECTION 4 : VISUALIZATIONS ──────────────────────────────

# --- Plot 1: Raw Closing Price ---
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Close'], color='steelblue', linewidth=1.2)
plt.title("AAPL Stock Closing Price (2018–2023)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot1_raw_price.png")
plt.show()
print("\n[4] Plot 1 saved: plot1_raw_price.png")

# --- Plot 2: Moving Averages ---
df['MA30'] = df['Close'].rolling(window=30).mean()
df['MA90'] = df['Close'].rolling(window=90).mean()

plt.figure(figsize=(12, 4))
plt.plot(df['Close'], label='Close Price', alpha=0.5, color='steelblue')
plt.plot(df['MA30'],  label='30-Day MA',   color='orange', linewidth=1.5)
plt.plot(df['MA90'],  label='90-Day MA',   color='red',    linewidth=1.5)
plt.title("AAPL Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot2_moving_averages.png")
plt.show()
print("[5] Plot 2 saved: plot2_moving_averages.png")

# --- Plot 3: Daily Returns ---
df['Daily Return'] = df['Close'].pct_change() * 100

plt.figure(figsize=(12, 4))
plt.plot(df['Daily Return'], color='tomato', linewidth=0.8)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title("AAPL Daily Returns (%)")
plt.xlabel("Date")
plt.ylabel("Return (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot3_daily_returns.png")
plt.show()
print("[6] Plot 3 saved: plot3_daily_returns.png")

# --- Plot 4: Distribution of Daily Returns ---
plt.figure(figsize=(8, 4))
sns.histplot(df['Daily Return'].dropna(), bins=60, kde=True, color='steelblue')
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return (%)")
plt.tight_layout()
plt.savefig("plot4_return_distribution.png")
plt.show()
print("[7] Plot 4 saved: plot4_return_distribution.png")

# --- Plot 5: Decomposition ---
print("\n[8] DECOMPOSING TIME SERIES...")
weekly = df['Close'].resample('W').mean()

decomposition = seasonal_decompose(weekly, model='multiplicative', period=52)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Observed',    color='steelblue')
decomposition.trend.plot(ax=axes[1],    title='Trend',       color='orange')
decomposition.seasonal.plot(ax=axes[2], title='Seasonality', color='green')
decomposition.resid.plot(ax=axes[3],    title='Residuals',   color='red')
plt.suptitle("Time Series Decomposition (AAPL Weekly)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("plot5_decomposition.png")
plt.show()
print("[9] Plot 5 saved: plot5_decomposition.png")


# ── SECTION 5 : STATIONARITY TEST ───────────────────────────
print("\n[10] STATIONARITY TEST (Augmented Dickey-Fuller):")

def adf_test(series, name="Series"):
    result = adfuller(series.dropna())
    print(f"\n    {name}")
    print(f"    ADF Statistic : {result[0]:.4f}")
    print(f"    p-value       : {result[1]:.4f}")
    if result[1] <= 0.05:
        print("    ✅ Stationary (p ≤ 0.05) — ready for ARIMA")
    else:
        print("    ❌ Non-Stationary (p > 0.05) — needs differencing")

adf_test(df['Close'], "Original Close Price")

df['Close_diff'] = df['Close'].diff()
adf_test(df['Close_diff'], "After 1st Differencing")


# ── SECTION 6 : ACF & PACF ──────────────────────────────────
print("\n[11] PLOTTING ACF & PACF...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['Close_diff'].dropna(),  lags=40, ax=axes[0],
         title="ACF (Autocorrelation)")
plot_pacf(df['Close_diff'].dropna(), lags=40, ax=axes[1],
          title="PACF (Partial Autocorrelation)")
plt.suptitle("ACF & PACF — Used to choose ARIMA(p,d,q)", fontsize=12)
plt.tight_layout()
plt.savefig("plot6_acf_pacf.png")
plt.show()
print("[12] Plot 6 saved: plot6_acf_pacf.png")


# ── SECTION 7 : ARIMA MODEL ──────────────────────────────────
print("\n[13] BUILDING ARIMA MODEL...")

monthly = df['Close'].resample('ME').mean()

train = monthly[:-12]
test  = monthly[-12:]

print(f"    Training months : {len(train)}")
print(f"    Testing months  : {len(test)}")

model  = ARIMA(train, order=(2, 1, 2))
fitted = model.fit()

print("\n    ARIMA Model Summary:")
print(fitted.summary())


# ── SECTION 8 : FORECAST ─────────────────────────────────────
print("\n[14] FORECASTING NEXT 12 MONTHS...")

forecast_result = fitted.get_forecast(steps=12)
forecast_mean   = forecast_result.predicted_mean
forecast_ci     = forecast_result.conf_int()

# --- Plot 7: Forecast vs Actual ---
plt.figure(figsize=(12, 5))
plt.plot(train.index, train,
         label='Training Data',  color='steelblue')
plt.plot(test.index,  test,
         label='Actual Price',   color='green',  linewidth=2)
plt.plot(forecast_mean.index, forecast_mean,
         label='Forecast',       color='red',    linewidth=2, linestyle='--')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color='pink', alpha=0.4, label='95% Confidence Interval')
plt.title("AAPL Stock Price: ARIMA Forecast vs Actual (Monthly)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot7_arima_forecast.png")
plt.show()
print("[15] Plot 7 saved: plot7_arima_forecast.png")


# ── SECTION 9 : EVALUATION ───────────────────────────────────
print("\n[16] MODEL EVALUATION:")

mae  = mean_absolute_error(test, forecast_mean)
rmse = np.sqrt(mean_squared_error(test, forecast_mean))
mape = np.mean(np.abs((test.values - forecast_mean.values)
                       / test.values)) * 100

print(f"    MAE  : ${mae:.2f}")
print(f"    RMSE : ${rmse:.2f}")
print(f"    MAPE : {mape:.2f}%")

# --- Plot 8: Actual vs Predicted ---
plt.figure(figsize=(10, 4))
plt.plot(test.index, test.values,
         label='Actual',    color='green', marker='o')
plt.plot(forecast_mean.index, forecast_mean.values,
         label='Predicted', color='red',   marker='x', linestyle='--')
plt.title("Actual vs Predicted — Test Period (Last 12 Months)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot8_actual_vs_predicted.png")
plt.show()
print("[17] Plot 8 saved: plot8_actual_vs_predicted.png")


# ── SECTION 10 : CONCLUSION ──────────────────────────────────
print("\n" + "=" * 60)
print("   CONCLUSION")
print("=" * 60)
print(f"""
  Dataset    : Apple Inc. (AAPL) Stock Price (Kaggle)
  Period     : 2018–2023  |  Daily → Monthly for ARIMA

  Analysis Performed:
    ✅ Raw price trend visualization
    ✅ 30-day & 90-day moving averages
    ✅ Daily return & volatility analysis
    ✅ Time series decomposition
    ✅ Stationarity test (ADF Test)
    ✅ ACF & PACF plots
    ✅ ARIMA(2,1,2) forecasting

  Model Evaluation (Last 12 Months):
    MAE  : ${mae:.2f}
    RMSE : ${rmse:.2f}
    MAPE : {mape:.2f}%

  Key Findings:
    - AAPL shows a strong long-term upward trend.
    - Non-stationary in raw form; stationary after differencing.
    - ARIMA captures general trend direction.
    - Moving averages clearly show bull and bear phases.
""")
print("=" * 60)
print("   PROJECT COMPLETE!")
print("=" * 60)