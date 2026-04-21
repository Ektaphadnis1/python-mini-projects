# 🐍 Python Mini Project

A collection of Python data science mini projects built as part of a coursework assignment. Covers two domains — **Healthcare Classification** and **Stock Price Time Series Forecasting**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Projects](https://img.shields.io/badge/Projects-2-orange)

---

## 📁 Repository Structure

```
python-mini-projects/
│
├── project11_healthcare/
│   ├── healthcare.py          ← Main Python script
│   ├── data.csv               ← Kaggle dataset (Breast Cancer Wisconsin)
│   ├── plot1_distribution.png
│   ├── plot2_radius_mean.png
│   ├── plot3_boxplot.png
│   ├── plot4_heatmap.png
│   ├── plot5_pairplot.png
│   ├── plot6_confusion_matrices.png
│   ├── plot7_model_comparison.png
│   └── README.md
│
├── project12_timeseries/
│   ├── timeseries.py          ← Main Python script
│   ├── AAPL.csv               ← Kaggle dataset (Apple Stock Price)
│   ├── plot1_raw_price.png
│   ├── plot2_moving_averages.png
│   ├── plot3_daily_returns.png
│   ├── plot4_return_distribution.png
│   ├── plot5_decomposition.png
│   ├── plot6_acf_pacf.png
│   ├── plot7_arima_forecast.png
│   ├── plot8_actual_vs_predicted.png
│   └── README.md
│
└── README.md                  ← You are here
```

---

## 📊 Projects Overview

| # | Project | Dataset | Source | Type | Models |
|---|---------|---------|--------|------|--------|
| 11 | [🏥 Healthcare Case Study](./project11_healthcare/) | Breast Cancer Wisconsin | Kaggle | Classification | Logistic Regression, Decision Tree, Random Forest |
| 12 | [📈 Time Series Case Study](./project12_timeseries/) | Apple Stock Price (AAPL) | Kaggle | Forecasting | ARIMA(2,1,2) |

---

## ⚙️ Install All Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

---

## ▶️ How to Run

```bash
# Project 11
cd project11_healthcare
python healthcare.py

# Project 12
cd project12_timeseries
python timeseries.py
```

---

## 👤 Author

**Your Name**
GitHub: [@your-username](https://github.com/your-username)

> *Built as part of a Python Mini Projects coursework assignment.*