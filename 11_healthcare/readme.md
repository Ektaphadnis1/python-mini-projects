# 🏥 Mini Project 11 — Healthcare Case Study
### Breast Cancer Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Problem Statement

Breast cancer is one of the most common cancers worldwide. Early and accurate detection is critical for improving patient survival rates. This project builds machine learning models to predict whether a tumor is **Malignant (cancerous)** or **Benign (non-cancerous)** based on cell nucleus features from digitized medical images.

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| **Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) |
| **File** | `data.csv` |
| **Rows** | 569 |
| **Features** | 30 numeric features |
| **Target** | `diagnosis` — B (Benign) / M (Malignant) |
| **Missing Values** | None |

### Key Features
- `radius_mean` — mean distance from center to perimeter
- `texture_mean` — standard deviation of gray-scale values
- `perimeter_mean` — mean size of the tumor core
- `area_mean` — mean area of the tumor
- `smoothness_mean` — local variation in radius lengths

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Plotting charts |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | ML models, preprocessing, evaluation |

---

## ⚙️ How to Run

### 1. Download dataset from Kaggle
Go to: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
Download `data.csv` and place it in this folder.

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the script
```bash
python healthcare.py
```

---

## 🔍 Project Workflow

```
Load CSV → EDA → Visualizations → Preprocessing → Model Training → Evaluation → Conclusion
```

| Step | Description |
|------|-------------|
| Data Loading | Read `data.csv` using `pd.read_csv()` |
| EDA | Shape, dtypes, statistics, missing values, class distribution |
| Preprocessing | Drop ID column, encode diagnosis, train/test split (80/20), StandardScaler |
| Visualizations | 7 plots — distribution, histogram, boxplot, heatmap, pairplot, confusion matrix, accuracy bar |
| Model Building | Logistic Regression, Decision Tree, Random Forest |
| Evaluation | Accuracy score, Classification Report, Confusion Matrix |

---

## 📊 Visualizations Generated

| File | Description |
|------|-------------|
| `plot1_distribution.png` | Count of Benign vs Malignant tumors |
| `plot2_radius_mean.png` | Histogram of radius mean by class |
| `plot3_boxplot.png` | Boxplot of area mean by tumor type |
| `plot4_heatmap.png` | Correlation heatmap of top 10 features |
| `plot5_pairplot.png` | Pairplot of 4 key features |
| `plot6_confusion_matrices.png` | Confusion matrix for all 3 models |
| `plot7_model_comparison.png` | Accuracy comparison bar chart |

---

## 🤖 Models & Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~97% |
| Decision Tree | ~94% |
| **Random Forest** | **~97%** |

---

## 💡 Key Findings

- `radius_mean`, `area_mean`, and `perimeter_mean` are the strongest predictors of malignancy.
- Malignant tumors consistently show **larger feature values** compared to benign ones.
- All three models achieved **above 90% accuracy**, showing the dataset is highly separable.
- Random Forest performed best overall due to its ensemble nature.

---

## 📚 References
- [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- [Scikit-Learn Docs](https://scikit-learn.org/stable/)

---

## 👤 Author
**Ekta Phadnis** — https://github.com/Ektaphadnis1/python-mini-projects/tree/main/11_healthcare

> *Mini Project 11 — Python Coursework Assignment*