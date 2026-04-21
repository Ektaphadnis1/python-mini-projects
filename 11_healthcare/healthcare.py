# ============================================================
#   MINI PROJECT 11 - HEALTHCARE CASE STUDY
#   Dataset : Breast Cancer Wisconsin (Kaggle CSV)
#   Goal    : Predict whether a tumor is Malignant or Benign
# ============================================================

# ── SECTION 1 : IMPORT LIBRARIES ────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             classification_report)

print("=" * 60)
print("   MINI PROJECT 11 : BREAST CANCER CASE STUDY")
print("=" * 60)


# ── SECTION 2 : LOAD DATASET ────────────────────────────────
# Make sure data.csv is in the same folder as this script
df = pd.read_csv('data.csv')

print("\n[1] DATASET LOADED FROM KAGGLE CSV")
print(f"    Rows    : {df.shape[0]}")
print(f"    Columns : {df.shape[1]}")
print("\n    First 5 rows:")
print(df.head())


# ── SECTION 3 : DATA EXPLORATION (EDA) ──────────────────────
print("\n[2] DATASET INFO:")
print(df.info())

print("\n[3] BASIC STATISTICS:")
print(df.describe())

print("\n[4] MISSING VALUES:")
print(df.isnull().sum())

# Drop 'id' column and any unnamed columns (not useful for ML)
df.drop(columns=[col for col in df.columns if 'id' in col.lower()
                 or 'unnamed' in col.lower()], inplace=True)

# The 'diagnosis' column: M = Malignant, B = Benign
# Encode: B → 0, M → 1
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

print("\n[5] TARGET CLASS DISTRIBUTION:")
print(df['diagnosis'].value_counts())
print("    (0 = Benign, 1 = Malignant)")


# ── SECTION 4 : VISUALIZATIONS ──────────────────────────────

# --- Plot 1: Target Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosis', data=df, palette='Set2')
plt.title("Tumor Type Distribution\n(0 = Benign, 1 = Malignant)")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.tight_layout()
plt.savefig("plot1_distribution.png")
plt.show()
print("\n[6] Plot 1 saved: plot1_distribution.png")

# --- Plot 2: Histogram of Radius Mean ---
plt.figure(figsize=(6, 4))
sns.histplot(df['radius_mean'], hue=df['diagnosis'], kde=True, palette='Set1')
plt.title("Radius Mean: Benign vs Malignant")
plt.xlabel("Radius Mean")
plt.tight_layout()
plt.savefig("plot2_radius_mean.png")
plt.show()
print("[7] Plot 2 saved: plot2_radius_mean.png")

# --- Plot 3: Boxplot - Area Mean by Diagnosis ---
plt.figure(figsize=(6, 4))
sns.boxplot(x='diagnosis', y='area_mean', data=df, palette='coolwarm')
plt.title("Area Mean by Tumor Type")
plt.xlabel("Diagnosis (0=Benign, 1=Malignant)")
plt.ylabel("Area Mean")
plt.tight_layout()
plt.savefig("plot3_boxplot.png")
plt.show()
print("[8] Plot 3 saved: plot3_boxplot.png")

# --- Plot 4: Correlation Heatmap (top 10 features) ---
plt.figure(figsize=(10, 7))
top_features = df.iloc[:, :10]
sns.heatmap(top_features.corr(), annot=True, fmt=".2f", cmap='Blues')
plt.title("Feature Correlation Heatmap (Top 10 Features)")
plt.tight_layout()
plt.savefig("plot4_heatmap.png")
plt.show()
print("[9] Plot 4 saved: plot4_heatmap.png")

# --- Plot 5: Pairplot ---
pair_df = df[['radius_mean', 'texture_mean', 'perimeter_mean',
              'area_mean', 'diagnosis']].copy()
pair_df['diagnosis'] = pair_df['diagnosis'].map({0: 'Benign', 1: 'Malignant'})
sns.pairplot(pair_df, hue='diagnosis', palette='Set1')
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.savefig("plot5_pairplot.png")
plt.show()
print("[10] Plot 5 saved: plot5_pairplot.png")


# ── SECTION 5 : PREPROCESSING ───────────────────────────────
print("\n[11] PREPROCESSING DATA...")

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"    Training samples : {X_train.shape[0]}")
print(f"    Testing samples  : {X_test.shape[0]}")


# ── SECTION 6 : MODEL BUILDING ──────────────────────────────
print("\n[12] TRAINING MODELS...")

models = {
    "Logistic Regression" : LogisticRegression(max_iter=10000),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "Random Forest"       : RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n    ── {name} ──")
    print(f"    Accuracy : {acc * 100:.2f}%")
    print("    Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Benign', 'Malignant']))


# ── SECTION 7 : CONFUSION MATRIX ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    ax.set_title(f"{name}\nAccuracy: {results[name]*100:.2f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("plot6_confusion_matrices.png")
plt.show()
print("\n[13] Plot 6 saved: plot6_confusion_matrices.png")


# ── SECTION 8 : MODEL COMPARISON ────────────────────────────
plt.figure(figsize=(7, 4))
plt.bar(results.keys(), [v * 100 for v in results.values()],
        color=['steelblue', 'tomato', 'seagreen'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(90, 100)
for i, (name, acc) in enumerate(results.items()):
    plt.text(i, acc * 100 + 0.1, f"{acc*100:.2f}%", ha='center', fontsize=11)
plt.tight_layout()
plt.savefig("plot7_model_comparison.png")
plt.show()
print("[14] Plot 7 saved: plot7_model_comparison.png")


# ── SECTION 9 : CONCLUSION ──────────────────────────────────
best_model = max(results, key=results.get)
print("\n" + "=" * 60)
print("   CONCLUSION")
print("=" * 60)
print(f"""
  Dataset  : Breast Cancer Wisconsin (Kaggle)
  Samples  : {df.shape[0]}  |  Features : {df.shape[1] - 1}
  Target   : Benign vs Malignant

  Models Tested:
    - Logistic Regression : {results['Logistic Regression']*100:.2f}%
    - Decision Tree       : {results['Decision Tree']*100:.2f}%
    - Random Forest       : {results['Random Forest']*100:.2f}%

  Best Model : {best_model} ({results[best_model]*100:.2f}% accuracy)

  Key Findings:
    - Radius, area, and perimeter are strongest predictors.
    - Malignant tumors show significantly larger feature values.
    - All models achieved above 90% accuracy.
""")
print("=" * 60)
print("   PROJECT COMPLETE!")
print("=" * 60)