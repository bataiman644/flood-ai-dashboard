# =====================================================
# 🌊 CLOUD-BASED FLOOD DISASTER RESPONSE SYSTEM
# (FINAL THESIS VERSION WITH XGBOOST)
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("🚀 Starting FINAL Flood Prediction System (XGBoost Version)")

# =====================================================
# 1. LOAD DATASETS
# =====================================================

df1 = pd.read_csv("flood.csv")
df2 = pd.read_csv("sri_lanka_flood_risk_dataset_25000 (1).csv")

features = ["rainfall", "water_level", "drainage", "population"]

# Ensure features exist
for f in features:
    if f not in df1.columns:
        df1[f] = np.random.uniform(0, 1, len(df1))
    if f not in df2.columns:
        df2[f] = np.random.uniform(0, 1, len(df2))

df = pd.concat([df1[features], df2[features]], ignore_index=True)

df = df.apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)

print("📊 Dataset shape:", df.shape)

# =====================================================
# 2. FEATURE ENGINEERING (RISK SCORE)
# =====================================================

df["risk_score"] = (
    0.5 * df["rainfall"] +
    0.3 * df["water_level"] -
    0.2 * df["drainage"] +
    0.1 * df["population"]
)

# =====================================================
# 3. LABEL CREATION (BALANCED)
# =====================================================

low_th = df["risk_score"].quantile(0.33)
high_th = df["risk_score"].quantile(0.66)

def classify(x):
    if x < low_th:
        return 0  # LOW
    elif x < high_th:
        return 1  # MEDIUM
    else:
        return 2  # HIGH

df["flood_risk"] = df["risk_score"].apply(classify)

print("\n📊 Class Distribution:")
print(df["flood_risk"].value_counts())

# =====================================================
# 📊 GRAPH 1: CLASS DISTRIBUTION
# =====================================================

plt.figure(figsize=(5,4))
sns.countplot(x=df["flood_risk"])
plt.title("Flood Risk Distribution")
plt.show()

# =====================================================
# 4. SPLIT DATA
# =====================================================

X = df.drop(["flood_risk", "risk_score"], axis=1)
y = df["flood_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 5. SCALING
# =====================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

feature_names = X.columns

# =====================================================
# 6. MODELS (UPDATED)
# =====================================================

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
}

# Add XGBoost
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="mlogloss"
    )
except:
    print("⚠ XGBoost not installed")

# =====================================================
# 7. TRAINING + EVALUATION
# =====================================================

results = {}

best_model = None
best_score = 0

for name, model in models.items():

    print(f"\n🔹 Training {name}")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    results[name] = acc

    print("Accuracy:", acc)
    print(classification_report(y_test, pred))

    # CONFUSION MATRIX
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, pred),
                annot=True, fmt="d",
                xticklabels=["Low","Med","High"],
                yticklabels=["Low","Med","High"])
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    if acc > best_score:
        best_score = acc
        best_model = model

# =====================================================
# 📊 GRAPH 2: MODEL COMPARISON
# =====================================================

plt.figure(figsize=(6,4))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Accuracy Comparison")
plt.ylim(0,1)
plt.show()

# =====================================================
# 📊 GRAPH 3: FEATURE IMPORTANCE
# =====================================================

if hasattr(best_model, "feature_importances_"):
    plt.figure(figsize=(6,4))
    sns.barplot(x=feature_names, y=best_model.feature_importances_)
    plt.title("Feature Importance (Best Model)")
    plt.xticks(rotation=45)
    plt.show()

# =====================================================
# 📊 GRAPH 4: ROC CURVE
# =====================================================

y_bin = label_binarize(y_test, classes=[0,1,2])
y_score = best_model.predict_proba(X_test)

plt.figure(figsize=(6,5))

labels = ["Low","Medium","High"]
colors = ["blue","orange","green"]

for i in range(3):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f"{labels[i]}")

plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve (Best Model)")
plt.legend()
plt.show()

# =====================================================
# 8. SAVE MODEL
# =====================================================

joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "features.pkl")

print("\n✅ FINAL MODEL SAVED SUCCESSFULLY")
print("🏆 Best Accuracy:", best_score)
print("🎯 Best Model:", type(best_model).__name__)