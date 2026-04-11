# =====================================================
# 🌊 CLOUD-BASED FLOOD DISASTER RESPONSE SYSTEM
# FINAL COMPLETE CODE (THESIS READY)
# =====================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

from imblearn.over_sampling import SMOTE

print("🚀 Starting Multi-Dataset Flood System...")

# =====================================================
# 1. LOAD DATASETS
# =====================================================
df1 = pd.read_csv("data/flood.csv")
df2 = pd.read_csv("data/sri_lanka_flood_risk_dataset_25000.csv")
df3 = pd.read_csv("data/flood_dataset_classification.csv")
df4 = pd.read_csv("data/modis_flood_features_paling cleaning (1).csv")

print("✅ All datasets loaded")

# =====================================================
# 2. FEATURE MAPPING (COMMON FORMAT)
# =====================================================
df1["rainfall"] = df1["MonsoonIntensity"]
df1["water_level"] = df1["RiverManagement"]
df1["drainage"] = df1["DrainageSystems"]
df1["population"] = df1["PopulationScore"]
df1["flood_risk"] = df1["FloodProbability"]

df2["rainfall"] = df2["rainfall_7d_mm"]
df2["water_level"] = df2["distance_to_river_m"]
df2["drainage"] = df2["drainage_index"]
df2["population"] = df2["population_density_per_km2"]
df2["flood_risk"] = df2["flood_risk_score"]

df3["rainfall"] = df3.iloc[:, 0]
df3["water_level"] = df3.iloc[:, 1]
df3["drainage"] = df3.iloc[:, 2]
df3["population"] = df3.iloc[:, 3]
df3["flood_risk"] = df3.iloc[:, -1]

df4["rainfall"] = df4.iloc[:, 0]
df4["water_level"] = df4.iloc[:, 1]
df4["drainage"] = df4.iloc[:, 2]
df4["population"] = df4.iloc[:, 3]
df4["flood_risk"] = df4.iloc[:, -1]

cols = ["rainfall", "water_level", "drainage", "population", "flood_risk"]

df = pd.concat([df1[cols], df2[cols], df3[cols], df4[cols]], ignore_index=True)

print("📊 Combined Dataset Shape:", df.shape)

# =====================================================
# 3. DATA CLEANING
# =====================================================
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

print("🧹 Data cleaned:", df.shape)

# =====================================================
# 4. SAFE SAMPLING
# =====================================================
if len(df) > 200000:
    df = df.sample(200000, random_state=42)
    print("⚡ Sampled to 200k rows")
else:
    print("⚡ Using full dataset")

# =====================================================
# 5. TARGET CLASSIFICATION
# =====================================================
def classify(x):
    if x < 0.4:
        return 0   # Low
    elif x < 0.7:
        return 1   # Medium
    else:
        return 2   # High

df["flood_risk"] = df["flood_risk"].apply(classify)

# =====================================================
# 6. SPLIT DATA
# =====================================================
X = df.drop("flood_risk", axis=1)
y = df["flood_risk"]

features = X.columns

# 📊 CLASS DISTRIBUTION
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Class Distribution (0=Low, 1=Medium, 2=High)")
plt.xlabel("Flood Risk Class")
plt.ylabel("Count")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# 7. SCALING
# =====================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# 8. SMOTE (FIXED)
# =====================================================
min_class = y_train.value_counts().min()

if min_class > 5:
    smote = SMOTE(k_neighbors=2, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("⚖ SMOTE applied")
else:
    print("⚠ SMOTE skipped (not enough samples)")

# =====================================================
# 9. MODELS
# =====================================================
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

lr = LogisticRegression(max_iter=500)
dt = DecisionTreeClassifier()

models = {
    "Random Forest": rf,
    "Logistic Regression": lr,
    "Decision Tree": dt
}

accuracy_scores = {}

# =====================================================
# 10. TRAIN + EVALUATE
# =====================================================
labels = ["Low", "Medium", "High"]

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    accuracy_scores[name] = acc

    print(f"\n📊 {name} Accuracy:", acc)
    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    annot = np.empty_like(cm).astype(str)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i,j]}\n({cm_percent[i,j]:.2f})"

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_percent, annot=annot, fmt="", cmap="Blues",
                xticklabels=labels, yticklabels=labels)

    plt.title(f"{name} Confusion Matrix (Count + %)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# =====================================================
# 11. ACCURACY COMPARISON GRAPH
# =====================================================
plt.figure(figsize=(6,4))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
plt.title("Model Accuracy Comparison")
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.show()

# =====================================================
# 12. FINAL MODEL
# =====================================================
rf.fit(X_train, y_train)

# =====================================================
# 13. FEATURE IMPORTANCE
# =====================================================
importance = pd.Series(rf.feature_importances_, index=features)
importance = importance.sort_values()

plt.figure(figsize=(6,4))
importance.plot(kind='barh')
plt.title("Feature Importance (Flood Prediction)")
plt.xlabel("Importance Score")
plt.show()

# =====================================================
# 14. MULTICLASS ROC CURVE
# =====================================================
y_bin = label_binarize(y_test, classes=[0,1,2])
y_score = rf.predict_proba(X_test)

plt.figure(figsize=(6,5))

colors = ["blue", "orange", "green"]
class_names = ["Low", "Medium", "High"]

for i in range(3):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, color=colors[i],
             label=f"{class_names[i]} (AUC={auc(fpr,tpr):.2f})")

plt.plot([0,1],[0,1],'k--')
plt.title("Multiclass ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

# =====================================================
# 15. PREDICTION DISTRIBUTION
# =====================================================
final_pred = rf.predict(X_test)

plt.figure(figsize=(6,4))
sns.countplot(x=final_pred)
plt.title("Prediction Distribution (Random Forest)")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.show()

# =====================================================
# 16. SAVE MODEL
# =====================================================
joblib.dump(rf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "features.pkl")

print("✅ FINAL MODEL SAVED SUCCESSFULLY")