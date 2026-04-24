"""
train_model.py
Trains a Random Forest classifier on the synthetic patient dataset.
Saves model + encoders to models/ directory.
Run: python models/train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent.parent
DATA   = BASE / "data" / "patient_data.csv"
MODELS = Path(__file__).parent

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Encode categoricals ───────────────────────────────────────────────────────
CAT_COLS = ["gender", "diagnosis", "discharge_type", "insurance"]
encoders = {}

for col in CAT_COLS:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

# ── Feature matrix ────────────────────────────────────────────────────────────
FEATURES = [
    "age", "num_prior_visits", "length_of_stay",
    "num_medications", "num_comorbidities",
    "gender_enc", "diagnosis_enc", "discharge_type_enc", "insurance_enc"
]

X = df[FEATURES]
y = df["readmitted_30days"]

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train model ───────────────────────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cv_acc  = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()

print(f"\nTest Accuracy  : {acc:.4f}")
print(f"ROC-AUC Score  : {roc_auc:.4f}")
print(f"5-Fold CV Acc  : {cv_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Readmitted","Readmitted"]))

# ── Feature importances ────────────────────────────────────────────────────────
fi = dict(zip(FEATURES, clf.feature_importances_.tolist()))
print("\nFeature importances:")
for k, v in sorted(fi.items(), key=lambda x: -x[1]):
    print(f"  {k:<25} {v:.4f}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred).tolist()

# ── Save artefacts ─────────────────────────────────────────────────────────────
joblib.dump(clf,      MODELS / "rf_model.pkl")
joblib.dump(encoders, MODELS / "encoders.pkl")

meta = {
    "features":          FEATURES,
    "cat_cols":          CAT_COLS,
    "accuracy":          round(acc, 4),
    "roc_auc":           round(roc_auc, 4),
    "cv_accuracy":       round(cv_acc, 4),
    "feature_importance": fi,
    "confusion_matrix":  cm,
    "train_size":        len(X_train),
    "test_size":         len(X_test),
    "classes":           ["Not Readmitted", "Readmitted"],
}
with open(MODELS / "model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved → models/rf_model.pkl")
print("Saved → models/encoders.pkl")
print("Saved → models/model_meta.json")
