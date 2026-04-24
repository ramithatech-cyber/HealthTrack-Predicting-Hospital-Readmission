"""
src/model.py
Model training, evaluation, and saving for HealthTrack.
Can be imported or run directly: python src/model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent.parent
MODELS_DIR   = BASE / "models"
GRAPHS_DIR   = BASE / "outputs" / "graphs"
RESULTS_DIR  = BASE / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def build_model() -> RandomForestClassifier:
    """Return a configured Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


def train(clf, X_train, y_train):
    """Fit the classifier."""
    print("Training Random Forest...")
    clf.fit(X_train, y_train)
    print("Training complete.")
    return clf


def evaluate(clf, X_train, X_test, y_test, feature_names: list) -> dict:
    """Evaluate classifier; return metrics dict."""
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cv_acc  = cross_val_score(clf, X_train, y_test[:len(X_train)]
                              if len(X_train) == len(y_test) else y_test,
                              cv=min(5, len(y_test)), scoring="accuracy").mean()
    cm      = confusion_matrix(y_test, y_pred).tolist()
    report  = classification_report(
        y_test, y_pred,
        target_names=["Not Readmitted", "Readmitted"]
    )

    # Cross-val on full data
    from sklearn.model_selection import cross_val_score as cvs
    # We only have test labels here; cv on train handled in train_model.py
    fi = dict(zip(feature_names, clf.feature_importances_.tolist()))

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print(f"\nClassification Report:\n{report}")

    return {
        "accuracy":           round(acc, 4),
        "roc_auc":            round(roc_auc, 4),
        "cv_accuracy":        round(acc, 4),   # placeholder; full CV in train_model.py
        "confusion_matrix":   cm,
        "feature_importance": fi,
        "features":           feature_names,
        "classes":            ["Not Readmitted", "Readmitted"],
    }


def plot_confusion_matrix(cm: list, save: bool = True):
    """Plot and optionally save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Readmitted", "Readmitted"],
                yticklabels=["Not Readmitted", "Readmitted"],
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title("Confusion Matrix", fontsize=10)
    plt.tight_layout()
    if save:
        fig.savefig(GRAPHS_DIR / "confusion_matrix.png", dpi=150)
        print("Saved: confusion_matrix.png")
    return fig


def plot_feature_importance(fi: dict, save: bool = True):
    """Plot feature importances as a horizontal bar chart."""
    nice = {
        "num_prior_visits":   "Prior Visits",
        "age":                "Age",
        "length_of_stay":     "Length of Stay",
        "num_medications":    "No. of Medications",
        "num_comorbidities":  "No. of Comorbidities",
        "diagnosis_enc":      "Diagnosis",
        "discharge_type_enc": "Discharge Type",
        "insurance_enc":      "Insurance",
        "gender_enc":         "Gender",
    }
    items  = sorted(fi.items(), key=lambda x: x[1])
    labels = [nice.get(k, k) for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels, values, color="#4a7fa5", edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance (Random Forest)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8, color="#888")
    plt.tight_layout()
    if save:
        fig.savefig(GRAPHS_DIR / "feature_importance.png", dpi=150)
        print("Saved: feature_importance.png")
    return fig


def save_results_report(metrics: dict, report_text: str):
    """Write classification report to outputs/results/."""
    out = RESULTS_DIR / "classification_report.txt"
    with open(out, "w") as f:
        f.write("HealthTrack – Model Evaluation Report\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Test Accuracy  : {metrics['accuracy']}\n")
        f.write(f"ROC-AUC Score  : {metrics['roc_auc']}\n\n")
        f.write("Classification Report:\n")
        f.write(report_text)
        f.write("\n\nFeature Importances:\n")
        for k, v in sorted(metrics["feature_importance"].items(),
                            key=lambda x: -x[1]):
            f.write(f"  {k:<25} {v:.4f}\n")
    print(f"Results saved → {out}")


def save_model(clf, encoders: dict, meta: dict):
    """Persist model, encoders, and metadata."""
    joblib.dump(clf,      MODELS_DIR / "rf_model.pkl")
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Model saved → models/rf_model.pkl")
    print("Encoders saved → models/encoders.pkl")
    print("Metadata saved → models/model_meta.json")


def load_model():
    """Load saved model, encoders, and metadata."""
    clf      = joblib.load(MODELS_DIR / "rf_model.pkl")
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")
    with open(MODELS_DIR / "model_meta.json") as f:
        meta = json.load(f)
    return clf, encoders, meta


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE))
    from src.preprocessing import (
        load_raw_data, encode_categoricals,
        get_feature_matrix, split_data
    )
    from sklearn.model_selection import cross_val_score

    df                              = load_raw_data()
    df_enc, encoders                = encode_categoricals(df)
    X, y                            = get_feature_matrix(df_enc)
    X_train, X_test, y_train, y_test = split_data(X, y)

    clf     = build_model()
    clf     = train(clf, X_train, y_train)

    # Full cross-validation
    cv_acc  = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    report  = classification_report(y_test, y_pred,
                  target_names=["Not Readmitted", "Readmitted"])
    cm      = confusion_matrix(y_test, y_pred).tolist()
    fi      = dict(zip(list(X.columns), clf.feature_importances_.tolist()))

    meta = {
        "features":           list(X.columns),
        "cat_cols":           ["gender", "diagnosis", "discharge_type", "insurance"],
        "accuracy":           round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":            round(roc_auc_score(y_test, y_proba), 4),
        "cv_accuracy":        round(cv_acc, 4),
        "feature_importance": fi,
        "confusion_matrix":   cm,
        "train_size":         len(X_train),
        "test_size":          len(X_test),
        "classes":            ["Not Readmitted", "Readmitted"],
    }

    save_model(clf, encoders, meta)
    save_results_report(meta, report)
    plot_confusion_matrix(cm)
    plot_feature_importance(fi)
    print("\nDone.")
