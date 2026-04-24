"""
utils/helpers.py
Shared helpers: model loading, prediction, chart generation.
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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
MODELS   = BASE / "models"
DATA_CSV = BASE / "data" / "patient_data.csv"

# ── Seaborn base style (plain, academic) ──────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)
BLUE  = "#4a7fa5"
RED   = "#b94a48"
GRAY  = "#888888"


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    """Load trained RF model, label encoders, and metadata."""
    clf      = joblib.load(MODELS / "rf_model.pkl")
    encoders = joblib.load(MODELS / "encoders.pkl")
    with open(MODELS / "model_meta.json") as f:
        meta = json.load(f)
    return clf, encoders, meta


def load_data():
    """Load the original training dataset."""
    return pd.read_csv(DATA_CSV)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(clf, encoders, patient: dict) -> tuple[str, float]:
    """
    Given a dict of raw patient inputs, encode and predict.
    Returns (risk_label, probability).
    """
    row = {}
    cat_cols = ["gender", "diagnosis", "discharge_type", "insurance"]
    num_cols = ["age", "num_prior_visits", "length_of_stay",
                "num_medications", "num_comorbidities"]

    for col in num_cols:
        row[col] = patient[col]

    for col in cat_cols:
        le  = encoders[col]
        val = patient[col]
        if val in le.classes_:
            row[col + "_enc"] = le.transform([val])[0]
        else:
            row[col + "_enc"] = 0   # fallback for unseen label

    features = [
        "age", "num_prior_visits", "length_of_stay",
        "num_medications", "num_comorbidities",
        "gender_enc", "diagnosis_enc", "discharge_type_enc", "insurance_enc"
    ]
    X    = pd.DataFrame([row])[features]
    prob = clf.predict_proba(X)[0][1]

    if prob < 0.35:
        risk = "Low"
    elif prob < 0.65:
        risk = "Medium"
    else:
        risk = "High"

    return risk, prob


# ── Charts ────────────────────────────────────────────────────────────────────
def chart_readmission_distribution(df: pd.DataFrame):
    """Bar chart: readmitted vs not readmitted."""
    counts = df["readmitted_30days"].value_counts().sort_index()
    labels = ["Not Readmitted", "Readmitted (≤30 days)"]
    colors = [BLUE, RED]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(labels, counts.values, color=colors, width=0.45, edgecolor="white")
    ax.set_ylabel("Patient Count", fontsize=9)
    ax.set_title("Readmission Distribution in Training Data", fontsize=10, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 15,
                str(val), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig


def chart_feature_importance(meta: dict):
    """Horizontal bar chart of feature importances."""
    fi   = meta["feature_importance"]
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
    items = sorted(fi.items(), key=lambda x: x[1])
    labels = [nice.get(k, k) for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.barh(labels, values, color=BLUE, edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.set_title("Feature Importance (Random Forest)", fontsize=10, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8, color=GRAY)
    plt.tight_layout()
    return fig


def chart_age_readmission(df: pd.DataFrame):
    """Histogram of age grouped by readmission status."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for label, val, color in [("Not Readmitted", 0, BLUE), ("Readmitted", 1, RED)]:
        ax.hist(df[df["readmitted_30days"] == val]["age"],
                bins=20, alpha=0.6, label=label, color=color, edgecolor="white")
    ax.set_xlabel("Age", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Age Distribution by Readmission Status", fontsize=10, pad=8)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def chart_readmission_by_diagnosis(df: pd.DataFrame):
    """Stacked bar: readmission rate per diagnosis."""
    grp   = df.groupby("diagnosis")["readmitted_30days"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(grp)), grp.values * 100, color=RED, alpha=0.75, edgecolor="white")
    ax.set_ylabel("Readmission Rate (%)", fontsize=9)
    ax.set_title("Readmission Rate by Diagnosis", fontsize=10, pad=8)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(grp.index, rotation=30, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def chart_confusion_matrix(meta: dict):
    """Heatmap of the confusion matrix from test evaluation."""
    cm  = np.array(meta["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Readmitted", "Readmitted"],
                yticklabels=["Not Readmitted", "Readmitted"],
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title("Confusion Matrix (Test Set)", fontsize=10, pad=8)
    plt.tight_layout()
    return fig


def chart_visits_vs_readmission(df: pd.DataFrame):
    """Box plot: prior visits by readmission status."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    data = [
        df[df["readmitted_30days"] == 0]["num_prior_visits"].values,
        df[df["readmitted_30days"] == 1]["num_prior_visits"].values,
    ]
    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(BLUE)
    bp["boxes"][1].set_facecolor(RED)
    ax.set_xticklabels(["Not Readmitted", "Readmitted"], fontsize=9)
    ax.set_ylabel("Prior Visits", fontsize=9)
    ax.set_title("Prior Visits vs Readmission", fontsize=10, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig
