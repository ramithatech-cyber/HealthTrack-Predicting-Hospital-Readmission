"""
src/analysis.py
Statistical analysis and EDA functions for HealthTrack.
Generates summary statistics and insight charts saved to outputs/graphs/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Output path ───────────────────────────────────────────────────────────────
GRAPHS_DIR = Path(__file__).parent.parent / "outputs" / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=0.95)
BLUE = "#4a7fa5"
RED  = "#b94a48"


# ── Summary statistics ────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    """Print basic dataset summary."""
    print("=" * 50)
    print(f"Dataset Shape   : {df.shape}")
    print(f"Total Patients  : {len(df)}")
    print(f"Readmitted      : {df['readmitted_30days'].sum()} ({df['readmitted_30days'].mean()*100:.1f}%)")
    print(f"Not Readmitted  : {(df['readmitted_30days'] == 0).sum()}")
    print("\nNumerical Summary:")
    print(df[["age", "num_prior_visits", "length_of_stay",
              "num_medications", "num_comorbidities"]].describe().round(2))


def readmission_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return readmission rate grouped by a categorical column."""
    result = df.groupby(group_col)["readmitted_30days"].agg(
        total="count",
        readmitted="sum"
    )
    result["rate_%"] = (result["readmitted"] / result["total"] * 100).round(1)
    return result.sort_values("rate_%", ascending=False)


# ── Charts ────────────────────────────────────────────────────────────────────
def plot_readmission_distribution(df: pd.DataFrame, save: bool = True):
    counts = df["readmitted_30days"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Not Readmitted", "Readmitted"], counts.values,
           color=[BLUE, RED], width=0.45, edgecolor="white")
    ax.set_title("Readmission Distribution", fontsize=11, pad=8)
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 15, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    if save:
        fig.savefig(GRAPHS_DIR / "readmission_distribution.png", dpi=150)
        print("Saved: readmission_distribution.png")
    return fig


def plot_age_distribution(df: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, val, color in [("Not Readmitted", 0, BLUE), ("Readmitted", 1, RED)]:
        ax.hist(df[df["readmitted_30days"] == val]["age"],
                bins=20, alpha=0.6, label=label, color=color, edgecolor="white")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution by Readmission", fontsize=11, pad=8)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(GRAPHS_DIR / "age_distribution.png", dpi=150)
        print("Saved: age_distribution.png")
    return fig


def plot_diagnosis_readmission_rate(df: pd.DataFrame, save: bool = True):
    grp = df.groupby("diagnosis")["readmitted_30days"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(grp)), grp.values * 100, color=RED, alpha=0.75, edgecolor="white")
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(grp.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Readmission Rate (%)")
    ax.set_title("Readmission Rate by Diagnosis", fontsize=11, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(GRAPHS_DIR / "diagnosis_readmission.png", dpi=150)
        print("Saved: diagnosis_readmission.png")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True):
    num_cols = ["age", "num_prior_visits", "length_of_stay",
                "num_medications", "num_comorbidities", "readmitted_30days"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=11, pad=8)
    plt.tight_layout()
    if save:
        fig.savefig(GRAPHS_DIR / "correlation_heatmap.png", dpi=150)
        print("Saved: correlation_heatmap.png")
    return fig


def plot_prior_visits_boxplot(df: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    data = [
        df[df["readmitted_30days"] == 0]["num_prior_visits"].values,
        df[df["readmitted_30days"] == 1]["num_prior_visits"].values,
    ]
    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(BLUE)
    bp["boxes"][1].set_facecolor(RED)
    ax.set_xticklabels(["Not Readmitted", "Readmitted"])
    ax.set_ylabel("Prior Visits")
    ax.set_title("Prior Visits vs Readmission", fontsize=11, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(GRAPHS_DIR / "prior_visits_boxplot.png", dpi=150)
        print("Saved: prior_visits_boxplot.png")
    return fig


def run_full_analysis(df: pd.DataFrame):
    """Run all analysis functions."""
    print_summary(df)
    print("\nReadmission rate by Diagnosis:")
    print(readmission_by_group(df, "diagnosis"))
    print("\nReadmission rate by Discharge Type:")
    print(readmission_by_group(df, "discharge_type"))

    print("\nGenerating charts...")
    plot_readmission_distribution(df)
    plot_age_distribution(df)
    plot_diagnosis_readmission_rate(df)
    plot_correlation_heatmap(df)
    plot_prior_visits_boxplot(df)
    print(f"\nAll charts saved to: {GRAPHS_DIR}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.preprocessing import load_raw_data
    df = load_raw_data()
    run_full_analysis(df)
