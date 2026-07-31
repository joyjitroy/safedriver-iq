r"""Generate PRISM-AR paper-ready figures from the real-data results CSV.

Usage:
    C:\prismar_venv\Scripts\python.exe generate_prism_ar_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results():
    csv_path = os.path.join(OUTPUT_DIR, "prism_ar_results.csv")
    summary_path = os.path.join(OUTPUT_DIR, "prism_ar_summary.json")
    df = pd.read_csv(csv_path)
    with open(summary_path, "r") as f:
        summary = json.load(f)
    return df, summary


def plot_score_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        ax.hist(subset["mean_score"], bins=20, alpha=0.5, label=dataset)
    ax.set_xlabel("Mean PRISM Safety Score")
    ax.set_ylabel("Number of Clips")
    ax.set_title("Distribution of Mean PRISM Safety Scores by Dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "score_distribution.png"), dpi=300)
    plt.close(fig)


def plot_dataset_metrics(df):
    metrics = ["cue_flicker_hz", "visual_clutter"]
    titles = ["Cue Flicker Rate (Hz)", "Visual Clutter (Mean Overlay Opacity)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, metric in enumerate(metrics):
        sns.boxplot(data=df, x="dataset", y=metric, ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Dataset source")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "dataset_metrics.png"), dpi=300)
    plt.close(fig)


def plot_min_distance_vs_score(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        ax.scatter(subset["min_distance_m"], subset["mean_score"], alpha=0.6, label=dataset)
    ax.set_xlabel("Minimum Ego-VRU Distance (m)")
    ax.set_ylabel("Mean PRISM Safety Score")
    ax.set_title("Safety Score vs. Minimum Ego-VRU Distance")
    ax.legend()
    ax.set_xlim(0, 30)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "distance_vs_score.png"), dpi=300)
    plt.close(fig)


def plot_tier_distribution(df):
    tier_rows = []
    for _, row in df.iterrows():
        tiers = eval(row["tier_distribution"]) if isinstance(row["tier_distribution"], str) else row["tier_distribution"]
        for tier, count in tiers.items():
            tier_rows.append({"dataset": row["dataset"], "tier": tier, "count": count})
    tier_df = pd.DataFrame(tier_rows)
    if tier_df.empty:
        return
    tier_counts = tier_df.groupby(["dataset", "tier"]).sum().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=tier_counts, x="tier", y="count", hue="dataset", ax=ax)
    ax.set_title("Tier Distribution by Dataset")
    ax.set_ylabel("Frame Count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "tier_distribution.png"), dpi=300)
    plt.close(fig)


def plot_ground_truth_comparison(df):
    """Bar chart of adaptive vs static accuracy and recall against ground truth."""
    metrics = [
        "adaptive_tier_accuracy", "static_tier_accuracy",
        "adaptive_intervention_recall", "static_intervention_recall",
        "adaptive_emergency_recall", "static_emergency_recall",
    ]
    mean_vals = df[metrics].mean().values
    labels = ["Accuracy\n(Adaptive)", "Accuracy\n(Static)",
              "Intervention\nRecall (Adaptive)", "Intervention\nRecall (Static)",
              "Emergency\nRecall (Adaptive)", "Emergency\nRecall (Static)"]
    colors = ["#2ca02c", "#d62728", "#2ca02c", "#d62728", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, mean_vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Adaptive vs. Static AR: Ground-Truth Tier Matching and High-Risk Recall")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ground_truth_comparison.png"), dpi=300)
    plt.close(fig)


def plot_ground_truth_distribution(df):
    """Ground-truth tier distribution by dataset."""
    tier_rows = []
    for _, row in df.iterrows():
        tiers = eval(row["gt_tier_distribution"]) if isinstance(row["gt_tier_distribution"], str) else row["gt_tier_distribution"]
        for tier, count in tiers.items():
            tier_rows.append({"dataset": row["dataset"], "tier": tier, "count": count})
    tier_df = pd.DataFrame(tier_rows)
    if tier_df.empty:
        return
    tier_counts = tier_df.groupby(["dataset", "tier"]).sum().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=tier_counts, x="tier", y="count", hue="dataset", ax=ax)
    ax.set_title("Ground-Truth Tier Distribution by Dataset")
    ax.set_ylabel("Frame Count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ground_truth_distribution.png"), dpi=300)
    plt.close(fig)


def plot_lead_time_distribution(df):
    """Cumulative distribution of warning lead times."""
    lead_times = df[df["warning_lead_time_s"] > 0]["warning_lead_time_s"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lead_times.values, np.arange(1, len(lead_times) + 1) / len(lead_times), linewidth=2)
    ax.set_xlabel("Warning Lead Time (s)")
    ax.set_ylabel("Cumulative Fraction of Clips")
    ax.set_title("Cumulative Warning Lead-Time Distribution")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "lead_time_distribution.png"), dpi=300)
    plt.close(fig)


def plot_ablation(ablation_df):
    """Bar chart of mean score and static under-warning rate per ablation."""
    ablation_df = ablation_df.groupby("ablation", as_index=False).agg(
        mean_score=("mean_score", "mean"),
        static_under_warning_rate=("static_under_warning_rate", "mean"),
    )
    labels = ablation_df["ablation"].tolist()
    x = np.arange(len(labels))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width / 2, ablation_df["mean_score"], width, label="Mean score", color="steelblue")
    ax1.set_ylabel("Mean PRISM-AR Score", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, ablation_df["static_under_warning_rate"], width, label="Static under-warning", color="coral")
    ax2.set_ylabel("Static Under-Warning Rate", color="coral")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax2.set_ylim(0, 1.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Ablation Study: Score Shift and Static Under-Warning Impact")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "ablation.png"), dpi=300)
    plt.close(fig)


def main():
    df, summary = load_results()
    ablation_df = pd.read_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"))
    print(f"Generating figures for {len(df)} clips...")
    plot_score_distribution(df)
    plot_dataset_metrics(df)
    plot_min_distance_vs_score(df)
    plot_tier_distribution(df)
    plot_ground_truth_comparison(df)
    plot_ground_truth_distribution(df)
    plot_lead_time_distribution(df)
    plot_ablation(ablation_df)
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
