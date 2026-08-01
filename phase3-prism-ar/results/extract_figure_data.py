"""Extract data behind all 8 PRISM-AR figures into an Excel workbook.

Each sheet corresponds to one figure with the exact data used to generate it.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = BASE_DIR
OUTPUT_EXCEL = os.path.join(RESULTS_DIR, "figure_data.xlsx")


def load_data():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "prism_ar_results.csv"))
    ablation_df = pd.read_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"))
    with open(os.path.join(RESULTS_DIR, "prism_ar_summary.json"), "r") as f:
        summary = json.load(f)
    return df, ablation_df, summary


def fig1_score_distribution(df):
    """Raw data for histogram: scene_id, dataset, mean_score."""
    return df[["scene_id", "dataset", "mean_score"]].copy()


def fig2_tier_distribution(df):
    """Frame counts per tier per dataset (PRISM-assigned)."""
    tier_rows = []
    for _, row in df.iterrows():
        tiers = eval(row["tier_distribution"]) if isinstance(row["tier_distribution"], str) else row["tier_distribution"]
        for tier, count in tiers.items():
            tier_rows.append({"dataset": row["dataset"], "tier": tier, "frame_count": count})
    tier_df = pd.DataFrame(tier_rows)
    return tier_df.groupby(["dataset", "tier"]).sum().reset_index()


def fig3_ground_truth_distribution(df):
    """Frame counts per ground-truth tier per dataset."""
    tier_rows = []
    for _, row in df.iterrows():
        tiers = eval(row["gt_tier_distribution"]) if isinstance(row["gt_tier_distribution"], str) else row["gt_tier_distribution"]
        for tier, count in tiers.items():
            tier_rows.append({"dataset": row["dataset"], "tier": tier, "frame_count": count})
    tier_df = pd.DataFrame(tier_rows)
    return tier_df.groupby(["dataset", "tier"]).sum().reset_index()


def fig4_ground_truth_comparison(df):
    """Mean adaptive vs static accuracy and recall metrics."""
    metrics = [
        "adaptive_tier_accuracy", "static_tier_accuracy",
        "adaptive_intervention_recall", "static_intervention_recall",
        "adaptive_emergency_recall", "static_emergency_recall",
    ]
    means = df[metrics].mean()
    stds = df[metrics].std()
    result = pd.DataFrame({"metric": metrics, "mean": means.values, "std": stds.values})
    return result


def fig5_distance_vs_score(df):
    """Scatter data: scene_id, dataset, min_distance_m, mean_score."""
    return df[["scene_id", "dataset", "min_distance_m", "mean_score"]].copy()


def fig6_lead_time_distribution(df):
    """CDF data: sorted lead times for clips with lead_time > 0."""
    lead_times = df[df["warning_lead_time_s"] > 0]["warning_lead_time_s"].sort_values().reset_index(drop=True)
    cdf = pd.DataFrame({
        "warning_lead_time_s": lead_times.values,
        "cumulative_fraction": np.arange(1, len(lead_times) + 1) / len(lead_times),
    })
    return cdf


def fig7_dataset_metrics(df):
    """Boxplot data: per-scenario metrics by dataset."""
    metrics = ["mean_score", "under_warning_rate", "over_warning_rate",
               "warning_lead_time_s", "cue_flicker_hz", "visual_clutter"]
    return df[["scene_id", "dataset"] + metrics].copy()


def fig8_ablation(ablation_df):
    """Ablation bar chart data: mean score and static under-warning per ablation condition."""
    abl_groups = ablation_df.groupby("ablation").agg(
        mean_score=("mean_score", "mean"),
        static_under_warning_rate=("static_under_warning_rate", "mean"),
        n_scenarios=("scene_id", "count"),
    ).reset_index()
    return abl_groups


def main():
    df, ablation_df, summary = load_data()

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        fig1_score_distribution(df).to_excel(writer, sheet_name="1_score_distribution", index=False)
        fig2_tier_distribution(df).to_excel(writer, sheet_name="2_tier_distribution", index=False)
        fig3_ground_truth_distribution(df).to_excel(writer, sheet_name="3_gt_tier_distribution", index=False)
        fig4_ground_truth_comparison(df).to_excel(writer, sheet_name="4_gt_comparison", index=False)
        fig5_distance_vs_score(df).to_excel(writer, sheet_name="5_distance_vs_score", index=False)
        fig6_lead_time_distribution(df).to_excel(writer, sheet_name="6_lead_time_cdf", index=False)
        fig7_dataset_metrics(df).to_excel(writer, sheet_name="7_dataset_metrics", index=False)
        fig8_ablation(ablation_df).to_excel(writer, sheet_name="8_ablation", index=False)

        # Also add a summary sheet
        summary_df = pd.DataFrame([
            {"key": k, "value": str(v)} for k, v in summary.items()
        ])
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"Saved figure data to: {OUTPUT_EXCEL}")
    print(f"Sheets: 8 figure sheets + 1 summary")


if __name__ == "__main__":
    main()
