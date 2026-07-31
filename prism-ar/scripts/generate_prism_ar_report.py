"""Generate a consolidated Markdown report with all PRISM-AR tables.

Outputs:
- results/prism_ar_real/report.md
- results/prism_ar_real/tables/*.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)


def main():
    main_df = pd.read_csv(os.path.join(OUTPUT_DIR, "prism_ar_results.csv"))
    ablation_df = pd.read_csv(os.path.join(OUTPUT_DIR, "ablation_results.csv"))
    robustness_df = pd.read_csv(os.path.join(OUTPUT_DIR, "robustness_results.csv"))

    # Table 1: Main results by dataset
    table1 = main_df.groupby("dataset")[
        ["mean_score", "min_distance_m", "warning_lead_time_s",
         "adaptive_under_warning_rate", "adaptive_over_warning_rate",
         "static_under_warning_rate", "static_over_warning_rate",
         "cue_flicker_hz", "visual_clutter",
         "cue_risk_monotonicity", "shap_alignment"]
    ].mean().round(2)
    table1.to_csv(os.path.join(TABLES_DIR, "main_by_dataset.csv"))

    # Table 2: Ablation
    table2 = ablation_df.groupby("ablation")[
        ["mean_score", "warning_lead_time_s",
         "adaptive_under_warning_rate", "adaptive_over_warning_rate",
         "static_under_warning_rate", "static_over_warning_rate",
         "cue_flicker_hz", "visual_clutter"]
    ].mean().round(2)
    table2.to_csv(os.path.join(TABLES_DIR, "ablation.csv"))

    # Table 3: Robustness
    table3 = robustness_df.groupby("condition")[
        ["mean_score", "warning_lead_time_s",
         "adaptive_under_warning_rate", "adaptive_over_warning_rate",
         "static_under_warning_rate", "static_over_warning_rate",
         "cue_flicker_hz", "visual_clutter"]
    ].mean().round(2)
    table3.to_csv(os.path.join(TABLES_DIR, "robustness.csv"))

    # Table 4: Ground-truth comparison
    gt_cols = [
        "adaptive_tier_accuracy", "static_tier_accuracy",
        "adaptive_intervention_recall", "static_intervention_recall",
        "adaptive_emergency_recall", "static_emergency_recall",
    ]
    table4 = main_df.groupby("dataset")[gt_cols].mean().round(2)
    table4.to_csv(os.path.join(TABLES_DIR, "ground_truth_comparison.csv"))

    # Table 5: Adverse-condition escalation
    # Flag adverse conditions: night/dusk/dawn or rain/snow/fog
    def is_adverse(row):
        lighting = str(row["lighting"]).lower()
        weather = str(row["weather"]).lower()
        adverse_light = any(x in lighting for x in ["dark", "dusk", "dawn", "night"])
        adverse_weather = any(x in weather for x in ["rain", "snow", "fog", "sleet"])
        return adverse_light or adverse_weather

    main_df["adverse_condition"] = main_df.apply(is_adverse, axis=1)
    table5 = main_df.groupby("adverse_condition")[
        ["mean_score", "adaptive_under_warning_rate", "adaptive_intervention_recall",
         "adaptive_emergency_recall", "cue_risk_monotonicity", "shap_alignment"]
    ].mean().round(2)
    table5.index = table5.index.map({True: "adverse", False: "clean"})
    table5.to_csv(os.path.join(TABLES_DIR, "adverse_condition.csv"))

    # Statistical significance: paired Wilcoxon test on tier accuracy and intervention recall
    def wilcoxon_note(a, b, label):
        diff = np.array(a) - np.array(b)
        if np.all(diff == 0):
            return f"{label}: no difference (p = 1.00)"
        try:
            stat, p = stats.wilcoxon(a, b, zero_method="zsplit")
            return f"{label}: adaptive mean={np.mean(a):.2f}, static mean={np.mean(b):.2f}, p={p:.4f}"
        except Exception as e:
            return f"{label}: could not compute ({e})"

    sig_lines = [
        wilcoxon_note(main_df["adaptive_tier_accuracy"], main_df["static_tier_accuracy"], "Tier accuracy"),
        wilcoxon_note(main_df["adaptive_intervention_recall"], main_df["static_intervention_recall"], "Intervention recall"),
        wilcoxon_note(main_df["adaptive_emergency_recall"], main_df["static_emergency_recall"], "Emergency recall"),
    ]

    # Ground-truth tier distribution
    import ast
    gt_total = {"silent": 0, "advisory": 0, "intervention": 0, "emergency": 0}
    for td in main_df["gt_tier_distribution"]:
        d = ast.literal_eval(td) if isinstance(td, str) else td
        for k, v in d.items():
            gt_total[k] = gt_total.get(k, 0) + v
    total_gt = sum(gt_total.values())
    gt_pct = {k: 100.0 * v / total_gt for k, v in gt_total.items()}

    # Markdown report
    lines = [
        "# PRISM-AR Real-Data Evaluation Report",
        "",
        f"Total clips: {len(main_df)}",
        "",
        "## 1. Main Results by Dataset",
        "",
        table1.to_markdown(),
        "",
        "## 2. Ablation Study",
        "",
        table2.to_markdown(),
        "",
        "## 3. Robustness Study",
        "",
        table3.to_markdown(),
        "",
        "## 4. Ground-Truth Tier Comparison",
        "",
        "Ground-truth tiers are derived from per-frame TTC and distance thresholds:",
        "emergency (distance < 1m or TTC < 0.5s), intervention (< 2m or < 1.0s),",
        "advisory (< 5m or < 2.5s), silent otherwise.",
        "",
        table4.to_markdown(),
        "",
        "### Ground-truth tier distribution",
        "",
        f"- silent: {gt_total['silent']} ({gt_pct['silent']:.1f}%)",
        f"- advisory: {gt_total['advisory']} ({gt_pct['advisory']:.1f}%)",
        f"- intervention: {gt_total['intervention']} ({gt_pct['intervention']:.1f}%)",
        f"- emergency: {gt_total['emergency']} ({gt_pct['emergency']:.1f}%)",
        "",
        "## 5. Statistical Significance (Adaptive vs. Static)",
        "",
    ]
    lines.extend([f"- {s}" for s in sig_lines])
    lines.extend([
        "",
        "## 6. Cue-Risk Monotonicity and SHAP Alignment",
        "",
        "- **Cue-risk monotonicity** measures whether AR cue intensity increases as the PRISM safety score decreases (Spearman rho). More negative values indicate stronger monotonic escalation.",
        "- **SHAP alignment** measures the fraction of high-risk frames where the dominant risk component (environmental, trajectory, or VRU proximity) matches the explainable top factor selected by the PRISM engine.",
        "",
        f"Mean cue-risk monotonicity: {main_df['cue_risk_monotonicity'].mean():.3f}",
        f"Mean SHAP alignment: {main_df['shap_alignment'].mean():.3f}",
        "",
        "## 7. Adverse-Condition Escalation",
        "",
        table5.to_markdown(),
        "",
        "## 8. Runtime / Latency",
        "",
        "Reference implementation latency per frame (measured on a representative subset of clips):",
        "",
    ])

    # Add latency numbers if available
    latency_path = os.path.join(OUTPUT_DIR, "latency_results.json")
    if os.path.exists(latency_path):
        import json
        with open(latency_path, "r") as f:
            latency = json.load(f)
        for k, v in latency["mean_ms_per_frame"].items():
            lines.append(f"- {k}: {v:.4f} ms/frame")
    else:
        lines.append("- Latency results not yet computed.")

    lines.extend([
        "",
        "## 9. Output Files",
        "",
        "- `prism_ar_results.csv`: per-clip results",
        "- `ablation_results.csv`: ablation results",
        "- `robustness_results.csv`: robustness results",
        "- `figures/`: score distribution, dataset metrics, tier distribution, distance-vs-score,",
        "  ground-truth comparison, ground-truth distribution, lead-time distribution, ablation",
        "- `tables/`: main results, ablation, robustness, ground-truth comparison",
        "- `images/`: sample paired overlays (no-AR, static, adaptive, oracle)",
        "",
    ])

    report_path = os.path.join(OUTPUT_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
