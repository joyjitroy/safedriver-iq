# PRISM-AR Real-Data Evaluation Report

Total clips: 231

## 1. Main Results by Dataset

| dataset            |   mean_score |   min_distance_m |   warning_lead_time_s |   adaptive_under_warning_rate |   adaptive_over_warning_rate |   static_under_warning_rate |   static_over_warning_rate |   cue_flicker_hz |   visual_clutter |   cue_risk_monotonicity |   shap_alignment |
|:-------------------|-------------:|-----------------:|----------------------:|------------------------------:|-----------------------------:|----------------------------:|---------------------------:|-----------------:|-----------------:|------------------------:|-----------------:|
| argoverse          |        66.69 |             7.04 |                  0    |                             0 |                            0 |                        0    |                          0 |             0.51 |             0.18 |                   -0.65 |             0    |
| nuscenes           |        65.38 |             5.99 |                  0.85 |                             0 |                            0 |                        0.19 |                          0 |             0.34 |             0.19 |                   -0.63 |             0.06 |
| prism_ar_synthetic |        47.09 |             0.39 |                  2.38 |                             0 |                            0 |                        1    |                          0 |             1.09 |             0.44 |                   -0.88 |             0.8  |
| waymo              |        68.47 |             6.08 |                  1.28 |                             0 |                            0 |                        0.32 |                          0 |             1.16 |             0.16 |                   -0.66 |             0.16 |

## 2. Ablation Study

| ablation      |   mean_score |   warning_lead_time_s |   adaptive_under_warning_rate |   adaptive_over_warning_rate |   static_under_warning_rate |   static_over_warning_rate |   cue_flicker_hz |   visual_clutter |
|:--------------|-------------:|----------------------:|------------------------------:|-----------------------------:|----------------------------:|---------------------------:|-----------------:|-----------------:|
| full          |        62.37 |                  1.02 |                             0 |                            0 |                        0.32 |                          0 |             0.5  |             0.23 |
| no_trajectory |        65.77 |                  0.48 |                             0 |                            0 |                        0.23 |                          0 |             0.32 |             0.22 |
| no_vru        |        52.49 |                  3.83 |                             0 |                            0 |                        0.79 |                          0 |             0.55 |             0.38 |
| no_weather    |        64.23 |                  0.58 |                             0 |                            0 |                        0.25 |                          0 |             0.51 |             0.2  |

## 3. Robustness Study

| condition   |   mean_score |   warning_lead_time_s |   adaptive_under_warning_rate |   adaptive_over_warning_rate |   static_under_warning_rate |   static_over_warning_rate |   cue_flicker_hz |   visual_clutter |
|:------------|-------------:|----------------------:|------------------------------:|-----------------------------:|----------------------------:|---------------------------:|-----------------:|-----------------:|
| clean       |        62.66 |                  1    |                             0 |                            0 |                        0.29 |                          0 |             0.47 |             0.22 |
| delay_10    |        62.43 |                  0.68 |                             0 |                            0 |                        0.26 |                          0 |             0.54 |             0.23 |
| delay_5     |        61.59 |                  0.91 |                             0 |                            0 |                        0.29 |                          0 |             0.5  |             0.24 |
| drop_0.1    |        62.55 |                  0.98 |                             0 |                            0 |                        0.28 |                          0 |             0.45 |             0.22 |
| drop_0.3    |        62.46 |                  0.87 |                             0 |                            0 |                        0.29 |                          0 |             0.4  |             0.23 |
| noise_0.1   |        62.62 |                  1    |                             0 |                            0 |                        0.29 |                          0 |             0.46 |             0.22 |
| noise_0.5   |        62.41 |                  1.12 |                             0 |                            0 |                        0.31 |                          0 |             0.5  |             0.23 |

## 4. Ground-Truth Tier Comparison

Ground-truth tiers are derived from per-frame TTC and distance thresholds:
emergency (distance < 1m or TTC < 0.5s), intervention (< 2m or < 1.0s),
advisory (< 5m or < 2.5s), silent otherwise.

| dataset            |   adaptive_tier_accuracy |   static_tier_accuracy |   adaptive_intervention_recall |   static_intervention_recall |   adaptive_emergency_recall |   static_emergency_recall |
|:-------------------|-------------------------:|-----------------------:|-------------------------------:|-----------------------------:|----------------------------:|--------------------------:|
| argoverse          |                     0.69 |                   0.54 |                           0    |                            0 |                        0    |                         0 |
| nuscenes           |                     0.75 |                   0.36 |                           0.13 |                            0 |                        0    |                         0 |
| prism_ar_synthetic |                     0.61 |                   0.48 |                           0.46 |                            0 |                        0.35 |                         0 |
| waymo              |                     0.84 |                   0.31 |                           0.21 |                            0 |                        0    |                         0 |

### Ground-truth tier distribution

- silent: 3158 (32.7%)
- advisory: 4380 (45.4%)
- intervention: 1478 (15.3%)
- emergency: 629 (6.5%)

## 5. Statistical Significance (Adaptive vs. Static)

- Tier accuracy: adaptive mean=0.71, static mean=0.43, p=0.0000
- Intervention recall: adaptive mean=0.20, static mean=0.00, p=0.0000
- Emergency recall: adaptive mean=0.09, static mean=0.00, p=0.0000

## 6. Cue-Risk Monotonicity and SHAP Alignment

- **Cue-risk monotonicity** measures whether AR cue intensity increases as the PRISM safety score decreases (Spearman rho). More negative values indicate stronger monotonic escalation.
- **SHAP alignment** measures the fraction of high-risk frames where the dominant risk component (environmental, trajectory, or VRU proximity) matches the explainable top factor selected by the PRISM engine.

Mean cue-risk monotonicity: -0.703
Mean SHAP alignment: 0.251

## 7. Adverse-Condition Escalation

| adverse_condition   |   mean_score |   adaptive_under_warning_rate |   adaptive_intervention_recall |   adaptive_emergency_recall |   cue_risk_monotonicity |   shap_alignment |
|:--------------------|-------------:|------------------------------:|-------------------------------:|----------------------------:|------------------------:|-----------------:|
| clean               |        65.35 |                             0 |                           0.03 |                        0    |                    -0.7 |             0.16 |
| adverse             |        50.54 |                             0 |                           0.62 |                        0.33 |                    -0.7 |             0.48 |

## 8. Runtime / Latency

Reference implementation latency per frame (measured on a representative subset of clips):

- environmental: 0.0001 ms/frame
- trajectory: 0.0011 ms/frame
- vru: 0.0005 ms/frame
- fusion: 0.0024 ms/frame
- cue_mapping: 0.0001 ms/frame
- total: 0.0042 ms/frame

## 9. Output Files

- `prism_ar_results.csv`: per-clip results
- `ablation_results.csv`: ablation results
- `robustness_results.csv`: robustness results
- `figures/`: score distribution, dataset metrics, tier distribution, distance-vs-score,
  ground-truth comparison, ground-truth distribution, lead-time distribution, ablation
- `tables/`: main results, ablation, robustness, ground-truth comparison
- `images/`: sample paired overlays (no-AR, static, adaptive, oracle)
