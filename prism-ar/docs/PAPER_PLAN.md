# PRISM-AR — IEEE TVT Paper Plan

**Target Journal:** IEEE Transactions on Vehicular Technology (TVT)  
**Submission Track:** Special Issue on Advanced Driving Intelligence for Autonomous Vehicles  
**Manuscript Deadline:** 1 August 2026  
**Target Length:** 12 pages (Regular paper limit for SI)

---

## Title

**PRISM-AR: Explainable Risk-Adaptive Driving Intelligence for Vulnerable Road User Communication in Autonomous Vehicles**

Alternative: *SafeDriver-VRU: Risk-Adaptive External Communication for Autonomous Vehicles Using Inverse Crash Probability and Agentic Risk Reasoning*

---

## Positioning Statement

PRISM-AR extends SafeDriver-IQ and PRISM into an explainable, risk-adaptive autonomous driving intelligence framework for complex VRU scenarios. SafeDriver-IQ provides the inverse crash-probability safety score. PRISM fuses environmental, trajectory, and VRU-interaction risks through an agentic DQN + SHAP reasoning layer. PRISM-AR adds the final external communication decision layer: it converts the AV’s internal safety intelligence into adaptive pedestrian/cyclist-facing cues.

This is **not** an AR visualization paper. It is an AV intelligence and intervention-policy paper where the selected intervention happens to be external VRU communication.

---

## Section Outline

| # | Section | Est. Pages | Status |
|---|---|---|---|
| — | Abstract | 0.2 | ❌ Pending |
| 1 | Introduction | 0.6 | ❌ Pending |
| 2 | Related Work | 0.8 | ❌ Pending |
| 3 | System Architecture | 2.75 | ❌ Pending |
| &nbsp;&nbsp;&nbsp;3.1 | SafeDriver-IQ inverse crash-probability foundation | 0.5 | |
| &nbsp;&nbsp;&nbsp;3.2 | PRISM multi-model risk reasoning (env + trajectory + VRU) | 1.0 | |
| &nbsp;&nbsp;&nbsp;3.3 | PRISM-AR external communication policy | 0.75 | |
| &nbsp;&nbsp;&nbsp;3.4 | Static eHMI baseline and oracle upper bound | 0.5 | |
| 4 | Dataset and Scenario Extraction | 0.6 | ❌ Pending |
| 5 | Evaluation Metrics and Validation Protocol | 0.5 | ❌ Pending |
| 6 | Results and Discussion | 2.5 | ❌ Pending |
| 7 | Ablation Study | 0.75 | ❌ Pending |
| 8 | Robustness Analysis | 0.5 | ❌ Pending |
| 9 | Limitations | 0.4 | ❌ Pending |
| 10 | Conclusion | 0.25 | ❌ Pending |
| — | Author Biographies | 0.75 | ✅ User has |
| — | References | 0.75 | ❌ Pending |
| | **Total** | **~11.7 pages** | |

---

## Figures (6)

| # | Label | Figure | Source / Status |
|---|---|---|---|
| 1 | `fig:architecture` | PRISM-AR 4-layer architecture diagram | User generating via ChatGPT |
| 2 | `fig:scores` | Score distribution by dataset | `results/figures/score_distribution.png` ✅ |
| 3 | `fig:tier_compare` | Tier distribution — PRISM vs ground truth, grouped bar | `results/figures/ground_truth_distribution.png` ✅ |
| 4 | `fig:gt_comp` | Adaptive vs static ground-truth comparison | `results/figures/ground_truth_comparison.png` ✅ |
| 5 | `fig:ablation` | Ablation score shift / static under-warning impact | `results/figures/ablation.png` ✅ |
| 6 | `fig:lead_time` | Warning lead-time CDF | `results/figures/lead_time_distribution.png` ✅ |

---

## Tables (7)

| # | Label | Table | Status |
|---|---|---|---|
| 1 | `tab:related` | Related work comparison (6 systems × 5 dimensions) | ❌ |
| 2 | `tab:tier_schema` | Risk tier schema: score bands, cue color, opacity, icon | ❌ |
| 3 | `tab:datasets` | Dataset summary: clips, frames, VRUs, conditions | ❌ |
| 4 | `tab:main` | Main results by dataset | ✅ `results/tables/main_by_dataset.csv` |
| 5 | `tab:ablation` | Ablation and robustness summary | ✅ `results/tables/ablation.csv`, `robustness.csv` |
| 6 | `tab:latency` | Per-layer runtime latency (ms/frame) | ✅ `results/latency_results.json` |
| 7 | `tab:adverse` | Adverse-condition escalation (clean vs adverse) | ✅ `results/tables/adverse_condition.csv` |

---

## Key Metrics to Report

### Already computed
- Mean PRISM safety score by dataset
- Tier distribution (PRISM + ground truth)
- Adaptive vs static tier accuracy (0.71 vs 0.43, p < 0.0001)
- Intervention recall (0.20 vs 0.00, p < 0.0001)
- Emergency recall (0.09 vs 0.00, p < 0.0001)
- Under-warning / over-warning rates
- Warning lead time
- Cue flicker and visual clutter
- Ablation: env / trajectory / VRU removal
- Robustness: noise, delay, frame drop

### Completed metrics
- **Cue-risk monotonicity** — Spearman rho = -0.703 (mean across clips)
- **SHAP alignment** — 0.251 (fraction of high-risk frames where top factor matches dominant component)
- **Runtime/latency** — total 0.0042 ms/frame (reference implementation)
- **Adverse-condition escalation** — mean score drops from 65.35 (clean) to 50.54 (adverse); intervention recall 0.03 → 0.62; emergency recall 0.00 → 0.33

---

## Lineage for the Paper

| Component | Question It Answers |
|---|---|
| SafeDriver-IQ | "How risky is this driving context?" |
| PRISM | "How should the AV classify and reason about this dynamic scene?" |
| PRISM-AR | "How should the AV communicate that risk to vulnerable road users?" |

---

## Safe Claims

- "PRISM-AR improves risk alignment and warning appropriateness of external VRU communication in simulated and dataset-derived AV scenarios."
- "Risk-adaptive communication is safer and more appropriate than static eHMI/AR communication under the evaluated metrics."
- "PRISM-AR generalizes across three public AV datasets and controlled synthetic near-miss scenarios."

## Do NOT Claim

- "PRISM-AR improves pedestrian behavior."
- "PRISM-AR reduces real-world crashes."
- "Pedestrians trust PRISM-AR more."

These require human-subject studies.

---

## Files Ready for Writing

- `results/report.md` — consolidated summary
- `results/figures/` — 7 figures
- `results/tables/` — 5 CSVs
- `results/prism_ar_results.csv` — per-clip results
- `results/ablation_results.csv` — ablation results
- `results/robustness_results.csv` — robustness results
- `results/latency_results.json` — per-layer runtime

---

## Next Actions

1. Generate `fig:architecture` with ChatGPT
2. Draft LaTeX starting from Abstract → Section 3 → Section 6
3. Build Tables 1, 2, 3 manually
4. Build bibliography (~25 references)
5. Internal review and polish
6. Submit to TVT SI Author Portal by 1 August 2026
