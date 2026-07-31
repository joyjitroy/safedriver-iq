# SafeDriver-IQ: Inverse Crash Modeling for Driver Competency

**Tagline:** *"Quantifying Driver Competency Through Inverse Crash Modeling"*

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Agentic AI](https://img.shields.io/badge/Status-Agentic%20AI%20Phase%201%20Complete-success.svg)](src/agent/)

## Project Overview

SafeDriver-IQ transforms crash data into a continuous safety score that tells drivers in real-time how close they are to crash conditions and what specific actions would make them safer, with special focus on protecting vulnerable road users (VRUs).

### 🆕 NEW: Comprehensive Crash Factor Investigation (Notebook 04)
A deep-dive multi-dataset investigation combining **CRSS** (417K crashes) and **Waymo Open Motion Dataset** to answer 8 core research questions:
1. What factors contribute to vehicle crashes?
2. Which features best predict crash probability?
3. How to classify driver behavior from crash and good-driving data?
4. What data is most critical for VRU/pedestrian/cyclist predictions?
5. What patterns can improve crash prevention model training?
6. What historical crash trends are present across 2016–2023?
7. What environmental conditions uniquely elevate crash risk?
8. How to systematically perform root cause analysis?

**Key addition — Contextual Feature Synthesis:** CRSS captures crash *outcomes* but is silent on contextual preconditions. A new `ContextualFeatureGenerator` synthesises 16 research-calibrated risk dimensions (see [Section 6](#section-6--comprehensive-crash-factor-analysis-beyond-crss-data) in notebook 04) drawn from NHTSA, FHWA-HSM, AAA Foundation, and IIHS sources, enabling richer training and what-if simulation.

### 🤖 Real-Time Agentic Decision Layer (Phase 1 extension)
A rule/RL-based decision layer (`src/agent/`) wraps the Phase 1 tabular safety score with **autonomous decision-making**:
- Real-time risk assessment and autonomous interventions
- Continuous learning from driving experiences
- Multi-modal driver notifications (visual, audio, haptic)
- Transparent, explainable AI reasoning
- Code: [`src/agent/core/`](src/agent/core/) (decision engine) · [`src/agent/perception/`](src/agent/perception/) · [`src/agent/control/`](src/agent/control/) · [`src/agent/learning/`](src/agent/learning/) · [`src/agent/explainability/`](src/agent/explainability/) — see [`demo_agentic_ai.py`](demo_agentic_ai.py) to run it

> **Note:** This is a Phase 1 add-on around the single tabular safety-score model, not to be confused with **PRISM (Phase 2)**, which independently fuses three separate risk models (environmental, trajectory, VRU) via its own RL agent — see below.

### The Problem
- **7,500+ pedestrian deaths/year** in the USA (40-year high)
- **1,000+ cyclist deaths/year**
- Traditional systems are **reactive** (emergency braking) not **proactive**
- No system tells drivers "you're driving safely" or "improve these specific behaviors"

### Our Solution
Instead of predicting crashes, we model the **distance from crash** - quantifying how "safe" a driving scenario is by measuring its statistical distance from crash-producing conditions.

### 🕶️ PRISM-AR: Risk-Adaptive AR Cues for VRUs (extends PRISM / Phase 2)
PRISM-AR (in [`prism-ar/`](prism-ar/)) takes PRISM's continuous safety score and intervention tier and maps them to **adaptive augmented-reality cues** for pedestrians and cyclists — e.g., a translucent amber safe zone at "advisory", a flashing red no-cross boundary at "emergency". It is being prepared for submission to **IEEE Transactions on Vehicular Technology**. See [`prism-ar/README.md`](prism-ar/README.md) for details.

## 📄 Research Publication

This repository serves as the **reference implementation and experimental foundation** for the following research:

### Phase 1: SafeDriver-IQ

**Paper Title:** *Real-Time Driver Safety Scoring Through Inverse Crash Probability Modeling*

This research was developed in collaboration with the **American Center for Mobility (ACM)**, a federally designated proving ground for connected and automated vehicle technology. ACM is featuring the work through its publication channels, reflecting growing industry interest in proactive safety intelligence. The paper was presented at the **2026 IEEE International Conference on Electro/Information Technology (EIT)** in La Crosse, Wisconsin, and at the **IEEE CTSoc Technical Talks Webinar**, and is forthcoming in **IEEE Xplore**.

**Read the paper:** https://arxiv.org/abs/2603.14841

### Key Results (Phase 1)

- **87%** of crashes involve **2+ co-occurring risk factors**
- **4.5x** non-linear risk compounding over baseline
- Model performance: **AP = 0.891**, **precision = 0.941**, **recall = 0.480**
- SHAP-ablation correlation: **r = 0.94**
- Estimated **22.7% crash reduction** with adoption

### Phase 2: PRISM

Phase 2 of this research, an agentic multi-model architecture for proactive safety intervention in autonomous transportation, has been accepted for presentation at the **American Society of Civil Engineers (ASCE) 2027** conference. Validation artifacts are in the [`asce2027/`](asce2027/) directory.

### Key Results (Phase 2)

- **1,296 scenarios** validated across nuScenes (10), Argoverse 2 (1,000), and Waymo WOMD (286)
- **Mean safety score: 68.0/100**
- **77.6%** of scenarios classified as **advisory**
- **3.8% near-miss rate**
- **~11%** escalated to **intervention/emergency**
- Cross-dataset calibration without retraining

### Phase 3: PRISM-AR

**Paper Title:** *Explainable Risk-Adaptive Driving Intelligence for Vulnerable Road User Communication in Automated Vehicles*

Phase 3 extends PRISM's internal AV risk reasoning into external, VRU-facing communication. PRISM-AR (PRISM with Augmented Reality) adds a tiered external communication policy that maps PRISM's fused risk state to adaptive AR cues for pedestrians and cyclists across four escalation levels: silent, information, warning, and emergency. Prepared for submission to **IEEE Transactions on Vehicular Technology (TVT)**, Special Issue on Advanced Driving Intelligence for Autonomous Vehicles (manuscript deadline: 1 August 2026). Source and validation artifacts are in the [`prism-ar/`](prism-ar/) directory.

**Authors:** Joyjit Roy, Meng Lu, Arijit Roy, Sushanta Das, Samaresh Kumar Singh

### Key Results (Phase 3)

- **231 scenario clips** evaluated across three public AV datasets plus controlled near-miss scenarios
- Proxy ground-truth tier accuracy improves from **0.43 (static eHMI)** to **0.71 (PRISM-AR)**
- Cue-risk monotonicity: Spearman **ρ = -0.703** (Wilcoxon p < 0.0001)
- Activates warning/emergency cues in risk-critical cases the static baseline fails to escalate
- Sub-millisecond per-frame latency (reference implementation)

### 📌 Relationship to This Project
Each phase of the SafeDriver-IQ system was **designed, implemented, and validated first**, and the insights, models, and experimental findings from this project directly led to the corresponding research publications: the Phase 1 arXiv paper, the Phase 2 ASCE2027 paper, and the Phase 3 PRISM-AR manuscript prepared for IEEE TVT.

In other words:
- ✅ This repository = **working system + experiments** (Phase 1: `src/`, Phase 2: `phase2-agentic-multimodel/`, Phase 3: `prism-ar/`)
- ✅ The papers = **formalization of methodology, results, and contributions** for each phase

### 🚀 What the Paper Formalizes
The research paper builds on this project and formally introduces:

- **Inverse Crash Probability Modeling** → foundation of the safety score  
- **Continuous Driver Safety Scoring (0–100)** instead of binary crash prediction  
- **Distance-from-crash formulation** using learned decision boundaries  
- **Integration of crash data (CRSS) + behavioral data (Waymo)**  
- **Explainable safety feedback mechanisms for real-time systems**  

### 🔬 How This Repo Maps to the Paper

| Research Concept | Implementation in This Repo |
|----------------|---------------------------|
| Inverse crash modeling | `src/safety_score.py` |
| Feature engineering (120+) | `src/feature_engineering.py` |
| Contextual risk synthesis | `src/contextual_feature_generator.py` |
| Model training (RF/XGBoost) | `src/models.py` |
| Behavioral insights | `src/driver_behavior_classifier.py` |
| Explainability (SHAP) | `notebooks/03_shap_analysis.ipynb` |
| Real-time scoring system | `src/realtime_calculator.py` |

### 🧾 Main Contributors

- Samaresh Kumar Singh
- Joyjit Roy

### 📚 Citation
```bibtex
@article{safedriveriq,
  author  = {Roy, Joyjit and Singh, Samaresh Kumar and Das, Sushanta},
  title   = {Real-Time Driver Safety Scoring Through Inverse Crash Probability Modeling},
  journal = {arXiv preprint arXiv:2603.14841},
  year    = {2026},
  doi     = {10.48550/arXiv.2603.14841},
  url     = {https://arxiv.org/abs/2603.14841},
  note    = {Submitted to IEEE EIT 2026, University of Wisconsin-La Crosse}
}
```

### 📢 Research Flyers

Visual one-page summaries for each phase of the work.

#### Phase 1: SafeDriver-IQ

![SafeDriver-IQ Phase 1 Flyer](docs/flyers/safedriver_iq_phase1_flyer.png)

Phase 1 introduces inverse crash probability modeling: a continuous 0–100 safety score derived from NHTSA CRSS crash data and Waymo driving behavior, demonstrating that 87% of crashes involve two or more co-occurring risk factors.

#### Phase 2: PRISM (ASCE2027)

![PRISM Phase 2 Flyer](docs/flyers/prism_phase2_flyer.png)

Phase 2 extends the foundation into an agentic multi-model architecture with environmental, trajectory, and VRU risk models fused by a DQN agent. PRISM was validated across nuScenes, Argoverse 2, and Waymo WOMD, achieving a mean safety score of 68/100 with 77.6% of scenarios classified as advisory.

## Key Innovations

### Phase 1: SafeDriver-IQ

| Traditional Approach | SafeDriver-IQ (Novel) |
|---------------------|----------------------|
| Binary crash prediction | Continuous safety score (0-100) |
| "30% crash risk" | "Safety score: 72/100 → Improve to 85+" |
| Reactive warnings | Proactive guidance with specific actions |
| General risk factors | VRU-specific safety models |
| CRSS-only training | CRSS + Waymo + synthesised contextual features |

**Novel Contributions (Phase 1):**
1. **Inverse Safety Score Formulation** - Continuous safety metric (0-100) instead of binary crash prediction
2. **Good Driver Profile Extraction** - First empirical characterisation of safe driving from crash data + Waymo behavioural data
3. **VRU-Specific Safety Modeling** - Dedicated models for pedestrian, cyclist, and work zone encounters
4. **Contextual Feature Synthesis** - 16 research-calibrated risk dimensions generated from `ContextualFeatureGenerator` to fill CRSS data gaps
5. **Multi-Method Feature Consensus** - Random Forest, XGBoost, Permutation Importance, and SHAP combined into a single consensus ranking
6. **Real-Time Integration Architecture** - Practical system design for in-vehicle deployment

### Phase 2: PRISM

| Traditional Approach | PRISM (Novel) |
|---------------------|----------------------|
| Single risk model | Three parallel risk models (environmental, trajectory, VRU interaction) fused by an RL agent |
| Static thresholds on one score | Q-net RL policy selecting one of four graduated intervention tiers |
| One-shot decisions | Short-term and long-term memory across decisions |
| Opaque model output | Per-decision SHAP-based explanations |
| Dataset-specific retraining | Cross-dataset calibration (nuScenes, Argoverse 2, Waymo WOMD) without retraining |

**Novel Contributions (Phase 2):**
1. **Multi-Model Risk Fusion** - Parallel environmental, trajectory-kinematic, and VRU-interaction risk models integrated via reinforcement learning
2. **Agentic Tiered Decision Policy** - Q-net RL policy replacing static thresholds, with an asymmetric reward that penalizes under-reaction 2x over over-reaction
3. **Social-Force + LSTM VRU Modeling** - Physics-informed Social Force Model combined with a learned LSTM residual correction for ego-VRU conflict prediction
4. **Explainable Agentic Reasoning** - SHAP-based per-decision explanations with short/long-term memory
5. **Cross-Dataset Generalization** - Validated on nuScenes, Argoverse 2, and Waymo WOMD without retraining

### Phase 3: PRISM-AR

| Traditional Approach | PRISM-AR (Novel) |
|---------------------|----------------------|
| Static, predefined eHMI signals | Risk-adaptive AR cues driven directly by PRISM's fused risk state |
| Binary or single-level warnings | Four-tier graduated external communication (silent, information, warning, emergency) |
| External signals disconnected from AV's internal state | External communication explicitly grounded in explainable internal risk reasoning |
| Single-condition evaluation | 231 scenario clips across three public AV datasets plus controlled near-miss scenarios |

**Novel Contributions (Phase 3):**
1. **Risk-Adaptive External Communication** - First framework directly linking an AV's internal fused risk state to VRU-facing AR cues
2. **Explainable Tiered Escalation** - Four-level cue policy (silent, information, warning, emergency) grounded in the fused PRISM risk score
3. **Cue-Risk Monotonicity Validation** - Spearman ρ = -0.703 (Wilcoxon p < 0.0001) alignment between cue intensity and risk severity
4. **Real-Time Feasibility** - Sub-millisecond per-frame reference implementation
5. **Multi-Baseline Evaluation Protocol** - Paired comparison against no-interface, static eHMI, and oracle upper-bound policies

## Dataset

**CRSS (Crash Report Sampling System)** — NHTSA national crash database
- **417,335 crash records** (2016–2023, 8 years)
- **38,462 VRU crashes** (pedestrians + cyclists)
- **1,032,571 person records**
- Tables: `ACCIDENT`, `VEHICLE`, `PERSON`, `PBTYPE`, `FACTOR`, `DISTRACT`, `DRIMPAIR`, `WEATHER`, and more

**Waymo Open Motion Dataset (WOMD v1.2)** — Real-world autonomous driving scenarios
- **6 splits**: training (1,000 shards), training_20s, validation, validation_interactive, testing, testing_interactive
- **91 timesteps per scenario** at 10 Hz (1 s context + 8 s future horizon)
- Captures: agent trajectories (vehicles, pedestrians, cyclists), road graph, traffic signals, speed limits
- Used for: Good driver profiling, near-miss detection, behavioral pattern extraction
- Stored via **Git LFS** in `waymo/motion_dataset/`

**nuScenes v1.0-mini** — Public autonomous-driving benchmark
- **10 scenes**, 12 sensors (6 cameras, 5 radars, 1 LiDAR)
- Full 3D bounding-box tracks for vehicles, pedestrians, and cyclists
- Used for: PRISM/PRISM-AR validation in complex urban intersections and adverse-lighting scenes
- Stored under `phase2-agentic-multimodel/datasets/nuscenes-mini/`

**Argoverse 2** — Large-scale motion forecasting dataset
- **24,988 scenarios** in the validation split (1,000 sampled as PRISM/PRISM-AR candidate clips)
- 10 Hz agent tracks (position, velocity, heading) across multiple US cities
- Used for: PRISM/PRISM-AR validation across diverse urban environments and per-city breakdowns
- Stored under `phase2-agentic-multimodel/datasets/argoverse2-val/`

### Dataset Summary by Paper

| Paper | Dataset | Size | Type | Purpose |
|---|---|---|---|---|
| Phase 1 (EIT2026) | NHTSA CRSS 2016–2023 | 417,335 crash records | Historical crash outcomes | Train inverse crash probability model |
| Phase 2 (ASCE2027) | nuScenes mini | 10 scenarios | Autonomous driving scenes | Validate PRISM across diverse urban environments |
| Phase 2 (ASCE2027) | Argoverse 2 | 1,000 scenarios | Autonomous driving scenes | Validate PRISM across diverse urban environments |
| Phase 2 (ASCE2027) | Waymo WOMD | 286 scenarios | Autonomous driving scenes | Validate PRISM across diverse urban environments |
| Phase 3 (PRISM-AR/TVT) | nuScenes v1.0-mini | 10 source scenes → 96 evaluated clips | VRU-interaction clips | Evaluate risk-adaptive AR cues |
| Phase 3 (PRISM-AR/TVT) | Argoverse 2 | 1,000 candidate → 50 evaluated clips | VRU-interaction clips | Evaluate risk-adaptive AR cues |
| Phase 3 (PRISM-AR/TVT) | Waymo WOMD | 286 candidate → 25 evaluated clips | VRU-interaction clips | Evaluate risk-adaptive AR cues |
| Phase 3 (PRISM-AR/TVT) | Synthetic near-miss generator | 60 scenarios | Controlled pedestrian-crossing scenes | Supplement rare warning/emergency cases |

Phase 3 candidate clips are drawn from the same three AV datasets as Phase 2 (1,296 total candidate clips), then filtered by a VRU-interaction extractor (20 m ego-approach radius, decreasing distance, ≥5 frames visible) down to 231 total evaluated scenario clips. See [`prism-ar/README.md`](prism-ar/README.md) and `prism-ar/src/prism_ar/dataset_generation/scenario_extractor.py` for the extraction logic.

## System Architecture

### Architecture Diagrams

#### SafeDriver-IQ (Phase 1) Architecture

![SafeDriver-IQ Architecture](docs/images/safedriver_iq_architecture.png)

SafeDriver-IQ is the first-generation inverse crash modeling system. It combines national crash statistics (CRSS 2016–2023) with real-world behavioral data (Waymo WOMD) to train a binary crash classifier, then inverts its predicted probability of *not* crashing into a continuous 0–100 safety score. The pipeline includes data ingestion, feature engineering (120+ variables), crash pattern analysis, model training, and a real-time scoring interface that maps scores to five risk levels: Critical, High, Medium, Low, and Excellent.

#### PRISM (Phase 2) Agentic Multi-Model Architecture

![PRISM Architecture](docs/images/prism_architecture.png)

PRISM (Proactive Risk Intelligence and Safety Management) extends SafeDriver-IQ into a four-layer agentic architecture:

1. **Layer 1 — Data Ingestion and Normalization**: Converts heterogeneous AV datasets (nuScenes, Argoverse 2, Waymo WOMD) into a unified `DrivingScene` representation.
2. **Layer 2 — Parallel Risk Models**: Runs three independent models concurrently:
   - **Environmental Risk**: Reuses the frozen SafeDriver-IQ random forest as a context estimator.
   - **Trajectory Kinematic**: Evaluates speed, acceleration, and yaw-rate exceedances.
   - **VRU Interaction**: Uses a Social Force Model + LSTM to predict ego-VRU conflicts.
3. **Layer 3 — Agentic Reasoning**: Fuses risks via a DQN reinforcement learning agent, selects one of four intervention tiers (silent, advisory, intervention, emergency), and provides SHAP-based explanations with short-term and long-term memory.
4. **Layer 4 — Applications**: Supports ADAS integration, fleet risk management, and infrastructure planning without dataset-specific retraining.

Unlike Phase 1's fixed decision boundary, Layer 3's tier selection is a trained Q-net RL policy (not a fixed weighted formula) that maps the 8-dimensional fused risk state to one of the four graduated tiers, using an asymmetric reward that penalizes under-reaction 2x over over-reaction.

#### Phase 1 → Phase 2 Mapping

| | Phase 1 (SafeDriver-IQ) | Phase 2 (PRISM) |
|---|---|---|
| Safety score range | 0–100 (inverse crash probability) | 0–100 (inherited, continuous) |
| Output categories | 5 risk levels: Critical, High, Medium, Low, Excellent | 4 intervention tiers: silent, advisory, intervention, emergency |
| Decision mechanism | Static thresholds on the inverse safety score | Q-net RL policy over the fused 8-dim risk state |
| Risk sources | Single frozen Random Forest (environmental only) | Three parallel models (environmental RF + trajectory LSTM + VRU SFM/LSTM) fused by the RL agent |

The environmental Random Forest is **frozen and reused as-is** from Phase 1 in Phase 2 (`safedriver_iq_bridge.py`), acting as a context multiplier rather than the sole risk signal — this is how Phase 2 addresses the Phase 1 limitation noted below (no response to road condition/VRU presence/speed).

#### PRISM-AR (Phase 3) Risk-Adaptive AR Architecture

![PRISM-AR Architecture](prism-ar/results/figures/F1_PRISM_AR_Architecture.png)

PRISM-AR is the third stage in the SafeDriver-IQ → PRISM → PRISM-AR sequence. It adds an **augmented reality external communication layer** (HMD overlays, projected cues, pedestrian-facing vehicle displays) on top of PRISM's risk reasoning. The reference implementation evaluated in the paper is a **deterministic** four-layer pipeline (not the Q-net RL policy used in Phase 2):

1. **Layer 1 — Data Ingestion and Scene Abstraction**: Standardizes nuScenes, Argoverse 2, and Waymo WOMD into the unified `DrivingScene` representation (10 Hz agent tracks, weather/lighting/road/time attributes). NHTSA CRSS is *not* treated as a scene source here — it only feeds the environmental risk bridge below.
2. **Layer 2 — Multi-Model Risk Engine**: Three parallel modules, each producing a risk value on a common [0, 1] scale:
   - **Environmental risk** — blends a trained scene-context model with the SafeDriver-IQ CRSS-derived estimate: `r_env = 0.95 * r_model + 0.05 * r_crss`.
   - **Trajectory risk** — closed-form, saturating function of time-to-collision (TTC) between ego and the nearest VRU.
   - **VRU-interaction risk** — closed-form, saturating function of minimum ego–VRU distance.
3. **Layer 3 — Risk Fusion and Tier Selection**: Deterministic weighted fusion `r_fused = clip(w_env·r_env + w_traj·r_traj + w_vru·r_vru)`, with **w_env = 0.40, w_traj = 0.30, w_vru = 0.30** for all reported results. Converted to a safety score `S = 100·(1 - r_fused)` and mapped to one of four tiers by fixed thresholds.
4. **Layer 4 — External VRU Communication Policy**: Deterministic tier-to-cue lookup table (opacity increases with tier; flashing reserved for the highest tier), delivered through configurable VRU-facing channels (HMD, projection, exterior display).

> **Terminology note:** the paper names the four tiers **Silent, Information, Warning, Emergency**, whereas the current codebase (`prism-ar/src/prism_ar/prism/risk_engine.py`) uses **silent, advisory, intervention, emergency**. These refer to the same four escalation levels but with different labels for the two middle tiers — worth reconciling before final submission.
>
> The architecture supports an **agentic extension** (a DQN policy over the fused risk state, with SHAP explanations and short-term memory, analogous to Phase 2's Layer 3) but this extension is **not evaluated** in the PRISM-AR reference results — all reported numbers use the fixed-weight, threshold-based pipeline above.

#### Phase 2 → Phase 3 Mapping

| | Phase 2 (PRISM) | Phase 3 (PRISM-AR, reference implementation) |
|---|---|---|
| Tier selection | Q-net RL policy over 8-dim fused state | Fixed weights + threshold lookup (deterministic) |
| Risk fusion | Learned via RL reward | Closed-form weighted sum, `w = (0.40, 0.30, 0.30)` |
| Output | Internal intervention tier (AV-facing) | External AR cue mapped from the same tier concept (VRU-facing) |
| Explainability | Per-decision SHAP + memory | Deterministic lookup (traceable by construction); DQN+SHAP extension designed but not evaluated |
| Tier labels | silent, advisory, intervention, emergency | Silent, Information, Warning, Emergency (paper) |

## Project Structure

```
├── asce2027/                   # ASCE2027 PRISM validation artifacts (reproducibility bundle)
│   ├── scripts/                # Analysis scripts (AV2, Waymo, nuScenes)
│   ├── data/                   # Validation results (CSV/JSON)
│   └── figures/                # Paper figures
├── phase2-agentic-multimodel/  # PRISM (Phase 2) source: sdiq package, models, tests
│   ├── src/sdiq/                # config, data loaders, kinematics, social force, agentic layer, LLM co-pilot
│   ├── tests/                   # pytest suite (61 tests)
│   └── docs/                    # setup, implementation plan, ASCE2027 architecture docs
├── prism-ar/                   # PRISM-AR (Phase 3) source: risk-adaptive AR cue mapping for VRUs
│   ├── src/prism_ar/            # data_ingestion, prism (risk engine), ar_overlay, dataset_generation, evaluation
│   ├── tests/                   # unit tests
│   ├── scripts/                 # run_*.py, generate_*.py, measure_*.py
│   ├── data/                    # scenario annotations + rendered AR overlay images
│   ├── results/                 # CSVs, JSON, figures, report.md
│   ├── notebooks/
│   └── docs/                    # PAPER_PLAN.md, architecture docx, Backups/
├── docs/
│   ├── images/                 # Architecture diagrams (SafeDriver-IQ + PRISM)
│   └── flyers/                 # Research flyers (Phase 1 + Phase 2)
├── CRSS_Data/                  # National crash database (2016-2023)
│   ├── 2016/                   # Year-wise crash data
│   ├── 2017/
│   └── ...
├── waymo/                      # Waymo Open Motion Dataset (Git LFS)
│   └── motion_dataset/
│       ├── datasets_scenario/  # Scenario-format TFRecords
│       └── tf_example_datasets/# TF Example-format TFRecords
├── data/
│   ├── raw/                    # Downloaded CRSS files
│   ├── processed/              # Cleaned datasets
│   └── external/               # VMT exposure data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_inverse_model.ipynb
│   ├── 03_shap_analysis.ipynb
│   └── 04_crash_factor_investigation.ipynb  # 8-investigation deep-dive
├── src/
│   ├── data_loader.py              # CRSS data loading
│   ├── preprocessing.py            # Data cleaning
│   ├── feature_engineering.py      # 120+ features
│   ├── models.py                   # Model training/saving
│   ├── safety_score.py             # Score computation
│   ├── realtime_calculator.py      # Live safety scoring
│   ├── scenario_simulator.py       # What-if analysis
│   ├── contextual_feature_generator.py  # synthesised contextual features
│   ├── crash_insights.py           # crash investigation utilities
│   ├── driver_behavior_classifier.py    # behavior clustering
│   ├── feature_importance.py       # multi-method feature consensus
│   ├── waymo_data_loader.py        # Waymo TFRecord loader
│   └── visualization.py            # Plotting utilities
├── src/agent/                       # Real-time agentic decision layer (Phase 1 extension, see above)
│   ├── core/                       # decision engine
│   ├── perception/                 # context engine
│   ├── control/                    # intervention controller
│   ├── learning/                   # continuous/online learning
│   └── explainability/
├── src/simulation/
│   └── scenario_simulator.py       # used by demo_agentic_ai.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   ├── test_preprocessing.py
│   └── test_realtime_calculator.py
├── results/
│   ├── figures/               # Visualizations
│   ├── tables/                # Analysis results
│   ├── crash_investigation_feature_importance.csv
│   ├── crash_investigation_behavior_clusters.csv
│   ├── crash_investigation_rf_model.pkl
│   └── models/                # Trained models
└── app/
    └── streamlit_app.py       # Interactive dashboard
```

## Quick Start (5 Minutes)

### Prerequisites
- Python 3.12+ installed
- CRSS data downloaded to `CRSS_Data/` directory
- Terminal/Command line access

### Setup & Run

```bash
# Navigate to project directory
cd safedriver-iq

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
# source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify data loading (shows 417K+ crashes)
python test_data_loader.py

# Run COMPLETE demonstration (trains model + all features)
python run_complete_demo.py --quick

# Or run quick data demo
python demo_quick.py

# Or explore interactively
jupyter notebook notebooks/01_data_exploration.ipynb

# Launch interactive dashboard (after training model)
streamlit run app/streamlit_app.py
```

### Expected Output
```
✓ 417,335 total crashes loaded
✓ 38,462 VRU crashes identified
✓ Data from 2016-2023 successfully loaded
```

## Quick Start — Phase 2 (PRISM)

Phase 2 lives in its own isolated environment under `phase2-agentic-multimodel/` and does not share Phase 1's virtualenv.

```bash
cd phase2-agentic-multimodel

# Show resolved dataset/model paths
.venv/bin/python -m sdiq.config

# Run the test suite (61 tests)
.venv/bin/python -m pytest -q

# Smoke-test the unified data loaders (nuScenes + AV2)
.venv/bin/python -m sdiq.data_loader

# Run the full agentic pipeline end-to-end (graduated output per nuScenes scene)
.venv/bin/python -m sdiq.main run

# Compare Phase 1 vs Phase 2 intervention coverage
.venv/bin/python -m sdiq.main coverage

# Per-model ablation (marginal effect of env/trajectory/VRU)
.venv/bin/python -m sdiq.main ablations

# Per-layer latency breakdown
.venv/bin/python -m sdiq.main latency
```

See [phase2-agentic-multimodel/README.md](phase2-agentic-multimodel/README.md) for milestone status and [phase2-agentic-multimodel/docs/setup.md](phase2-agentic-multimodel/docs/setup.md) for environment details.

## Quick Start — Phase 3 (PRISM-AR)

Phase 3 also lives in its own folder (`prism-ar/`) with an editable `src/` layout, independent of Phase 1 and Phase 2's environments.

```bash
cd prism-ar

# 1. Install dependencies (editable install using the src/ layout in pyproject.toml)
pip install -r requirements.txt
pip install -e .

# 2. Run tests
pytest tests/ -v

# 3. Run the full real-data pipeline (Waymo + Argoverse 2 + nuScenes + synthetic near-miss)
python scripts/run_prism_ar_real_data.py --max_scenes 50

# 4. Generate the paper figures
python scripts/generate_prism_ar_figures.py

# 5. Run ablation and robustness studies
python scripts/run_ablation_study.py
python scripts/run_robustness_study.py
```

On Windows, `scripts/setup_venv.bat` can be used instead of step 1 to create an isolated virtual environment first. See [prism-ar/README.md](prism-ar/README.md) for dataset path configuration and implementation notes.

## Detailed Setup Instructions

### Step 1: Clone/Download Project
```bash
cd /path/to/your/workspace
git clone https://github.com/ssam18/safedriver-iq.git
cd safedriver-iq
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
# source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- **Data Processing:** pandas, numpy, pyarrow
- **Machine Learning:** scikit-learn, xgboost, lightgbm
- **Visualization:** matplotlib, seaborn, plotly
- **Model Interpretation:** shap
- **Interactive:** jupyter, streamlit

### Step 4: Extract CRSS Data
If data is still zipped:
```bash
cd CRSS_Data
for year in 2016 2017 2018 2019 2020 2021 2022 2023; do
    unzip -o ${year}/CRSS${year}CSV.zip -d ${year}/
done
cd ..
```

### Step 5: Verify Setup
```bash
python test_data_loader.py
```

Should show successful loading of 417K+ crash records.

### Step 6: Run Tests (Optional)
```bash
# Activate virtual environment (REQUIRED before running tests)
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# Install test dependencies
pip install -r requirements-test.txt

# Run all tests with basic output
pytest

# Run all tests with verbose output and short traceback (recommended)
pytest tests/ -v --tb=short

# Run all tests with verbose output and one-line traceback (most concise)
pytest tests/ -v --tb=line

# Run tests without coverage calculation (faster)
pytest tests/ -v --tb=short --no-cov

# Run tests in quiet mode with one-line traceback (minimal output)
pytest tests/ -q --tb=line

# Run with coverage report (detailed analysis)
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # On Mac
# xdg-open htmlcov/index.html  # On Linux
```

**Test Output Options:**
- `-v` = verbose mode (shows each test name)
- `-q` = quiet mode (minimal output, just pass/fail counts)
- `--tb=short` = shorter traceback on failures (recommended for debugging)
- `--tb=line` = one-line traceback (most concise, good for CI/CD)
- `--no-cov` = skip coverage calculation (faster test runs)

**Test Realtime Calculator** (verifies condition changes affect scores):
```bash
# Run realtime calculator tests to verify model sensitivity
pytest tests/test_realtime_calculator.py -v --tb=short -s
```

Expected: 65 tests total (53 pass + 12 realtime tests with 5 expected failures due to known model limitations)

## Pipeline

### Phase 1: Data Preparation
- Load CRSS datasets (2016-2023)
- Load Waymo Open Motion Dataset (TFRecord parsing via `WaymoDataLoader`)
- Filter VRU crashes
- Feature engineering (120+ variables)
- Create exposure-weighted baseline

### Phase 2: Crash Factor Investigation (Notebook 04 — NEW)
- **Investigation 1** — Primary crash factors (temporal, environmental, VRU interactions)
- **Investigation 2** — Feature selection via 4-method consensus (RF, XGBoost, Permutation, SHAP)
- **Investigation 3** — Driver behavior classification (CRSS crash clusters + Waymo good-driver profiling)
- **Investigation 4** — Critical data for crash/VRU prediction
- **Investigation 5** — Crash prevention patterns and high-risk combinations
- **Investigation 6** — Historical year-over-year trends (2016–2023)
- **Investigation 7** — Environmental uniqueness analysis (rare high-severity conditions)
- **Investigation 8** — Root cause analysis causal chain framework
- **Section 6** — Contextual feature synthesis with `ContextualFeatureGenerator` (16 research-calibrated risk factors)

### Phase 3: Crash Pattern Analysis
- Clustering → Identify crash archetypes
- Association Rules → Find co-occurring risk factors
- Feature Importance → Rank risk contributors

### Phase 4: Inverse Safety Model
- Train crash classifier (Random Forest / XGBoost, n_estimators=200)
- Extract decision boundaries
- Compute "distance from crash boundary" = Safety Score
- Profile "good driver" = maximises safety score (using Waymo behavioural data)

### Phase 5: Validation & Visualization
- Cross-validation metrics
- SHAP analysis for interpretability
- Dashboard for results presentation

## ASCE2027 Validation Artifacts

The `asce2027/` folder contains the reproducibility bundle for the ASCE2027 conference paper:

- **Scripts**: `compute_stats.py`, `check_tier_mapping.py`, `find_thresholds.py`, `sensitivity_final.py`, `av2_and_shap_outputs.py`, `fig2_score_distribution_final.py`, `generate_paper_outputs.py`, `waymo_validation.py`, `waymo_validation_run.py`
- **Data**: `av2_validation_1000.csv`, `av2_validation_by_city.csv`, `av2_scenario_summary.csv`, `scenario_summary.csv`, `waymo_validation_results.json`, `waymo_scenario_summary.csv`, `fig2_score_distribution_data.csv`
- **Figures**: Score distribution, tier distribution, ablation, latency, near-miss, SHAP, and VRU proximity plots

These artifacts support the paper's results: mean safety score 68/100, 77.6% advisory, 3.8% near-miss rate, and ~11% escalation to intervention/emergency.

### Reproducing the ASCE2027 figures/tables

```bash
cd asce2027/scripts
python generate_paper_outputs.py   # regenerates data/ and figures/ from validation results
python compute_stats.py            # summary statistics used in the paper
python check_tier_mapping.py       # verifies score-to-tier assignment
python find_thresholds.py          # threshold search used for tier boundaries
python sensitivity_final.py        # sensitivity analysis
```

## New Features (Just Completed! 🎉)

### 🚀 Full Pipeline Implemented

**1. Comprehensive Crash Factor Investigation (Notebook 04)**
- 8 structured investigations using CRSS + Waymo datasets
- Multi-method feature importance consensus (RF + XGBoost + Permutation + SHAP)
- Driver behavior classification linking crash patterns to Waymo good-driving profiles
- Root cause analysis causal chain framework
- Results saved: `results/crash_investigation_feature_importance.csv`, `results/crash_investigation_rf_model.pkl`

**2. Contextual Feature Generator (`src/contextual_feature_generator.py`)**
- Synthesises 16 research-backed risk dimensions missing from CRSS
- Top risk factors by weight:

| Weight | Factor | Source |
|--------|--------|--------|
| 0.28 | DUI risk — late night + weekend + bar density | NHTSA |
| 0.24 | Black ice — temperature < 35°F + precipitation | NHTSA |
| 0.20 | Active work zone with workers on roadway | NHTSA |
| 0.18 | Rush hour — dense traffic + tailgating | FHWA-HSM |
| 0.16 | Aggressive surrounding drivers | AAA Foundation |
| 0.15 | Narrow lane (<11 ft) on horizontal curve | FHWA-HSM |
| 0.14 | Driver fatigue — 2–6 AM circadian low | NHTSA |
| 0.13 | Distracted driving (phone / in-cabin) | NHTSA/IIHS |

- Enables what-if simulation across any risk factor combination

**3. Waymo Data Loader (`src/waymo_data_loader.py`)**
- Parses Waymo Open Motion Dataset TFRecord format (v1.2)
- Extracts per-agent state (position, velocity, heading), road graph, traffic signals
- Computes crash indicators: TTC, min inter-agent distance, near-miss flags
- Supports all 6 dataset splits (training, validation, testing + interactive variants)

**4. Model Training**
- Complete inverse safety model training pipeline
- Three model types: Random Forest (n_estimators=200, max_depth=10), XGBoost (n_estimators=200, max_depth=6, lr=0.1), Gradient Boosting
- Automated best model selection based on performance
- Model saving/loading with feature persistence
- Automated best model selection based on performance
- Model saving/loading with feature persistence

**5. Safety Score Calculation**
- Continuous scores (0-100) instead of binary prediction
- Five risk levels: Critical, High, Medium, Low, Excellent
- Confidence intervals for each prediction
- Distance from crash boundary computation

**6. Real-Time Calculator**
- Instant safety score for any driving scenario
- Specific, actionable improvement recommendations
- Scenario comparison capabilities
- Batch analysis for multiple scenarios

**7. Interactive Dashboard**
- Web-based Streamlit application
- Real-time safety score calculator interface
- Scenario comparison tools
- Improvement suggestion engine
- Batch analysis with visualizations
- About page with methodology explanation

**8. Scenario Simulator**
- Factorial scenario generation
- Monte Carlo random sampling
- Time-series trip simulation
- Risk pattern templates (high-risk, low-risk, night, weather, speed, VRU)
- Comprehensive test suite generator

**9. SHAP Interpretability**
- Global feature importance analysis
- Individual prediction explanations
- Feature interaction detection
- Waterfall plots for high/medium/low safety scenarios
- Decision plots comparing multiple scenarios
- Comprehensive interpretation report

## Demonstration & Results

### What You Can Demo NOW

#### 1. Data Loading & Scale
```bash
python test_data_loader.py
```
Shows: 417K crashes, 38K VRU crashes, 8 years of national data

#### 2. Quick Demo (All Key Insights)
```bash
python demo_quick.py
```
Shows:
- Data loading statistics
- VRU crash trends over time
- Temporal patterns (peak times, seasonal)
- Feature engineering capabilities
- Novel approach explanation
- Expected impact projections

#### 3. Interactive Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
Includes:
- Comprehensive data quality analysis
- VRU crash distribution and trends
- Temporal pattern visualizations
- Environmental factor analysis
- Injury severity patterns

#### 4. Crash Factor Investigation (NEW)
```bash
jupyter notebook notebooks/04_crash_factor_investigation.ipynb
```
Includes:
- 8 structured investigations with CRSS + Waymo data
- Multi-method feature importance consensus
- Driver behavior clustering
- Contextual feature synthesis (Section 6)
- What-if sensitivity analysis
- Root cause causal chain framework

### Key Insights Available

**Crash Patterns:**
- Peak crash times: Evening rush hour (5-7 PM)
- High-risk periods: Weekend nights
- VRU crashes concentrated in urban areas
- Dark/poor lighting significantly increases risk

**VRU Statistics (2023):**
- 2,907 pedestrians involved in crashes
- 2,026 bicyclists involved in crashes
- Fatal injury rate: ~5-7% for VRUs (vs. ~2% vehicle occupants)

**Feature Engineering:**
- 120+ features created from raw data
- Temporal, environmental, location, VRU-specific categories
- Interaction features for complex scenarios

### Demo Scenarios

Key notebooks for demonstrations:
- **[01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)** — Data quality, VRU trends, temporal patterns
- **[02_train_inverse_model.ipynb](notebooks/02_train_inverse_model.ipynb)** — Full inverse safety model training
- **[03_shap_analysis.ipynb](notebooks/03_shap_analysis.ipynb)** — SHAP interpretability deep-dive
- **[04_crash_factor_investigation.ipynb](notebooks/04_crash_factor_investigation.ipynb)** — 8-investigation crash factor analysis with Waymo integration

## Expected Impact

With 20% adoption, SafeDriver-IQ could prevent:
- **1,500 pedestrian deaths/year** (20% reduction)
- **200 cyclist deaths/year** (20% reduction)
- **170 work zone deaths/year** (20% reduction)
- **30,000 VRU injuries/year** (20% reduction)

**Total impact: 1,870+ lives saved annually**

## Documentation

- **[README.md](README.md)** — Project overview & setup (this file)
- **[PROJECT_SETUP_SUMMARY.md](PROJECT_SETUP_SUMMARY.md)** — Detailed setup reference
- **[notebooks/04_crash_factor_investigation.ipynb](notebooks/04_crash_factor_investigation.ipynb)** — Comprehensive crash factor investigation

## Known Issues & Limitations

The current trained model does not respond meaningfully to changes in **road condition**, **VRU presence**, or **speed relative to limit**. This is a fundamental limitation of training on crash-only data: the model never learned what "truly safe" driving looks like. It remains useful for weather, lighting, and temporal risk patterns. For test evidence, run `pytest tests/test_realtime_calculator.py -v --tb=short`.

The PRISM architecture (Phase 2) addresses this by adding separate trajectory-kinematic and VRU-interaction models, plus a DQN fusion agent. Validation results are in the `asce2027/` directory.

## Contributing

This is a research project. For questions or collaboration:
- Review [notebooks/04_crash_factor_investigation.ipynb](notebooks/04_crash_factor_investigation.ipynb) for the latest investigation results
- Run `demo_quick.py` to see current capabilities
- Check issues for planned features

## License

[To be determined - typically MIT or Apache 2.0 for research code]

## Acknowledgments

- **American Center for Mobility (ACM)** for collaboration and domain guidance as a federally designated CAV proving ground
- **NHTSA** for CRSS data availability
- **SafeDriver-IQ** novel methodology development
- Python scientific computing community (pandas, scikit-learn, etc.)

