# PRISM-AR: Risk-Adaptive AR Interfaces for Vulnerable Road Users

This project extends the PRISM/SafeDriver-IQ agentic multi-model risk engine into a risk-adaptive AR communication layer for pedestrians and cyclists.

## Objective

Build and evaluate PRISM-AR: a system that maps PRISM's continuous 0–100 safety score and intervention tier (Silent, Advisory, Intervention, Emergency) to dynamic AR cues for VRUs.

## Project Structure

```
prism_ar/
├── data_ingestion/     # Unified DrivingScene format + dataset loaders (Waymo/Argoverse/nuScenes/CRSS)
├── prism/              # PRISM risk engine bridge
├── ar_overlay/         # AR cue mapping (adaptive, static, oracle, no-AR)
├── evaluation/         # Metrics and evaluation scripts
├── dataset_generation/ # Scenario extraction + AR overlay rendering
└── tests/              # Unit tests

run_prism_ar_real_data.py   # End-to-end pipeline on real datasets
run_ablation_study.py       # Ablation experiments
run_robustness_study.py     # Noise/delay/frame-drop robustness
generate_prism_ar_figures.py # Figure generation
setup_venv.bat              # Windows venv setup
results/figures/            # Output figures
```

## Datasets

Real datasets are accessed through short-path Windows junctions (to avoid MAX_PATH issues):

- `C:\data_prismar\waymo` -> Waymo Open Motion Dataset
- `C:\data_prismar\argoverse2` -> Argoverse 2 validation
- `C:\data_prismar\nuscenes` -> nuScenes mini
- `C:\data_prismar\crss` -> CRSS data

Outputs are written to `C:\prismar_out\prism_ar_real` (junction to `results/prism_ar_real/`).

## Quick Start

```powershell
# 1. Create the clean virtual environment
.\setup_venv.bat

# 2. Run tests
C:\prismar_venv\Scripts\python.exe -m pytest prism_ar/tests/ -v

# 3. Run the full real-data pipeline
C:\prismar_venv\Scripts\python.exe run_prism_ar_real_data.py --max_scenes 50

# 4. Generate figures
C:\prismar_venv\Scripts\python.exe generate_prism_ar_figures.py

# 5. Run ablation and robustness studies
C:\prismar_venv\Scripts\python.exe run_ablation_study.py
C:\prismar_venv\Scripts\python.exe run_robustness_study.py
```

## Important Notes

- The current `prism_ar/prism/risk_engine.py` is a **reference implementation** with simplified rules. For paper-ready results, replace it with the trained PRISM/SafeDriver-IQ weights.
- The 2D synthetic dataset (`generate_prism_ar_dataset.py`) is retained only as a smoke-test baseline.
- All heavy dataset dependencies (TensorFlow, av2, nuscenes-devkit) are avoided; custom pure-Python parsers are used instead.

## Reference

Pratticò et al., *Comparing State-of-the-Art and Emerging Augmented Reality Interfaces for Autonomous Vehicle-to-Pedestrian Communication*, IEEE Transactions on Vehicular Technology, 2021.
