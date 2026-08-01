# PRISM-AR: Risk-Adaptive AR Interfaces for Vulnerable Road Users

This project extends the PRISM/SafeDriver-IQ agentic multi-model risk engine into a risk-adaptive AR communication layer for pedestrians and cyclists.

## Objective

Build and evaluate PRISM-AR: a system that maps PRISM's continuous 0–100 safety score and intervention tier (Silent, Advisory, Intervention, Emergency) to dynamic AR cues for VRUs.

## Project Structure

```
phase3-prism-ar/
├── src/prism_ar/
│   ├── data_ingestion/     # Unified DrivingScene format + dataset loaders (Waymo/Argoverse/nuScenes/CRSS)
│   ├── prism/              # PRISM risk engine bridge
│   ├── ar_overlay/         # AR cue mapping (adaptive, static, oracle, no-AR)
│   ├── evaluation/         # Metrics and evaluation scripts
│   └── dataset_generation/ # Scenario extraction + AR overlay rendering
├── tests/                  # Unit tests
├── scripts/
│   ├── run_prism_ar_real_data.py   # End-to-end pipeline on real datasets
│   ├── run_ablation_study.py       # Ablation experiments
│   ├── run_robustness_study.py     # Noise/delay/frame-drop robustness
│   ├── generate_prism_ar_figures.py # Figure generation
│   └── setup_venv.bat              # Windows venv setup
├── data/                       # Scenario annotations + rendered AR overlay images
├── results/figures/            # Output figures
├── notebooks/
├── docs/                       # PAPER_PLAN.md, architecture docx, Backups/
├── pyproject.toml              # editable install (src/ layout)
└── requirements.txt
```

## Datasets

Real datasets are accessed through short-path Windows junctions (to avoid MAX_PATH issues):

- `C:\data_prismar\waymo` -> Waymo Open Motion Dataset
- `C:\data_prismar\argoverse2` -> Argoverse 2 validation
- `C:\data_prismar\nuscenes` -> nuScenes mini
- `C:\data_prismar\crss` -> CRSS data

Outputs are written to `C:\prismar_out\prism_ar_real` (junction to `results/prism_ar_real/`).

## Quick Start

```bash
cd phase3-prism-ar

# 1. Install dependencies (editable install using the src/ layout in pyproject.toml)
pip install -r requirements.txt
pip install -e .

# 2. Run tests
pytest tests/ -v

# 3. Run the full real-data pipeline
python scripts/run_prism_ar_real_data.py --max_scenes 50

# 4. Generate figures
python scripts/generate_prism_ar_figures.py

# 5. Run ablation and robustness studies
python scripts/run_ablation_study.py
python scripts/run_robustness_study.py
```

On Windows, `scripts/setup_venv.bat` can be used instead of steps 1 to create an isolated virtual environment first.

## Important Notes

- `src/prism_ar/prism/risk_engine.py` defines both `PRISMRiskEngine` (reference, fixed rule-based environmental risk, default weights 0.15/0.40/0.45) and `TrainedPRISMRiskEngine` (trained scene-context model blended with the SafeDriver-IQ CRSS estimate, weights 0.40/0.30/0.30). **All paper-reported results use `TrainedPRISMRiskEngine`**, matching Eq. (4)/(5) in the manuscript.
- The 2D synthetic dataset (`scripts/generate_prism_ar_dataset.py`) is retained only as a smoke-test baseline.
- All heavy dataset dependencies (TensorFlow, av2, nuscenes-devkit) are avoided; custom pure-Python parsers are used instead.

## Reference

Pratticò et al., *Comparing State-of-the-Art and Emerging Augmented Reality Interfaces for Autonomous Vehicle-to-Pedestrian Communication*, IEEE Transactions on Vehicular Technology, 2021.
