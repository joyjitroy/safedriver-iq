# Phase 2 POC — Environment & Project Setup

## Project layout
```
phase2-agentic-multimodel/
├── pyproject.toml          # package metadata (src layout) + pytest config
├── requirements.txt        # frozen, resolved versions
├── README.md
├── docs/
│   ├── implementation-plan.md
│   ├── setup.md                       (this file)
│   ├── asce2027-architecture.docx
│   └── architecture-flow-diagram.png
├── src/sdiq/               # the package (import as `sdiq`)
│   ├── config.py           # all paths + LLM provider config
│   ├── nuscenes_mini.py    # devkit-free nuScenes reader
│   └── data_loader.py      # unified DrivingScene / AgentTrack / EgoState
├── tests/                  # pytest suite
├── reference/              # original ASCE2027 sanity scripts (provenance)
├── datasets/
│   ├── nuscenes-mini/      # v1.0-mini/ + samples/ sweeps/ maps/
│   └── argoverse2-val/     # 24,988 Argoverse 2 scenarios
└── .venv/                  # isolated POC virtualenv
```

## TL;DR
```bash
cd phase2-agentic-multimodel
.venv/bin/python -m sdiq.config          # show resolved paths
.venv/bin/python -m pytest -q            # full suite (M0 + M1)
.venv/bin/python -m sdiq.data_loader     # smoke-test both loaders
```

## Why a dedicated `.venv`
Two hard dependency conflicts forced an isolated environment:

1. **`nuscenes-devkit` hard-pins `numpy<2.0`** — irreconcilable with `torch>=2` and
   `shap>=0.50`, which require `numpy>=2`. Installing it into the shared `src/.venv`
   downgraded numpy and broke `shap`, `opencv`, and the dev `pandas` there. We
   **reverted** the shared env to its original `numpy==2.3.5` and do **not** depend on
   the devkit. nuScenes mini is read directly from its JSON tables by
   [`src/sdiq/nuscenes_mini.py`](../src/sdiq/nuscenes_mini.py).
2. **`numba` (a `shap`/`av2` dep) rejects `numpy>=2.4`** ("Numba needs NumPy 2.3 or
   less"). The base interpreter that `.venv` inherits ships numpy 2.4.6, so we pin
   `numpy==2.3.5` *inside* `.venv` (isolated — affects nothing else).

`.venv` was created with `--system-site-packages` so it inherits the heavy pre-built
`torch` (CUDA) from the base interpreter instead of re-downloading ~2 GB, then layers
the POC-specific packages on top. `sdiq` is installed editable (`pip install -e .`),
so `import sdiq` works from any cwd without `PYTHONPATH`.

## Datasets (all present locally; no download needed)
| Dataset | Path | Notes |
|---|---|---|
| nuScenes v1.0-mini | `datasets/nuscenes-mini/` | 10 scenes, 404 keyframes, 18,538 anns, 12 sensors |
| Argoverse 2 val | `datasets/argoverse2-val/` | 24,988 scenarios (trajectory parquet + HD map) |
| NHTSA CRSS | `../CRSS_Data/` (Phase 1 repo) | tabular, for the RF bridge |
| Waymo WOMD | `../waymo/` (Phase 1 repo) | trajectory TFRecords |

All paths resolve through [`src/sdiq/config.py`](../src/sdiq/config.py) and can be
overridden with `SDIQ_*` env vars (e.g. `SDIQ_NUSCENES_DATAROOT`, `SDIQ_AV2_VAL_ROOT`).

## Frozen Phase 1 model (the RF bridge, M2)
`../results/models/best_safety_model.pkl` — the **64-feature** production RandomForest
matching `feature_names.txt`. (The ASCE2027 docx mislabels `crash_investigation_rf_model.pkl`
as the 64-feature bridge, but that file is a 5-feature investigation model; see the note
in `config.py`.) `.venv` pins `scikit-learn==1.8.0` to match the version that pickled it,
so it loads cleanly via `joblib.load`.

## Recreating `.venv` from scratch
```bash
cd phase2-agentic-multimodel
/home/samaresh/src/.venv/bin/python -m venv --system-site-packages .venv
.venv/bin/pip install av2 pytest
.venv/bin/pip install --ignore-installed "numpy==2.3.5" matplotlib   # avoid OS-numpy-1.x leak
.venv/bin/pip install "scikit-learn==1.8.0" shap xgboost
.venv/bin/pip install -e .                                            # editable install of sdiq
```
See [`requirements.txt`](../requirements.txt) for the exact resolved versions.

## LLM co-pilot (Hybrid scope, added in M7 — off the safety-critical path)
Configured in `src/sdiq/config.py` via `SDIQ_LLM_PROVIDER`:
- `off` (default) — classical SHAP explanations only; no extra deps.
- `claude` — Anthropic API (`pip install anthropic`; models `claude-opus-4-8` / `claude-haiku-4-5`, key via `ANTHROPIC_API_KEY`).
- `local` — llama.cpp OpenAI-compatible server (`pip install httpx`) at `SDIQ_LLM_BASE_URL`.

The co-pilot is **off the safety-critical path**: every call has a hard timeout
(`SDIQ_LLM_TIMEOUT_S`, default 8 s) and falls back to the deterministic SHAP explanation
on timeout/error. It never sets the risk score or intervention tier.
