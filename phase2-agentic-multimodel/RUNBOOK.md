# SafeDriver-IQ Phase 2 — Validation & Run Runbook

A step-by-step guide to validate, test, and run the entire Phase 2 implementation
locally and confirm every layer is correct. Each step lists the command, what it
proves, and the expected result so you can verify as you go.

All commands are run from the project root:

```bash
cd /home/samaresh/src/safedriver-iq/phase2-agentic-multimodel
```

Everything uses the project's isolated virtualenv `.venv` (never the shared
`src/.venv`). The trained model artifacts ship in `artifacts/`, so steps 1–6 work
immediately; retraining (step 7) is optional.

---

## 0. Prerequisites (one-time, already done on this machine)

- `.venv/` exists with the pinned deps (see [docs/setup.md](docs/setup.md)).
- Datasets present under `datasets/` (nuScenes mini + Argoverse 2 val).
- Phase 1 repo one level up (`../`) for the frozen RF model.

Quick check that the environment is intact:

```bash
.venv/bin/python -c "import numpy, torch, sklearn, shap, av2; \
print('numpy', numpy.__version__, '| torch', torch.__version__, \
'| cuda', torch.cuda.is_available(), '| sklearn', sklearn.__version__)"
```

**Expect:** `numpy 2.3.5 | torch 2.12.0+cu130 | cuda True | sklearn 1.8.0`
(CUDA may be `False` on a CPU-only machine — everything still runs, just slower.)

If `.venv` is missing, recreate it per the "Recreating `.venv`" block in
[docs/setup.md](docs/setup.md).

---

## 1. Validate configuration & data paths

```bash
.venv/bin/python -m sdiq.config
```

**Proves:** all dataset roots, the frozen RF model path, and the LLM provider
resolve correctly.
**Expect:** paths under `datasets/nuscenes-mini` and `datasets/argoverse2-val`,
RF model at `../results/models/best_safety_model.pkl`, `LLM provider : off`, and a
final line `config OK` (no `PROBLEMS`).

---

## 2. Run the full test suite (the primary correctness gate)

```bash
.venv/bin/python -m pytest -q
```

**Proves:** all 9 modules behave correctly — data loaders, RF bridge, kinematics,
VRU/SFM, fusion, RL agent, LLM co-pilot, and the end-to-end pipeline.
**Expect:** `61 passed` (≈30–40 s; the first run is slower due to CUDA/Phase-1
warm-up). One harmless `matplotlib Axes3D` warning is fine.

Per-module breakdown if you want to run them individually:

```bash
.venv/bin/python -m pytest tests/test_data_loader.py -q        # M1  (4)
.venv/bin/python -m pytest tests/test_bridge.py -q             # M2  (6)
.venv/bin/python -m pytest tests/test_kinematics.py -q         # M3  (8)
.venv/bin/python -m pytest tests/test_vru.py -q                # M4  (9)
.venv/bin/python -m pytest tests/test_scenario_summary.py -q   # M5  (8)
.venv/bin/python -m pytest tests/test_agentic.py -q            # M6  (9)
.venv/bin/python -m pytest tests/test_llm_copilot.py -q        # M7  (7)
.venv/bin/python -m pytest tests/test_main.py -q               # M8  (6)
.venv/bin/python -m pytest tests/test_nuscenes_mini.py tests/test_av2_val.py -q  # M0 sanity
```

---

## 3. Validate the data layer (M0–M1)

```bash
.venv/bin/python -m sdiq.nuscenes_mini      # devkit-free nuScenes reader
.venv/bin/python -m sdiq.data_loader        # unified loaders, both datasets
```

**Proves:** nuScenes mini (10 scenes / 404 keyframes / 18,538 annotations) and
Argoverse 2 load into the common `DrivingScene`/`AgentTrack` schema.
**Expect:** `nuscenes_mini` prints the first scene + 12 sensor channels;
`data_loader` lists all 10 nuScenes scenes (note `scene-1094 [night, rain] | 85
agents (60 VRU)`) and 3 Argoverse 2 scenes at 10 Hz.

---

## 4. Validate each risk model standalone (M2–M4)

```bash
# M2 — environmental RF bridge: night/rain amplifies, day damps
.venv/bin/python -m sdiq.safedriver_iq_bridge

# M4 — Social Force Model VRU interaction risk (constant-velocity collision-course)
.venv/bin/python -m sdiq.social_force
```

**Proves (M2):** the frozen 64-feature RF maps to a context multiplier.
**Expect:** day scenes `mult≈0.96`; night `mult≈1.31–1.34`; night+rain
`scene-1094 mult=1.350`.

**Proves (M4):** the SFM flags the close ego–VRU encounters.
**Expect:** `scene-0061 max_risk=0.719 near_misses=1` and `scene-1094
max_risk=0.435`; far scenes `0.000`.

(The M3 trajectory model and M4 LSTM are validated via their training entry points
in step 7 and via the pipeline in step 6.)

---

## 5. Validate fusion, the agent, and the co-pilot (M5–M7)

```bash
# M5 — fuse 3 models into the ScenarioSummary state vector
.venv/bin/python -m sdiq.scenario_summary

# M6 — train the RL policy (fast, synthetic) + agentic decisions with SHAP
.venv/bin/python -m sdiq.agentic_layer

# M7 — LLM co-pilot wiring with a fake completion (no network)
.venv/bin/python -m sdiq.llm_copilot
```

**Expect (M5):** a table where night/rain scenes carry `env_mult≈1.3` and the two
conflict scenes (0061, 1094) reach the highest `fused` risk.
**Expect (M6):** `tier_accuracy=0.907 underreaction=0.003`, then per-scene
decisions where 0061 and 1094 → `emergency` with SHAP-based explanations.
**Expect (M7):** per-scene `summary`/`why` lines tagged `[claude-haiku-4-5]`
(from the demo's fake model) and a memory digest tagged `[claude-opus-4-8]`.

---

## 6. Run the end-to-end pipeline + evaluations (M8 — the headline)

```bash
.venv/bin/python -m sdiq.main run         # graduated output, all 4 layers, per scene
.venv/bin/python -m sdiq.main coverage    # Phase 1 vs Phase 2 — the closed gap
.venv/bin/python -m sdiq.main ablations   # each model's marginal contribution
.venv/bin/python -m sdiq.main latency     # per-layer timing
.venv/bin/python -m sdiq.main all         # everything above
```

**Proves:** the complete system and the three paper results.
**Expect:**
- `run` — `scene-0061` and `scene-1094` → `EMERGENCY`; the rest advisory/silent.
- `coverage` — "Phase 2 changed 4/10 interventions: 1 escalated … 3 de-escalated".
- `ablations` — the `+vru` column is the one that lifts 0061/1094 to
  `intervention`; `full` → `emergency`.
- `latency` — env ≈73 ms, trajectory ≈368 ms, vru ≈75 ms, RL ≈2 ms; with the
  honest note that Python threads don't beat sequential (GIL).

These numbers back Tables in [docs/paper/results.tex](docs/paper/results.tex).

---

## 7. (Optional) Retrain the artifacts from scratch

The shipped artifacts in `artifacts/` reproduce the paper numbers. To regenerate
them (GPU recommended; CPU works but is slow):

```bash
# Trajectory LSTM — ~7 min on the RTX 4060 (parquet I/O dominates)
SDIQ_TRAIN_SCENARIOS=2500 .venv/bin/python -m sdiq.kinematic_features
# Expect: val_auc≈0.95, saves artifacts/kinematic_lstm.pt

# VRU SFM+LSTM correction — ~3 min
SDIQ_TRAIN_SCENARIOS=1200 .venv/bin/python -m sdiq.vru_features
# Expect: "ADE improvement: ~32%", saves artifacts/vru_lstm.pt

# RL policy — seconds (synthetic states)
.venv/bin/python -m sdiq.agentic_layer
# Expect: tier_accuracy≈0.91, saves artifacts/agentic_policy.pt
```

Re-run step 2 afterwards to confirm the suite still passes with the fresh models.

---

## 8. (Optional) Enable the real LLM co-pilot

The co-pilot is `off` by default (deterministic SHAP explanations only). To enable
natural-language narration:

```bash
# Anthropic API (Claude)
.venv/bin/pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
SDIQ_LLM_PROVIDER=claude .venv/bin/python -m sdiq.main run

# OR a local llama.cpp server (OpenAI-compatible) on :8080
.venv/bin/pip install httpx
SDIQ_LLM_PROVIDER=local SDIQ_LLM_BASE_URL=http://127.0.0.1:8080/v1 \
  .venv/bin/python -m sdiq.main run
```

The co-pilot is off the safety-critical path: an 8 s timeout (`SDIQ_LLM_TIMEOUT_S`)
and automatic fallback to the SHAP explanation mean the pipeline behaves
identically if the LLM is slow, unavailable, or disabled.

---

## What "correct" looks like (acceptance checklist)

- [ ] Step 1: `config OK`, paths resolve.
- [ ] Step 2: `61 passed`.
- [ ] Step 3: 10 nuScenes scenes; `scene-1094` shows 60 VRU.
- [ ] Step 4: bridge multipliers 0.96 (day) / 1.35 (night+rain); SFM near-miss on 0061.
- [ ] Step 5: RL `tier_accuracy≈0.91`; 0061 & 1094 → emergency.
- [ ] Step 6: coverage "changed 4/10"; ablation `+vru` escalates 0061/1094.
- [ ] (If retrained) Step 7: AUC ≈0.95, ADE ≈−32%, tier acc ≈0.91.

If all boxes check, every layer of the Phase 2 implementation is verified end to end.

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ModuleNotFoundError: sdiq` | Run via `.venv/bin/python`; the package is editable-installed in `.venv`. Re-run `.venv/bin/pip install -e .` if needed. |
| `cuda False` / slow | CPU-only machine; everything still runs. Retraining will be slower. |
| Bridge errors importing `gpu_config` / `src` | The bridge adds the Phase 1 repo (`../`) to `sys.path` automatically; ensure the Phase 1 repo is one directory up with `results/models/best_safety_model.pkl` present. |
| `pytest` picks up `--cov=src` and errors | You're running the system `pytest` or the parent repo's config. Use `.venv/bin/python -m pytest`; the project's `pyproject.toml` provides its own config. |
| First test run very slow (minutes) | One-time CUDA/cuDNN + Phase 1 import warm-up; subsequent runs are ~30 s. |
| `nuscenes-devkit` / numpy conflicts | The POC is deliberately devkit-free; do not install `nuscenes-devkit` into `.venv` (it pins `numpy<2` and breaks the stack). |
