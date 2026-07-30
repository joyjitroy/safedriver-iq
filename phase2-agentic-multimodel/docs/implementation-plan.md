# SafeDriver-IQ Phase 2 — Agentic Multi-Model POC: Implementation Plan

**Scope decision:** *Hybrid.* Classical models + tabular RL handle the real-time risk loop
(deterministic, fast, on-GPU). A provider-agnostic **LLM co-pilot** sits *beside* the loop for
scenario summaries, natural-language intervention explanations, and the agentic memory narrative.
The LLM is **never on the safety-critical path** — if it is unavailable, the system degrades to the
classical SHAP explanation and keeps running.

**Target:** offline POC for ASCE2027 (matches the docx scope). Real-time pipeline, full RL training,
and federated learning remain Phase 3.

---

## 0. Environment & data prep (Milestone 0 — ~0.5 day)

Active env is `/home/samaresh/src/.venv` (NOT the empty `safedriver-iq/.venv`). It already has:
torch 2.10+cu128 (CUDA ✓), tensorflow 2.20, scikit-learn 1.8, shap 0.50, xgboost, lightgbm, pandas, numpy.

**Gaps to close:**
- [ ] `pip install nuscenes-devkit` (pin a 3.12-compatible version)
- [ ] `pip install av2` — Argoverse 2 devkit can be finicky on Py3.12; if it fails, fall back to reading
      `scenario_*.parquet` directly with pandas/pyarrow (the schema is stable, devkit not strictly required).
- [ ] LLM client deps for the chosen provider:
      - Claude API path: `pip install anthropic` (use model `claude-opus-4-8` for synthesis,
        `claude-haiku-4-5-20251001` for cheap per-scenario summaries).
      - Local path: point at the `llama.cpp` server (`/home/samaresh/src/llama.cpp`) via its OpenAI-compatible
        `/v1/chat/completions` endpoint.
- [x] Reconcile dataset paths. Original test scripts hard-coded `~/av-safety-poc/datasets/...`; real data now
      lives in `datasets/nuscenes-mini/` (nuScenes mini, ~5.1 GB) and `datasets/argoverse2-val/`
      (Argoverse 2 val, 6.1 GB, 24,988 scenarios). A single `src/sdiq/config.py` holds all dataset roots
      (env-overridable via `SDIQ_*`); the config-driven sanity tests live in `tests/`.

**Exit criteria:** both sanity test scripts run green; RF `.pkl` loads (see M2).

---

## Proposed module layout

The package lives under `phase2-agentic-multimodel/src/sdiq/` (src layout, editable-installed;
keeps the Phase 1 repo untouched and imports it as a lib). Tests live in a sibling `tests/`:

```
src/sdiq/
  config.py              # dataset roots, model paths, LLM provider config        [done]
  nuscenes_mini.py       # devkit-free nuScenes reader                            [done]
  data_loader.py         # M1 — unified loaders -> DrivingScene/AgentTrack/EgoState [done]
  kinematic_features.py  # M3 — LSTM trajectory-risk model
  vru_features.py        # M4 — Social Force Model + LSTM correction
  scenario_summary.py    # M5 — per-cycle structured scenario state (feeds RL + LLM)
  safedriver_iq_bridge.py# M2 — loads & wraps the frozen Phase 1 RF as env-risk multiplier
  agentic_layer.py       # M6 — RL fusion + memory + SHAP  (+ LLM co-pilot hook)
  llm_copilot.py         # M7 — provider-agnostic LLM client (Claude API | local llama.cpp)
  main.py                # M8 — end-to-end driver: scenario -> 3 models -> RL -> intervention + narrative
tests/                   # unit + integration tests per module
```

---

## Milestone 1 — Unified data loaders (`data_loader.py`) — ~1.5 days

- `load_nuscenes_mini()` → wraps `NuScenes(version="v1.0-mini")`; yields per-keyframe ego pose, annotated
  agents (class, box, velocity), and VRU subset (pedestrian/cyclist/PMD). 404 keyframes, 18,538 annotations.
- `load_av2_scenario(scenario_id)` → trajectory tracks (vehicle/ped/cyclist, 10 Hz) + HD map; VRU filter.
  Prefer devkit; pandas-parquet fallback ready.
- Reuse Phase 1 for the tabular side: import `src.waymo_data_loader` and the CRSS path for the RF features.
- Normalize all three into a common `AgentTrack` / `EgoState` dataclass so downstream models are dataset-agnostic.

**Verify:** assert counts (10 scenes / 24,988 AV2 scenarios), spot-check a night/after-rain nuScenes scene
(1077/1094/1100) actually surfaces pedestrians/cyclists.

---

## Milestone 2 — RF bridge (`safedriver_iq_bridge.py`) — ~0.5 day

- Load `results/crash_investigation_rf_model.pkl` (970 KB, 64 features) + `results/models/feature_names.txt`.
- Build the 64-feature vector from current scenario context (reuse Phase 1
  `contextual_feature_generator.py` so feature semantics match training).
- Expose `env_risk_multiplier(context) -> float in [~0.5, ~1.5]`: map RF `P(no-crash)` to a multiplier that
  **amplifies** the other two models in adverse conditions (per docx, RF is a context multiplier, not an equal input).
- **No retraining.** Frozen model.

**Verify:** golden-value test — a known feature row reproduces the score Phase 1 produces for it.

---

## Milestone 3 — Trajectory kinematic LSTM (`kinematic_features.py`) — ~2 days

- Inputs per agent: sequence of (speed, longitudinal/lateral accel, heading-change, jerk) at 10 Hz.
- Model: 2-layer LSTM, 64 hidden units (docx POC spec) → trajectory-risk scalar [0,1].
- Labels are **derived** (no crash labels exist on these sets): self-supervised next-step prediction +
  a kinematic-risk heuristic (hard braking, high jerk, lane-departure proxy) as the risk target. **Document this
  clearly** — it's the same synthetic-label limitation Phase 1 carries.
- Train on Argoverse 2 (24,988 scenarios) + Waymo; small enough for the local GPU.

**Verify:** held-out scenarios; sanity that aggressive trajectories score higher risk than smooth ones.

---

## Milestone 4 — VRU interaction (`vru_features.py`) — ~2 days

- **Social Force Model** (Helbing & Molnár 1995): physics, no training. Compute pedestrian/cyclist predicted
  paths, min ego–VRU distance, TTC, near-miss flag.
- **LSTM correction layer**: learns the residual between SFM prediction and observed VRU motion (nuScenes +
  AV2) to sharpen short-horizon forecasts.
- Output: VRU-risk scalar [0,1] + interpretable sub-signals (min-distance, TTC).

**Verify:** on nuScenes night/VRU scenes, near-miss flags fire where annotations show close ego–pedestrian geometry.

---

## Milestone 5 — Scenario summary (`scenario_summary.py`) — ~0.5 day

- Assemble the per-cycle **structured state**: `{env_risk, env_multiplier, trajectory_risk, vru_risk,
  min_ttc, min_distance, scene_meta(night/rain/urban)}`.
- This single object is the contract feeding **both** the RL agent (numeric vector) and the LLM co-pilot
  (JSON for the prompt). Keep it small and stable.

---

## Milestone 6 — Agentic reasoning layer (`agentic_layer.py`) — ~2.5 days

- **State** = scenario-summary vector; **fused risk** = `f(env_multiplier · [trajectory_risk, vru_risk])`.
- **Action** = 4 graduated tiers (70–100 silent · 40–70 advisory · 20–40 specific intervention · 0–20 emergency).
- **RL**: lightweight Q-learning / small policy net (docx POC scope — *not* full training).
- **Reward** = risk-reduction − false-alarm penalty, computed against derived near-miss/TTC signals.
- **Memory**: short-term (current drive ring buffer) + long-term (persisted learned scenario table).
- **SHAP**: per-intervention attribution over the three model inputs → the always-available, deterministic
  explanation (this is the safety-critical explanation; the LLM narrative is additive).

**Verify:** scripted scenarios (school-zone, night-rain VRU crossing, icy) escalate to the expected tier;
SHAP attributions are non-degenerate.

---

## Milestone 7 — LLM co-pilot (`llm_copilot.py`) — ~1.5 days  *(the Hybrid addition)*

Provider-agnostic client; **off the critical path**, called async/after the RL decision.

- `summarize_scenario(scenario_summary) -> str` — concise NL description of the driving situation
  (cheap model / local llama.cpp).
- `explain_intervention(scenario_summary, action, shap_values) -> str` — turns the SHAP attribution + tier
  into a human-readable rationale ("Advisory issued: pedestrian entering crosswalk 1.8 s TTC under wet-night
  conditions amplifying environmental risk ×1.3").
- `narrate_memory(long_term_memory) -> str` — periodic natural-language digest of what the agent has learned
  (the "agentic memory narrative").
- **Guardrails:** strict timeout + fallback to the SHAP-only explanation; the LLM never sets the intervention
  tier or the safety score. Structured (JSON-in / text-out) prompts; log every call for the paper's reproducibility.
- **Provider switch in `config.py`:** `claude` (anthropic SDK, `claude-opus-4-8` / `claude-haiku-4-5-20251001`)
  or `local` (llama.cpp OpenAI-compatible endpoint).

---

## Milestone 8 — Integration (`main.py`) + evaluation — ~1.5 days

- End-to-end: pick scenario → run 3 models in **parallel** (docx rationale: 87% of crashes are multi-factor) →
  scenario summary → RL decision + SHAP → LLM narrative → graduated output.
- **Evaluation for the paper:**
  - Coverage: show Phase 2 now responds to VRU/night/rain cases Phase 1 was blind to (the documented Phase 1 gap).
  - Ablations: RF-only vs +trajectory vs +VRU vs full fusion.
  - Latency budget per layer (justify "parallel, every cycle").
  - LLM explanation quality: spot-check + note it's non-critical.

---

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| `av2` devkit won't install on Py3.12 | Med | Direct parquet read fallback (schema stable). |
| No real crash labels on AV2/nuScenes | High (validity) | Derived near-miss/TTC targets; document as a stated limitation (consistent with Phase 1). |
| RL undertrained at POC scope | Med | Docx already scopes full RL training out; demonstrate the loop + memory, not SOTA policy. |
| LLM latency/cost or offline runs | Low | Off critical path; timeout→SHAP fallback; local llama.cpp option. |
| AV2 is trajectories-only (no sensors) | Low | Multimodal perception demonstrated on nuScenes mini; AV2 used for kinematics/forecasting only. |
| Feature-vector drift vs RF training | Med | Reuse Phase 1 `contextual_feature_generator.py`; golden-value bridge test. |

## Rough timeline
~12–14 working days for the full POC (M0–M8), single workstation + your CUDA GPU. Classical models train in
hours; the LLM layer is additive and can be built in parallel with M3/M4.
