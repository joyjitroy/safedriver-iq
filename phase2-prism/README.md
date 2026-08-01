# SafeDriver-IQ — Phase 2: Agentic Multi-Model (POC)

Phase 2 of [SafeDriver-IQ](../README.md). It extends the Phase 1 tabular safety-scoring
model into an **agentic, multi-model** system that fuses three risk models — the frozen
Phase 1 environmental RandomForest, an LSTM trajectory-kinematics model, and a
Social-Force + LSTM VRU-interaction model — through an RL reasoning layer with memory and
SHAP explanations, plus a **hybrid LLM co-pilot** for natural-language summaries and
intervention rationales (off the safety-critical path).

Target venue: ASCE2027. Scope: offline proof-of-concept (no real-time pipeline / federated
learning — those are Phase 3).

## Quick start
```bash
cd phase2-agentic-multimodel
.venv/bin/python -m sdiq.config        # show resolved dataset/model paths
.venv/bin/python -m pytest -q          # run the test suite
.venv/bin/python -m sdiq.data_loader   # smoke-test the unified loaders
```
See [docs/setup.md](docs/setup.md) for environment details and
[docs/implementation-plan.md](docs/implementation-plan.md) for the full milestone plan.

## Layout
| Path | What |
|---|---|
| [src/sdiq/](src/sdiq/) | the `sdiq` package (config, loaders, models) |
| [tests/](tests/) | pytest suite |
| [datasets/nuscenes-mini/](datasets/) | nuScenes v1.0-mini (10 scenes, 12 sensors) |
| [datasets/argoverse2-val/](datasets/) | Argoverse 2 val (24,988 scenarios) |
| [docs/](docs/) | plan, setup, ASCE2027 architecture docx + diagram |
| [reference/](reference/) | original ASCE2027 sanity scripts (provenance) |

## Status
- **M0 — environment & data** ✅ isolated `.venv`, devkit-free nuScenes, config-driven sanity tests.
- **M1 — unified data loaders** ✅ `DrivingScene` / `AgentTrack` / `EgoState` over nuScenes + AV2.
- **M2 — RF env-risk bridge** ✅ `safedriver_iq_bridge.py` reuses the frozen 64-feature RF as a context multiplier (night/rain amplifies, benign damps).
- **M3 — LSTM trajectory-kinematics** ✅ `kinematics.py` (Savitzky-Golay features + calibrated derived risk) + `kinematic_features.py` (2-layer anticipatory LSTM). Trained on 2,500 AV2 scenarios / 72,880 windows → **val AUC 0.950**, MSE 0.044; transfers to nuScenes 2 Hz. Artifact: `artifacts/kinematic_lstm.pt`.
- **M4 — VRU interaction** ✅ `social_force.py` (Helbing-Molnár SFM + constant-velocity collision-course risk / TTC / near-miss) + `vru_features.py` (LSTM residual correction in a VRU-centric canonical frame). Trained on 1,200 AV2 scenarios / 16,357 windows → **SFM+LSTM cuts forecast ADE 32%** (0.65→0.44 m) vs the SFM physics baseline; flags the close ego-VRU encounters on nuScenes (scene-0061 → 0.72 risk + near-miss). Artifact: `artifacts/vru_lstm.pt`.
- **M5 — scenario summary** ✅ `scenario_summary.py` runs all three models and fuses them into one stable `ScenarioSummary` (`to_vector()` for the RL agent, `to_dict()` for the LLM). Env model is a context multiplier; night/rain+VRU scenes (1094) escalate to *intervention*. The integration contract for M6/M7.
- **M6 — agentic reasoning** ✅ `agentic_layer.py`: a Q-net RL policy maps the 8-dim state to 4 graduated tiers (silent/advisory/intervention/emergency) from an asymmetric reward (under-reaction penalized 2×); short/long-term memory + **SHAP** per-decision explanations. Policy: **tier accuracy 0.91, under-reaction 0.3%**. On nuScenes the VRU-near-miss (0061) and night+rain+VRU (1094) scenes escalate to *emergency*. Artifact: `artifacts/agentic_policy.pt`.
- **M7 — LLM co-pilot** ✅ `llm_copilot.py`: provider-agnostic (Claude API via the Anthropic SDK, or a local llama.cpp OpenAI-compatible server), **off the safety-critical path** — scenario summaries, intervention narration, memory digest. Hard timeout with deterministic **SHAP fallback**; never sets risk/tier. Defaults to `off`.
- **M8 — end-to-end + evaluation** ✅ `main.py`: `AgenticPipeline` runs all four layers → graduated output. Evaluations: **coverage** (Phase 2 re-aligns 4/10 interventions — catches the daytime VRU near-miss Phase 1 misses, drops 3 hazard-free night false-alarms), **ablations** (the VRU model is what escalates the conflict scenes), **latency** (per-layer, honest about the GIL limit on threaded parallelism).

**POC complete — all 8 milestones done; 61 tests passing.**

```bash
.venv/bin/python -m sdiq.main run        # graduated output per nuScenes scene
.venv/bin/python -m sdiq.main coverage   # Phase 1 vs Phase 2 (the gap)
.venv/bin/python -m sdiq.main ablations  # each model's marginal effect
.venv/bin/python -m sdiq.main latency    # per-layer timing
```
