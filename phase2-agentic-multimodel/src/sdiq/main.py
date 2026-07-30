"""M8 — end-to-end agentic pipeline + evaluation.

Wires all four architecture layers into one call and provides the paper-oriented
evaluations:

    scene
      └─ Layer 2: env-RF + trajectory-LSTM + VRU(SFM+LSTM)   [parallel every cycle]
           └─ Layer 3a: fuse -> ScenarioSummary (state vector)
                └─ Layer 3b: RL policy -> graduated tier + SHAP
                     └─ Layer 3c: LLM co-pilot narrative (off the critical path)
                          └─ graduated output

CLI:
    python -m sdiq.main run         # full pipeline over nuScenes, graduated output per scene
    python -m sdiq.main coverage    # Phase-1 (env only) vs Phase-2 (full) — the gap
    python -m sdiq.main ablations   # env / +trajectory / +VRU / full — each model's marginal effect
    python -m sdiq.main latency     # per-layer timing; sequential vs parallel model cycle
    python -m sdiq.main all         # everything
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from sdiq.agentic_layer import AgenticReasoner, TIER_NAMES, N_TIERS
from sdiq.data_loader import DrivingScene, iter_nuscenes_scenes
from sdiq.llm_copilot import LLMCopilot
from sdiq.scenario_summary import VECTOR_FIELDS, ScenarioSummarizer, ScenarioSummary


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    scene_id: str
    source: str
    summary: ScenarioSummary
    tier: int
    tier_name: str
    narrative: str
    timings_ms: dict = field(default_factory=dict)


class AgenticPipeline:
    """The full Phase 2 system: 3 models -> fusion -> RL decision + SHAP -> narrative."""

    def __init__(self, summarizer=None, reasoner=None, copilot=None,
                 parallel_models: bool = True) -> None:
        self.summarizer = summarizer or ScenarioSummarizer()
        self.reasoner = reasoner or AgenticReasoner()
        self.copilot = copilot or LLMCopilot()
        self.parallel_models = parallel_models

    def process(self, scene: DrivingScene, narrate: bool = True,
                explain: bool = True) -> PipelineResult:
        t = {}
        t0 = time.perf_counter()
        summary = self.summarizer.summarize(scene, parallel=self.parallel_models)
        t["models+fusion"] = (time.perf_counter() - t0) * 1e3

        t1 = time.perf_counter()
        decision = self.reasoner.decide(summary, explain=explain)
        t["rl+shap"] = (time.perf_counter() - t1) * 1e3

        t2 = time.perf_counter()
        narrative = (self.copilot.explain_intervention(summary, decision)
                     if narrate else decision.explanation)
        t["narrate"] = (time.perf_counter() - t2) * 1e3
        t["total"] = (time.perf_counter() - t0) * 1e3

        return PipelineResult(scene.scene_id, scene.source, summary,
                              decision.tier, decision.tier_name, narrative, t)


# graduated output (the 4-tier ASCE2027 output)
_TIER_GLYPH = {0: "·", 1: "›", 2: "‼", 3: "■"}


def format_result(r: PipelineResult) -> str:
    s = r.summary
    flags = ("night " if s.is_night else "") + ("rain" if s.is_rain else "")
    flags = flags.strip() or "clear"
    nm = f" near-miss@{s.min_distance:.1f}m/{s.min_ttc:.1f}s" if s.vru_near_misses else ""
    return (
        f"{_TIER_GLYPH[r.tier]} {r.scene_id:11s} [{flags:11s}] "
        f"score={s.combined_safety_score:5.1f} → {r.tier_name.upper():12s}\n"
        f"    env×{s.env_multiplier:.2f}  traj={s.trajectory_risk:.2f}  "
        f"vru={s.vru_risk:.2f}{nm}  ({s.n_vru} VRU)\n"
        f"    {r.narrative}"
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def _score_to_tier(score_0_100: float) -> int:
    """The ASCE2027 graduated bands on a 0-100 safety score (higher = safer)."""
    if score_0_100 >= 70:
        return 0
    if score_0_100 >= 40:
        return 1
    if score_0_100 >= 20:
        return 2
    return 3


def evaluate_coverage(scenes, pipeline: AgenticPipeline) -> list[dict]:
    """Phase 1 (environmental RF alone) vs Phase 2 (full agentic) intervention tier.

    Phase 1's documented blind spot: VRU presence barely moves its score, so daytime
    pedestrian/cyclist conflicts read as merely 'advisory'. Phase 2's VRU + trajectory
    models escalate exactly those cases.
    """
    rows = []
    for sc in scenes:
        r = pipeline.process(sc, narrate=False, explain=False)
        s = r.summary
        phase1_tier = _score_to_tier(s.env_safety_score)   # env RF only
        phase2_tier = r.tier
        rows.append({
            "scene": sc.scene_id,
            "night": s.is_night, "rain": s.is_rain, "n_vru": s.n_vru,
            "near_miss": s.vru_near_misses,
            "phase1": phase1_tier, "phase2": phase2_tier,
            "escalated": phase2_tier > phase1_tier,
            "deescalated": phase2_tier < phase1_tier,
        })
    return rows


def evaluate_ablations(scenes, pipeline: AgenticPipeline) -> list[dict]:
    """Marginal contribution of each model: feed the RL agent state vectors with
    components ablated (zeroed) and report the resulting tier."""
    idx = {f: i for i, f in enumerate(VECTOR_FIELDS)}
    keep_env = [idx["env_risk"], idx["env_multiplier"], idx["is_night"], idx["is_rain"]]
    traj_idx = [idx["trajectory_risk"]]
    vru_idx = [idx["vru_risk"], idx["proximity"], idx["imminence"]]

    def masked(v, keep):
        m = np.zeros_like(v)
        for i in keep:
            m[i] = v[i]
        return m

    rows = []
    for sc in scenes:
        summary = pipeline.summarizer.summarize(sc)
        v = summary.to_vector()
        configs = {
            "env": masked(v, keep_env),
            "env+traj": masked(v, keep_env + traj_idx),
            "env+vru": masked(v, keep_env + vru_idx),
            "full": v,
        }
        tiers = {name: pipeline.reasoner.decide(vec, explain=False, remember=False).tier
                 for name, vec in configs.items()}
        rows.append({"scene": sc.scene_id, "n_vru": summary.n_vru,
                     "near_miss": summary.vru_near_misses, **tiers})
    return rows


def evaluate_latency(scenes, pipeline: AgenticPipeline, repeats: int = 2) -> dict:
    """Per-layer wall-clock. Models are timed individually to contrast a sequential
    cycle (sum) with the parallel cycle the architecture specifies (max)."""
    S = pipeline.summarizer
    per_model = {"env": [], "trajectory": [], "vru": []}
    rl, shap_t, parallel_cycle, seq_cycle = [], [], [], []
    for sc in scenes * repeats:
        # individual model timings
        t = time.perf_counter(); S.env.assess(sc); per_model["env"].append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); S.kin.score_scene(sc); per_model["trajectory"].append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); S.vru.assess_scene(sc); per_model["vru"].append((time.perf_counter() - t) * 1e3)
        # parallel vs sequential model cycle
        t = time.perf_counter(); S.summarize(sc, parallel=True); parallel_cycle.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); summ = S.summarize(sc, parallel=False); seq_cycle.append((time.perf_counter() - t) * 1e3)
        # RL + SHAP
        t = time.perf_counter(); d = pipeline.reasoner.decide(summ, explain=False, remember=False); rl.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); pipeline.reasoner.decide(summ, explain=True, remember=False); shap_t.append((time.perf_counter() - t) * 1e3)

    mean = lambda xs: float(np.mean(xs))
    pm = {k: mean(v) for k, v in per_model.items()}
    return {
        "per_model_ms": pm,
        "sequential_models_sum_ms": sum(pm.values()),
        "parallel_models_max_ms": max(pm.values()),
        "measured_sequential_cycle_ms": mean(seq_cycle),
        "measured_parallel_cycle_ms": mean(parallel_cycle),
        "rl_decide_ms": mean(rl),
        "rl+shap_ms": mean(shap_t),
        "n_samples": len(scenes) * repeats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_coverage(rows):
    print("\n=== COVERAGE: Phase 1 (env RF only) vs Phase 2 (full agentic) ===")
    print("  Phase 1's intervention tracks only ambient conditions (night/weather);")
    print("  Phase 2 re-aligns it to the actual dynamic hazard (VRU + trajectory).")
    print(f"  {'scene':11s} {'cond':6s} {'VRU':>4s} {'NM':>3s} {'Phase1':>10s} {'Phase2':>12s}  change")
    esc = deesc = 0
    for r in rows:
        cond = ("N" if r["night"] else "-") + ("R" if r["rain"] else "-")
        mark = ""
        if r["escalated"]:
            esc += 1; mark = "  ⬆ caught missed hazard"
        elif r["deescalated"]:
            deesc += 1; mark = "  ⬇ avoided false alarm"
        print(f"  {r['scene']:11s} {cond:6s} {r['n_vru']:4d} {r['near_miss']:3d} "
              f"{TIER_NAMES[r['phase1']]:>10s} {TIER_NAMES[r['phase2']]:>12s}{mark}")
    print(f"  -> Phase 2 changed {esc + deesc}/{len(rows)} interventions: {esc} escalated "
          f"(daytime VRU conflict Phase 1 is blind to), {deesc} de-escalated "
          f"(hazard-free night Phase 1 over-warns on).")


def _print_ablations(rows):
    print("\n=== ABLATIONS: intervention tier as each model is added ===")
    print(f"  {'scene':11s} {'VRU':>4s} {'NM':>3s} {'env':>8s} {'+traj':>10s} {'+vru':>10s} {'full':>12s}")
    for r in rows:
        print(f"  {r['scene']:11s} {r['n_vru']:4d} {r['near_miss']:3d} "
              f"{TIER_NAMES[r['env']]:>8s} {TIER_NAMES[r['env+traj']]:>10s} "
              f"{TIER_NAMES[r['env+vru']]:>10s} {TIER_NAMES[r['full']]:>12s}")


def _print_latency(stats):
    print("\n=== LATENCY (mean ms over {} samples) ===".format(stats["n_samples"]))
    for k, v in stats["per_model_ms"].items():
        print(f"  model {k:11s}: {v:7.1f} ms")
    print(f"  RL decide               : {stats['rl_decide_ms']:7.1f} ms")
    print(f"  RL + SHAP explanation   : {stats['rl+shap_ms']:7.1f} ms")
    seq, par = stats["measured_sequential_cycle_ms"], stats["measured_parallel_cycle_ms"]
    print(f"  3-model cycle: sequential {seq:.0f} ms  vs  threaded {par:.0f} ms")
    ideal = stats["sequential_models_sum_ms"] / max(stats["parallel_models_max_ms"], 1e-6)
    print(f"  -> Theoretical parallel ceiling is ~{ideal:.1f}x (sum {stats['sequential_models_sum_ms']:.0f} "
          f"-> max {stats['parallel_models_max_ms']:.0f} ms), but Python THREADS do not realize it: "
          f"the env model's pure-Python feature engineering holds the GIL, and the trajectory LSTM "
          f"dominates. True 'parallel every cycle' needs process-level parallelism or GPU batching "
          f"(the trajectory model at {stats['per_model_ms']['trajectory']:.0f} ms is the batching target).")


def main(argv=None):
    import sys
    cmd = (argv or sys.argv[1:] or ["run"])[0]
    scenes = list(iter_nuscenes_scenes())
    pipeline = AgenticPipeline()

    if cmd in ("run", "all"):
        print("=== AGENTIC PIPELINE — graduated output over nuScenes ===")
        narrate = LLMCopilot().enabled
        for sc in scenes:
            print(format_result(pipeline.process(sc, narrate=narrate)))
    if cmd in ("coverage", "all"):
        _print_coverage(evaluate_coverage(scenes, pipeline))
    if cmd in ("ablations", "all"):
        _print_ablations(evaluate_ablations(scenes, pipeline))
    if cmd in ("latency", "all"):
        _print_latency(evaluate_latency(scenes, pipeline))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    main()
