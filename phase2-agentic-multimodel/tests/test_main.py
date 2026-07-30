"""M8 tests — end-to-end pipeline + evaluation routines.

Uses the real models (loaded once, module-scoped) on a small slice of nuScenes so the
wiring is exercised without being slow.
"""
from __future__ import annotations

import pytest

from sdiq.data_loader import iter_nuscenes_scenes
from sdiq.main import (
    AgenticPipeline, PipelineResult, _score_to_tier, evaluate_ablations,
    evaluate_coverage, evaluate_latency, format_result,
)


@pytest.fixture(scope="module")
def pipeline():
    return AgenticPipeline()


@pytest.fixture(scope="module")
def scenes():
    by_id = {s.scene_id: s for s in iter_nuscenes_scenes()}
    # one daytime VRU-conflict scene (the gap) + one night+rain scene
    return [by_id["scene-0061"], by_id["scene-1094"]]


def test_pipeline_process(pipeline, scenes):
    r = pipeline.process(scenes[0], narrate=False)
    assert isinstance(r, PipelineResult)
    assert 0 <= r.tier < 4 and r.tier_name
    assert r.narrative
    # timings populated and sane
    assert {"models+fusion", "rl+shap", "narrate", "total"} <= set(r.timings_ms)
    assert r.timings_ms["total"] > 0
    assert isinstance(format_result(r), str)


def test_parallel_matches_sequential(pipeline, scenes):
    # parallel model execution must produce the SAME assessment as sequential
    seq = pipeline.summarizer.summarize(scenes[1], parallel=False)
    par = pipeline.summarizer.summarize(scenes[1], parallel=True)
    assert seq.vru_risk == par.vru_risk
    assert seq.trajectory_risk == par.trajectory_risk
    assert seq.env_multiplier == par.env_multiplier
    assert seq.fused_risk == par.fused_risk


def test_score_to_tier_bands():
    assert _score_to_tier(85) == 0 and _score_to_tier(55) == 1
    assert _score_to_tier(30) == 2 and _score_to_tier(10) == 3


def test_coverage_detects_vru_gap(pipeline, scenes):
    rows = evaluate_coverage(scenes, pipeline)
    by = {r["scene"]: r for r in rows}
    # scene-0061 is a daytime VRU near-miss — Phase 1 (env only) under-reacts,
    # Phase 2 escalates.
    assert by["scene-0061"]["near_miss"] >= 1
    assert by["scene-0061"]["phase2"] > by["scene-0061"]["phase1"]
    assert by["scene-0061"]["escalated"]


def test_ablations_vru_model_drives_conflict_scene(pipeline, scenes):
    rows = evaluate_ablations(scenes, pipeline)
    by = {r["scene"]: r for r in rows}
    s = by["scene-0061"]
    # adding the VRU model lifts the tier above env / env+traj on the conflict scene
    assert s["env+vru"] > s["env"]
    assert s["full"] >= s["env+vru"]
    assert set(["env", "env+traj", "env+vru", "full"]) <= set(s)


def test_latency_eval_runs(pipeline, scenes):
    stats = evaluate_latency(scenes, pipeline, repeats=1)
    assert set(stats["per_model_ms"]) == {"env", "trajectory", "vru"}
    assert stats["rl_decide_ms"] > 0
    assert stats["parallel_models_max_ms"] <= stats["sequential_models_sum_ms"] + 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
