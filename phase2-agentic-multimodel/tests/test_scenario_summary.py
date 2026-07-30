"""M5 tests — the ScenarioSummary contract and the three-model fusion.

The dataclass/contract tests are exact and fast (no model loading). The fusion math is
tested with injected fake models so it's deterministic. One end-to-end test loads the
real models once to confirm the wiring.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sdiq.data_loader import AgentState, AgentTrack, DrivingScene, _fill_velocity
from sdiq.scenario_summary import (
    VECTOR_FIELDS, ScenarioSummary, ScenarioSummarizer,
)
from sdiq.safedriver_iq_bridge import EnvRisk


def _summary(**over) -> ScenarioSummary:
    base = dict(
        scene_id="s", source="argoverse2", is_night=False, is_rain=False,
        n_agents=5, n_vru=2, env_safety_score=50.0, env_risk=0.5, env_multiplier=1.0,
        env_risk_level="High", trajectory_risk=0.3, trajectory_mean_risk=0.1,
        vru_risk=0.4, vru_realistic_risk=0.2, vru_near_misses=0,
        min_distance=3.0, min_ttc=1.5, fused_risk=0.4, combined_safety_score=60.0,
    )
    base.update(over)
    return ScenarioSummary(**base)


# ---------------------------------------------------------------------------
# contract / dataclass
# ---------------------------------------------------------------------------
def test_vector_shape_and_order():
    s = _summary()
    v = s.to_vector()
    assert v.shape == (len(VECTOR_FIELDS),) == (8,)
    assert v.dtype == np.float32
    assert (v >= 0).all() and (v <= 2).all()   # multiplier can exceed 1


def test_vector_proximity_imminence():
    near = _summary(min_distance=0.5, min_ttc=0.2).to_vector()
    far = _summary(min_distance=20.0, min_ttc=10.0).to_vector()
    pi = VECTOR_FIELDS.index("proximity"); ii = VECTOR_FIELDS.index("imminence")
    assert near[pi] > far[pi] and near[ii] > far[ii]
    # no-VRU sentinels (inf) map to 0 proximity/imminence
    none = _summary(min_distance=float("inf"), min_ttc=float("inf")).to_vector()
    assert none[pi] == 0.0 and none[ii] == 0.0


def test_band_thresholds():
    assert _summary(combined_safety_score=85).band == "silent"
    assert _summary(combined_safety_score=55).band == "advisory"
    assert _summary(combined_safety_score=30).band == "intervention"
    assert _summary(combined_safety_score=10).band == "emergency"


def test_to_dict_json_handles_inf():
    s = _summary(min_distance=float("inf"), min_ttc=float("inf"))
    d = s.to_dict()
    assert d["min_distance"] is None and d["min_ttc"] is None
    assert d["band"] == s.band
    # round-trips through JSON
    again = json.loads(s.to_json())
    assert again["scene_id"] == "s" and again["min_ttc"] is None


# ---------------------------------------------------------------------------
# fusion math (fake models -> deterministic)
# ---------------------------------------------------------------------------
class _FakeEnv:
    def __init__(self, mult): self.mult = mult
    def assess(self, scene):
        return EnvRisk(safety_score=100 * (1 - (self.mult - 0.5)),
                       risk_level="High", env_risk=self.mult - 0.5,
                       multiplier=self.mult, scenario={})


class _FakeKin:
    def __init__(self, r): self.r = r
    def score_scene(self, scene):
        return {"per_agent": {}, "max_risk": self.r, "mean_risk": self.r / 2}


class _FakeVru:
    def __init__(self, r): self.r = r
    def assess_scene(self, scene):
        return {"max_risk": self.r, "mean_risk": self.r / 2, "near_misses": 1,
                "realistic_max_risk": self.r * 0.6, "min_distance": 2.0, "ttc": 1.0,
                "per_agent": {}}


def _tiny_scene(is_night=False, is_rain=False):
    states = [AgentState(t=i * 0.1, x=float(i), y=0.0) for i in range(5)]
    _fill_velocity(states)
    ego = AgentTrack("AV", "vehicle", states)
    return DrivingScene("s", "argoverse2", ego.states, [], sample_rate_hz=10.0,
                        is_night=is_night, is_rain=is_rain)


def test_fusion_multiplier_amplifies():
    # base = (0.5*traj + 1.0*vru)/1.5 ; fused = clip(mult * base)
    summ = ScenarioSummarizer(_FakeEnv(1.5), _FakeKin(0.4), _FakeVru(0.6))
    s = summ.summarize(_tiny_scene())
    base = (0.5 * 0.4 + 1.0 * 0.6) / 1.5
    assert math.isclose(s.fused_risk, min(1.0, 1.5 * base), rel_tol=1e-5)
    assert math.isclose(s.combined_safety_score, 100 * (1 - s.fused_risk), rel_tol=1e-5)


def test_fusion_night_vs_day_amplification():
    # same dynamic risk, higher multiplier -> higher fused risk
    day = ScenarioSummarizer(_FakeEnv(0.9), _FakeKin(0.5), _FakeVru(0.5)).summarize(_tiny_scene())
    night = ScenarioSummarizer(_FakeEnv(1.4), _FakeKin(0.5), _FakeVru(0.5)).summarize(_tiny_scene())
    assert night.fused_risk > day.fused_risk
    assert night.combined_safety_score < day.combined_safety_score


def test_vru_dominates_trajectory():
    # vru_weight > traj_weight: a VRU conflict drives fused risk more than equal trajectory
    vru_heavy = ScenarioSummarizer(_FakeEnv(1.0), _FakeKin(0.0), _FakeVru(0.9)).summarize(_tiny_scene())
    traj_heavy = ScenarioSummarizer(_FakeEnv(1.0), _FakeKin(0.9), _FakeVru(0.0)).summarize(_tiny_scene())
    assert vru_heavy.fused_risk > traj_heavy.fused_risk


# ---------------------------------------------------------------------------
# end-to-end with real models (loads artifacts once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_summarizer():
    return ScenarioSummarizer()


def test_end_to_end_real_models(real_summarizer):
    from sdiq.data_loader import iter_nuscenes_scenes
    scenes = {sc.scene_id: sc for sc in iter_nuscenes_scenes()}
    s = real_summarizer.summarize(scenes["scene-1094"])    # night + rain + VRU
    assert 0.0 <= s.fused_risk <= 1.0
    assert s.is_night and s.is_rain
    assert s.env_multiplier > 1.0                          # night/rain amplifies
    assert s.to_vector().shape == (8,)
    json.loads(s.to_json())                                # serializable


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
