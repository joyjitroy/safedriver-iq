"""M2 tests — the env-risk bridge wraps the frozen Phase 1 RF correctly.

Covers: the multiplier math (pure, no model), the scene->scenario mapping, and the
behavioural contract that adverse environments (night/rain) amplify while benign ones
damp — plus determinism and a regression guard on the score bands.
"""
from __future__ import annotations

import math

import pytest

from sdiq import config
from sdiq.data_loader import iter_nuscenes_scenes
from sdiq.safedriver_iq_bridge import EnvRiskBridge


# A module-scoped bridge — loading the Phase 1 model is the expensive part.
@pytest.fixture(scope="module")
def bridge() -> EnvRiskBridge:
    return EnvRiskBridge()


@pytest.fixture(scope="module")
def scenes_by_id() -> dict:
    return {sc.scene_id: sc for sc in iter_nuscenes_scenes()}


def test_multiplier_math():
    """_score_to_risk: 100->min, 0->max, 50->midpoint; always within bounds."""
    b = EnvRiskBridge.__new__(EnvRiskBridge)   # no model load needed for the math
    b.mult_min, b.mult_max = 0.5, 1.5
    assert b._score_to_risk(100.0) == (0.0, 0.5)
    assert b._score_to_risk(0.0) == (1.0, 1.5)
    risk50, mult50 = b._score_to_risk(50.0)
    assert math.isclose(risk50, 0.5) and math.isclose(mult50, 1.0)
    # Out-of-range scores clamp.
    assert b._score_to_risk(120.0)[1] == 0.5
    assert b._score_to_risk(-10.0)[1] == 1.5


def test_scene_to_scenario_mapping(scenes_by_id):
    night_rain = EnvRiskBridge.scene_to_scenario(scenes_by_id["scene-1094"])  # [night, rain]
    assert night_rain["IS_NIGHT"] == 1
    assert night_rain["POOR_LIGHTING"] == 1
    assert night_rain["ADVERSE_WEATHER"] == 1
    assert night_rain["WEATHER"] == 2
    assert night_rain["HOUR"] == 23
    # VRU counts are derived and consistent.
    assert night_rain["total_vru"] == night_rain["pedestrian_count"] + night_rain["cyclist_count"]
    assert night_rain["total_vru"] > 0

    day = EnvRiskBridge.scene_to_scenario(scenes_by_id["scene-0061"])  # daytime
    assert day["IS_NIGHT"] == 0
    assert day["ADVERSE_WEATHER"] == 0
    assert day["HOUR"] == 14


def test_night_amplifies_day(bridge, scenes_by_id):
    day = bridge.assess(scenes_by_id["scene-0061"])      # clear day
    night = bridge.assess(scenes_by_id["scene-1077"])    # night
    night_rain = bridge.assess(scenes_by_id["scene-1094"])  # night + rain

    # Adverse env -> lower safety score -> higher multiplier.
    assert night.safety_score < day.safety_score
    assert night.multiplier > day.multiplier
    assert night_rain.multiplier >= night.multiplier
    # Night scenes should land in the amplify region (>1), day near/below neutral.
    assert night.multiplier > 1.0
    assert day.multiplier <= 1.05
    assert night.risk_level == "Critical"


def test_multiplier_always_in_bounds(bridge, scenes_by_id):
    for sc in scenes_by_id.values():
        r = bridge.assess(sc)
        assert bridge.mult_min <= r.multiplier <= bridge.mult_max
        assert 0.0 <= r.env_risk <= 1.0
        assert 0.0 <= r.safety_score <= 100.0


def test_determinism(bridge, scenes_by_id):
    sc = scenes_by_id["scene-1100"]
    assert bridge.assess(sc).safety_score == bridge.assess(sc).safety_score


def test_score_band_regression(bridge, scenes_by_id):
    """Regression guard (sklearn pinned to 1.8.0): bands the model currently produces."""
    day = bridge.assess(scenes_by_id["scene-0061"]).safety_score
    night_rain = bridge.assess(scenes_by_id["scene-1094"]).safety_score
    assert 48.0 <= day <= 60.0, f"day score drifted: {day}"
    assert 10.0 <= night_rain <= 24.0, f"night+rain score drifted: {night_rain}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
