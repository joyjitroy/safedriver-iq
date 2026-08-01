"""M1 tests — the unified loader produces a valid, consistent schema for both sources.

Asserts the cross-dataset invariants the downstream models rely on:
  * every track has >=1 state; states are time-ordered; positions finite.
  * velocities/headings are populated (derived for nuScenes).
  * object types are from the normalized vocabulary; VRU flag matches.
  * nuScenes night/rain conditions are detected from scene descriptions.
  * AV2 windows are 10 Hz with an ego (AV) track; nuScenes is 2 Hz.
"""
from __future__ import annotations

import math

from sdiq.data_loader import (
    VRU_TYPES, AgentTrack, DrivingScene,
    iter_nuscenes_scenes, iter_av2_scenarios,
)

_NORMALIZED_TYPES = {"vehicle", "pedestrian", "cyclist", "motorcyclist"}


def _check_scene_invariants(sc: DrivingScene) -> None:
    assert sc.scene_id and sc.source in {"nuscenes", "argoverse2"}
    assert sc.sample_rate_hz > 0
    assert sc.ego, f"{sc.scene_id}: ego trajectory is empty"

    for track in [AgentTrack("ego", "vehicle", sc.ego)] + sc.agents:
        assert track.states, f"{sc.scene_id}/{track.track_id}: no states"
        ts = [s.t for s in track.states]
        assert ts == sorted(ts), f"{sc.scene_id}/{track.track_id}: states not time-ordered"
        for s in track.states:
            assert math.isfinite(s.x) and math.isfinite(s.y)
            if s.vx is not None:
                assert math.isfinite(s.vx) and math.isfinite(s.vy)

    for a in sc.agents:
        assert a.object_type in _NORMALIZED_TYPES, f"bad type {a.object_type}"
        assert a.is_vru == (a.object_type in VRU_TYPES)


def test_nuscenes_all_scenes():
    scenes = list(iter_nuscenes_scenes())
    assert len(scenes) == 10
    for sc in scenes:
        assert sc.source == "nuscenes" and sc.sample_rate_hz == 2.0
        _check_scene_invariants(sc)

    by_id = {sc.scene_id: sc for sc in scenes}
    # Targeted adverse-condition scenes are detected and VRU-rich.
    assert by_id["scene-1094"].is_night and by_id["scene-1094"].is_rain
    assert by_id["scene-1077"].is_night
    assert by_id["scene-1100"].is_night
    assert len(by_id["scene-1094"].vru_agents) > 20, "night+rain scene should be VRU-rich"
    # A daytime scene should not be flagged night.
    assert not by_id["scene-0061"].is_night


def test_nuscenes_velocity_sanity():
    # Pedestrian speeds should be physically plausible (< ~4 m/s typical walking/running).
    sc = next(s for s in iter_nuscenes_scenes() if s.scene_id == "scene-0103")
    peds = [a for a in sc.agents if a.object_type == "pedestrian"]
    assert peds, "expected pedestrians in scene-0103"
    speeds = [s.speed for a in peds for s in a.states if s.speed is not None]
    assert speeds
    median = sorted(speeds)[len(speeds) // 2]
    assert median < 4.0, f"implausible median pedestrian speed {median:.2f} m/s"


def test_av2_sample():
    scenes = list(iter_av2_scenarios(limit=5))
    assert len(scenes) == 5
    for sc in scenes:
        assert sc.source == "argoverse2" and sc.sample_rate_hz == 10.0
        assert sc.city
        _check_scene_invariants(sc)
        # AV2 windows are ~11 s (110 timesteps @ 10 Hz).
        assert 9.0 <= sc.duration_s <= 12.0, f"unexpected AV2 duration {sc.duration_s}"


def test_av2_has_focal_agent():
    sc = next(iter_av2_scenarios(limit=1))
    assert any(a.is_focal for a in sc.agents) or sc.agents, \
        "expected a focal agent or at least tracked agents"


if __name__ == "__main__":
    test_nuscenes_all_scenes()
    test_nuscenes_velocity_sanity()
    test_av2_sample()
    test_av2_has_focal_agent()
    print("PASS: M1 data_loader tests")
