"""M4 tests — Social Force Model, ego-VRU interaction risk, and the LSTM correction.

The SFM/geometry tests are exact and fast. The model tests use small synthetic scenes and
a short train so they run in seconds and don't depend on the full AV2 artifact.
"""
from __future__ import annotations

import numpy as np
import torch

from sdiq.config import VRU_RISK
from sdiq.data_loader import AgentState, AgentTrack, DrivingScene, _fill_velocity
from sdiq.social_force import (
    assess_scene_vru, closest_approach, constant_velocity_rollout,
    interaction_risk, sfm_rollout,
)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _track(xy, dt=0.1, otype="vehicle", tid="t"):
    states = [AgentState(t=i * dt, x=float(x), y=float(y)) for i, (x, y) in enumerate(xy)]
    _fill_velocity(states)
    return AgentTrack(tid, otype, states)


def _crossing_scene(T=60, dt=0.1):
    """Ego drives +x along y=0; a pedestrian crosses the ego's path from the side."""
    t = np.arange(T) * dt
    ego = _track([(8.0 * tt, 0.0) for tt in t], dt=dt, otype="vehicle", tid="AV")
    # ego reaches x=10 at t=1.25s; pedestrian crosses x=10 reaching y=0 at the same time
    # -> near-collision geometry (ped starts (10,-3), walks +y at 2.4 m/s).
    ped = _track([(10.0, -3.0 + 2.4 * tt) for tt in t], dt=dt, otype="pedestrian", tid="ped")
    return DrivingScene("cross", "argoverse2", ego.states, [ped], sample_rate_hz=1 / dt)


def _far_scene(T=60, dt=0.1):
    """Pedestrian on a distant parallel sidewalk — no conflict."""
    t = np.arange(T) * dt
    ego = _track([(8.0 * tt, 0.0) for tt in t], dt=dt, otype="vehicle", tid="AV")
    ped = _track([(8.0 * tt, 30.0) for tt in t], dt=dt, otype="pedestrian", tid="ped")
    return DrivingScene("far", "argoverse2", ego.states, [ped], sample_rate_hz=1 / dt)


# ---------------------------------------------------------------------------
# geometry / risk (exact)
# ---------------------------------------------------------------------------
def test_closest_approach_math():
    # ego at origin moving +x; vru parallel offset 2m moving +x -> constant 2m gap
    ego = np.stack([np.arange(10) * 1.0, np.zeros(10)], axis=1)
    vru = np.stack([np.arange(10) * 1.0, np.full(10, 2.0)], axis=1)
    d, ttc, idx = closest_approach(ego, vru, dt=0.1)
    assert abs(d - 2.0) < 1e-6

    # head-on converging -> min distance ~0 (they pass through each other)
    ego2 = np.stack([np.arange(10) * 1.0, np.zeros(10)], axis=1)
    vru2 = np.stack([9 - np.arange(10) * 1.0, np.zeros(10)], axis=1)
    d2, _, _ = closest_approach(ego2, vru2, dt=0.1)
    assert d2 <= 1.0


def test_interaction_risk_monotonic():
    # closer + more imminent -> higher risk; clamped to [0,1]
    assert interaction_risk(0.5, 0.0) == 1.0                      # inside d_critical, now
    assert interaction_risk(VRU_RISK.d_warn + 1, 0.0) == 0.0      # beyond warn
    near = interaction_risk(1.5, 0.5)
    far = interaction_risk(4.0, 0.5)
    assert near > far
    # same distance, sooner is riskier
    assert interaction_risk(2.0, 0.2) > interaction_risk(2.0, 3.5)


def test_constant_velocity_rollout_shape():
    pos = np.array([1.0, 2.0]); vel = np.array([3.0, 0.0])
    traj = constant_velocity_rollout(pos, vel, None, np.zeros((0, 2)), np.zeros((0, 2)),
                                     horizon=5, dt=0.1)
    assert traj.shape == (5, 2)
    assert np.allclose(traj[0], [1.3, 2.0])     # 1 step of 0.1s at 3 m/s


def test_sfm_pedestrian_avoids_ego():
    """Under SFM, a pedestrian heading at the ego should be pushed away (larger min dist
    than constant velocity)."""
    ped_pos = np.array([0.0, 0.0]); ped_vel = np.array([1.0, 0.0])
    ego_future = np.stack([5 - np.arange(1, 21) * 0.25, np.zeros(20)], axis=1)  # ego approaching
    cv = constant_velocity_rollout(ped_pos, ped_vel, ego_future, np.zeros((0, 2)),
                                   np.zeros((0, 2)), 20, 0.1)
    sfm = sfm_rollout(ped_pos, ped_vel, ego_future, np.zeros((0, 2)), np.zeros((0, 2)), 20, 0.1)
    d_cv = np.linalg.norm(ego_future - cv, axis=1).min()
    d_sfm = np.linalg.norm(ego_future - sfm, axis=1).min()
    assert d_sfm > d_cv - 1e-6, "SFM should not bring the ped closer than constant velocity"


# ---------------------------------------------------------------------------
# scene assessment
# ---------------------------------------------------------------------------
def test_scene_risk_crossing_vs_far():
    cross = assess_scene_vru(_crossing_scene())
    far = assess_scene_vru(_far_scene())
    assert cross.max_risk > 0.5, f"crossing ped should be high risk, got {cross.max_risk}"
    assert far.max_risk < 0.1, f"distant ped should be ~0 risk, got {far.max_risk}"
    assert cross.near_misses >= 1
    assert far.near_misses == 0


def test_scene_only_scores_vrus():
    cross = _crossing_scene()
    a = assess_scene_vru(cross)
    # only the pedestrian (a VRU) is assessed; the ego/vehicles are not VRUs
    assert all(i.object_type in ("pedestrian", "cyclist", "motorcyclist")
               for i in a.interactions)


# ---------------------------------------------------------------------------
# LSTM correction
# ---------------------------------------------------------------------------
def test_model_forward_shape():
    from sdiq.vru_features import VRUForecastLSTM, N_VRU_FEATURES
    m = VRUForecastLSTM(horizon=20)
    out = m(torch.randn(4, 10, N_VRU_FEATURES))
    assert out.shape == (4, 20, 2)


def test_build_dataset_and_sfm_baseline():
    from sdiq.vru_features import build_vru_dataset, _ade_fde, N_VRU_FEATURES
    scenes = [_crossing_scene() for _ in range(8)] + [_far_scene() for _ in range(8)]
    ds = build_vru_dataset(scenes, history=10, horizon=20, stride=3)
    assert ds.X.shape[1:] == (10, N_VRU_FEATURES)
    assert ds.sfm.shape[1:] == (20, 2) and ds.actual.shape[1:] == (20, 2)
    ade, fde = _ade_fde(ds.sfm, ds.actual)
    assert ade >= 0 and fde >= 0


def test_train_improves_or_matches_sfm():
    """Tiny train: SFM+LSTM ADE should not be worse than SFM alone on held-out data."""
    from sdiq.vru_features import train, VRURiskModel
    # mix of crossing + straight-walking peds gives the LSTM a learnable residual
    scenes = [_crossing_scene() for _ in range(30)] + [_far_scene() for _ in range(30)]
    model, rep, meta = train(scenes=scenes, epochs=20, save_path="/tmp/vru_test.pt", seed=0)
    assert rep.n_train > 0
    assert rep.corrected_ade <= rep.sfm_ade + 0.05, \
        f"LSTM degraded the forecast: sfm={rep.sfm_ade:.3f} corrected={rep.corrected_ade:.3f}"
    # inference roundtrip
    km = VRURiskModel(path="/tmp/vru_test.pt")
    out = km.assess_scene(_crossing_scene())
    assert set(out) >= {"max_risk", "mean_risk", "near_misses", "realistic_max_risk"}
    assert 0.0 <= out["max_risk"] <= 1.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
