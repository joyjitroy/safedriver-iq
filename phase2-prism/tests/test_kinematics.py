"""M3 tests — trajectory kinematics + LSTM risk model.

The pure-kinematics tests (feature math, heuristic monotonicity) are exact and fast.
The model tests use small synthetic datasets and a short train so they run in seconds
on CPU and don't depend on the full AV2 training artifact.
"""
from __future__ import annotations

import numpy as np
import torch

from sdiq.data_loader import AgentState, AgentTrack, DrivingScene, _fill_velocity
from sdiq.kinematics import (
    N_FEATURES, extract_kinematics, kinematic_risk, track_risk,
)


# ---------------------------------------------------------------------------
# synthetic track builders
# ---------------------------------------------------------------------------
def _track(xs, ys, dt=0.1, otype="vehicle", tid="t") -> AgentTrack:
    states = [AgentState(t=i * dt, x=float(x), y=float(y))
              for i, (x, y) in enumerate(zip(xs, ys))]
    _fill_velocity(states)
    return AgentTrack(tid, otype, states)


def _smooth_track(T=80, v=10.0, tid="s"):
    t = np.arange(T) * 0.1
    return _track(v * t, np.zeros(T), tid=tid)


def _aggressive_track(T=80, tid="a"):
    """Repeated hard braking + swerving so future windows carry high risk."""
    t = np.arange(T) * 0.1
    # speed oscillates hard (accel/brake ~6 m/s^2); lateral swerve too
    v = 8 + 6 * np.sin(2 * np.pi * t / 2.0)
    x = np.cumsum(np.clip(v, 0, None)) * 0.1
    y = 1.5 * np.sin(2 * np.pi * t / 2.0)
    return _track(x, y, tid=tid)


def _scene(tracks, sid="syn", rate=10.0) -> DrivingScene:
    ego = _smooth_track(tid="ego").states
    return DrivingScene(scene_id=sid, source="argoverse2", ego=ego,
                        agents=tracks, sample_rate_hz=rate)


# ---------------------------------------------------------------------------
# kinematics (pure)
# ---------------------------------------------------------------------------
def test_extract_shape_and_speed():
    trk = _smooth_track(T=60, v=12.0)
    f = extract_kinematics(trk)
    assert f.shape == (60, N_FEATURES)
    # constant 12 m/s straight line -> speed ~12, ~zero accel/yaw
    assert abs(f[5:-5, 0].mean() - 12.0) < 0.5
    assert np.abs(f[5:-5, 1]).max() < 0.5      # lon_accel
    assert np.abs(f[5:-5, 3]).max() < 0.1      # yaw_rate


def test_too_short_track_is_zeros():
    trk = _track([0, 1], [0, 0])               # length 2 < MIN_TRACK_LEN
    assert extract_kinematics(trk).shape[0] <= 2
    assert track_risk(trk) == 0.0


def test_heuristic_monotonicity():
    smooth = track_risk(_smooth_track())
    aggressive = track_risk(_aggressive_track())
    assert smooth < 0.05, f"smooth driving should be ~0 risk, got {smooth}"
    assert aggressive > 0.5, f"aggressive driving should be high risk, got {aggressive}"
    assert aggressive > smooth


def test_risk_bounded():
    r = kinematic_risk(extract_kinematics(_aggressive_track()))
    assert r.min() >= 0.0 and r.max() <= 1.0


# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------
def test_model_forward_shape():
    from sdiq.kinematic_features import KinematicRiskLSTM
    m = KinematicRiskLSTM()
    x = torch.randn(8, 20, N_FEATURES)
    out = m(x)
    assert out.shape == (8,)


def test_build_windows_shapes():
    from sdiq.kinematic_features import build_windows, feature_stats
    scenes = [_scene([_smooth_track(tid=f"s{i}"), _aggressive_track(tid=f"a{i}")])
              for i in range(6)]
    X, y = build_windows(scenes, history=20, horizon=20, stride=5)
    assert X.ndim == 3 and X.shape[1:] == (20, N_FEATURES)
    assert len(X) == len(y)
    assert (y >= 0).all() and (y <= 1).all()
    mean, std = feature_stats(X)
    assert std.min() > 0  # never zero (guards standardization)


def test_train_and_behavioral_discrimination():
    """Tiny end-to-end train, then assert aggressive tracks score above smooth ones."""
    from sdiq.kinematic_features import train, KinematicRiskModel

    scenes = [_scene([_smooth_track(tid=f"s{i}"), _smooth_track(v=8, tid=f"s2{i}"),
                      _aggressive_track(tid=f"a{i}")]) for i in range(40)]
    model, report, meta = train(scenes=scenes, epochs=40, batch_size=128,
                                save_path="/tmp/kin_test.pt", seed=0)
    assert report.n_train > 0
    km = KinematicRiskModel.from_model(model, meta)

    smooth_scores = [km.score_track(_smooth_track(v=v, tid=f"vs{v}")) for v in (6, 9, 11)]
    agg_scores = [km.score_track(_aggressive_track(tid=f"va{k}")) for k in range(3)]
    assert all(0.0 <= s <= 1.0 for s in smooth_scores + agg_scores)
    assert np.mean(agg_scores) > np.mean(smooth_scores) + 0.2, \
        f"poor discrimination: agg={np.mean(agg_scores):.2f} smooth={np.mean(smooth_scores):.2f}"


def test_inference_roundtrip_and_scene():
    from sdiq.kinematic_features import train, KinematicRiskModel

    scenes = [_scene([_smooth_track(tid=f"s{i}"), _aggressive_track(tid=f"a{i}")])
              for i in range(20)]
    _, _, meta = train(scenes=scenes, epochs=5, save_path="/tmp/kin_rt.pt", seed=1)
    km = KinematicRiskModel(path="/tmp/kin_rt.pt")            # load from disk
    out = km.score_scene(_scene([_smooth_track(), _aggressive_track()]))
    assert set(out) == {"per_agent", "max_risk", "mean_risk"}
    assert 0.0 <= out["max_risk"] <= 1.0
    assert len(out["per_agent"]) == 2


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
