"""M3 (part 1) — trajectory kinematics: feature extraction + derived risk heuristic.

Pure numpy + scipy, no torch, so it is fast and trivially unit-testable. Two jobs:

1. `extract_kinematics(track)` -> (T, 4) array of physical features per timestep:
   [speed, lon_accel, lat_accel, yaw_rate]. These are frame-invariant physical
   quantities (not raw positions), which lets a model trained on Argoverse 2 (10 Hz)
   transfer to nuScenes (2 Hz): derivatives are taken w.r.t. real time, and the
   Savitzky-Golay smoothing window is a fixed ~2 s in *time* (converted to a sample
   count per track), so both rates get equivalent denoising. (Jerk — a 3rd derivative —
   was dropped: it is noise-dominated even after smoothing.)

2. `kinematic_risk(feats)` -> (T,) derived risk in [0, 1]. There are no crash labels on
   these datasets, so risk is a heuristic over normalized exceedances of comfort/
   aggression thresholds (hard braking, hard accel, swerving, speeding), calibrated so
   normal driving ≈ 0. This is the SELF-SUPERVISION SIGNAL the LSTM learns to anticipate
   — and a documented limitation: the model learns a kinematic notion of risk, not
   observed collisions.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from sdiq.config import KINEMATICS as K, KIN_SMOOTH_SECONDS
from sdiq.data_loader import AgentTrack

N_FEATURES = 4
MIN_TRACK_LEN = 3       # need >=3 points for second derivatives (acceleration)
_POLYORDER = 2          # quadratic local fit -> smooth velocity & acceleration
_MIN_WINDOW = 5


def _window_for(dt: float, T: int) -> int:
    """Odd Savitzky-Golay window: ~KIN_SMOOTH_SECONDS of samples, clamped to the track."""
    win = int(round(KIN_SMOOTH_SECONDS / max(dt, 1e-3)))
    win = max(_MIN_WINDOW, win)
    win = min(win, T if T % 2 == 1 else T - 1)
    if win % 2 == 0:
        win += 1
    return max(win, _POLYORDER + 1 if (_POLYORDER + 1) % 2 else _POLYORDER + 2)


def extract_kinematics(track: AgentTrack) -> np.ndarray:
    """Return an (T, 4) feature array: [speed, lon_accel, lat_accel, yaw_rate].

    Velocity/acceleration are smoothed derivatives of position w.r.t. real time, so the
    result is consistent across Argoverse 2 (10 Hz) and nuScenes (2 Hz) and robust to
    sensor noise. Rows of zeros if the track is too short.
    """
    states = track.states
    T = len(states)
    if T < MIN_TRACK_LEN:
        return np.zeros((max(T, 0), N_FEATURES), dtype=np.float32)

    t = np.array([s.t for s in states], dtype=np.float64)
    x = np.array([s.x for s in states], dtype=np.float64)
    y = np.array([s.y for s in states], dtype=np.float64)
    dt = float(np.median(np.diff(t))) or 0.1

    if T >= _MIN_WINDOW:
        win = _window_for(dt, T)
        vx = savgol_filter(x, win, _POLYORDER, deriv=1, delta=dt)
        vy = savgol_filter(y, win, _POLYORDER, deriv=1, delta=dt)
        ax = savgol_filter(x, win, _POLYORDER, deriv=2, delta=dt)
        ay = savgol_filter(y, win, _POLYORDER, deriv=2, delta=dt)
    else:
        vx, vy = np.gradient(x, dt), np.gradient(y, dt)
        ax, ay = np.gradient(vx, dt), np.gradient(vy, dt)

    speed = np.hypot(vx, vy)

    # Course (direction of motion); freeze direction through near-stopped stretches.
    eps = 0.2  # m/s
    course = np.arctan2(vy, vx)
    moving = speed > eps
    if moving.any():
        idx = np.where(moving, np.arange(len(course)), 0)
        np.maximum.accumulate(idx, out=idx)
        course = course[idx]
    dir_x, dir_y = np.cos(course), np.sin(course)

    lon_accel = ax * dir_x + ay * dir_y       # along motion
    lat_accel = ax * dir_y - ay * dir_x       # signed cross-track (swerve/turn)
    yaw_rate = np.gradient(np.unwrap(course), dt)

    feats = np.stack([speed, lon_accel, lat_accel, yaw_rate], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def kinematic_risk(feats: np.ndarray) -> np.ndarray:
    """Per-timestep derived risk in [0, 1] from a (T, 4) feature array.

    Risk = saturating sum of normalized threshold-exceedances. A smooth, law-abiding
    trajectory scores ~0; hard braking / swerving / speeding push toward 1.
    """
    if feats.size == 0:
        return np.zeros((0,), dtype=np.float32)
    speed, lon_accel, lat_accel, yaw_rate = (feats[:, i] for i in range(N_FEATURES))

    exceed = (
        np.maximum(0.0, -lon_accel - K.hard_brake) +   # braking is negative lon_accel
        np.maximum(0.0, lon_accel - K.hard_accel) +
        np.maximum(0.0, np.abs(lat_accel) - K.lateral) +
        np.maximum(0.0, speed - K.speed)
    ) / K.scale

    # Saturate to [0, 1): smooth, monotonic in exceedance.
    return (1.0 - np.exp(-exceed)).astype(np.float32)


def track_risk(track: AgentTrack, reducer: str = "p90") -> float:
    """Scalar derived risk for a whole track (aggregate of per-step risk)."""
    risk = kinematic_risk(extract_kinematics(track))
    if risk.size == 0:
        return 0.0
    if reducer == "max":
        return float(risk.max())
    if reducer == "mean":
        return float(risk.mean())
    return float(np.percentile(risk, 90))  # p90: robust-to-noise "peak" risk
