"""M4 (part 1) — Social Force Model + ego-VRU interaction risk.

Pure numpy, no torch. The physics-based, training-free core of the VRU model (Helbing &
Molnár, 1995). For each vulnerable road user (pedestrian / cyclist) we:

1. Roll its short-horizon trajectory forward under social forces — a driving force toward
   its desired velocity plus exponential repulsion from neighbours and, most strongly,
   from the ego vehicle (peds give way to cars). This is a better forecast than constant
   velocity because the VRU *reacts* to the approaching ego.
2. Project the ego forward (constant velocity — what a real-time system would predict).
3. Compute the closest ego-VRU approach over the horizon -> (min_distance, TTC) and a
   near-miss flag, then an interaction risk in [0, 1].

Rate-agnostic: everything is in metres / seconds, horizons are specified in seconds and
converted to steps from each scene's own sample rate, so it works on Argoverse 2 (10 Hz)
and nuScenes (2 Hz) alike.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sdiq.config import SFM, VRU_RISK, VRU_HORIZON_S
from sdiq.data_loader import AgentState, AgentTrack, DrivingScene


# ---------------------------------------------------------------------------
# resampling onto a common scene time grid
# ---------------------------------------------------------------------------
def _states_arrays(states: list[AgentState]):
    t = np.array([s.t for s in states], dtype=np.float64)
    x = np.array([s.x for s in states], dtype=np.float64)
    y = np.array([s.y for s in states], dtype=np.float64)
    vx = np.array([(s.vx if s.vx is not None else 0.0) for s in states])
    vy = np.array([(s.vy if s.vy is not None else 0.0) for s in states])
    return t, x, y, vx, vy


def _resample(track: AgentTrack, grid: np.ndarray):
    """Linear-interpolate a track onto `grid` times. Returns pos(G,2), vel(G,2), valid(G,)."""
    if len(track.states) < 2:
        G = len(grid)
        return np.zeros((G, 2)), np.zeros((G, 2)), np.zeros(G, dtype=bool)
    t, x, y, vx, vy = _states_arrays(track.states)
    valid = (grid >= t[0] - 1e-9) & (grid <= t[-1] + 1e-9)
    pos = np.stack([np.interp(grid, t, x), np.interp(grid, t, y)], axis=1)
    vel = np.stack([np.interp(grid, t, vx), np.interp(grid, t, vy)], axis=1)
    return pos, vel, valid


# ---------------------------------------------------------------------------
# Social Force Model rollout
# ---------------------------------------------------------------------------
def _repulsion(pos: np.ndarray, sources: np.ndarray, a: float, b: float,
               r_sum: float) -> np.ndarray:
    """Sum of exponential repulsive accelerations on `pos` from `sources` (M,2)."""
    if len(sources) == 0:
        return np.zeros(2)
    diff = pos[None, :] - sources           # (M, 2) vectors source->ped
    dist = np.linalg.norm(diff, axis=1)
    dist = np.maximum(dist, 1e-3)
    n = diff / dist[:, None]
    mag = a * np.exp((r_sum - dist) / b)    # (M,)
    return (mag[:, None] * n).sum(axis=0)


def constant_velocity_rollout(vru_pos: np.ndarray, vru_vel: np.ndarray,
                              ego_future: np.ndarray,
                              neighbor_pos: np.ndarray, neighbor_vel: np.ndarray,
                              horizon: int, dt: float, params=SFM) -> np.ndarray:
    """Linear extrapolation of the VRU (no reaction). The conservative collision-course
    forecast used for the safety risk metric — same signature as `sfm_rollout` so the two
    are interchangeable as `forecaster` in `assess_scene_vru`."""
    steps = (np.arange(1, horizon + 1) * dt)[:, None]
    return vru_pos[None, :] + vru_vel[None, :] * steps


def sfm_rollout(vru_pos: np.ndarray, vru_vel: np.ndarray,
                ego_future: np.ndarray,
                neighbor_pos: np.ndarray, neighbor_vel: np.ndarray,
                horizon: int, dt: float, params=SFM) -> np.ndarray:
    """Forward-Euler SFM rollout of one VRU. Returns predicted positions (horizon, 2).

    ego_future: (horizon, 2) ego positions (the dominant repulsor).
    neighbor_pos/vel: (M, 2) other agents at t0, advanced at constant velocity.
    """
    pos = vru_pos.astype(np.float64).copy()
    vel = vru_vel.astype(np.float64).copy()
    desired_speed = min(np.linalg.norm(vel), params.v_max)
    e_dir = vel / max(np.linalg.norm(vel), 1e-6)        # desired direction (current heading)
    out = np.empty((horizon, 2))

    for k in range(horizon):
        # driving force toward desired velocity
        a_drive = (desired_speed * e_dir - vel) / params.tau
        # repulsion from ego (strong) and from neighbours (weak)
        a_ego = _repulsion(pos, ego_future[k][None, :], params.a_ego, params.b_ego,
                           params.radius_ped + params.radius_ego)
        nb = neighbor_pos + neighbor_vel * (k * dt) if len(neighbor_pos) else neighbor_pos
        a_nb = _repulsion(pos, nb, params.a_ped, params.b_ped, 2 * params.radius_ped)
        vel = vel + (a_drive + a_ego + a_nb) * dt
        sp = np.linalg.norm(vel)
        if sp > params.v_max:
            vel *= params.v_max / sp
        pos = pos + vel * dt
        out[k] = pos
    return out


# ---------------------------------------------------------------------------
# closest approach, TTC, interaction risk
# ---------------------------------------------------------------------------
def closest_approach(ego_traj: np.ndarray, vru_traj: np.ndarray, dt: float):
    """Return (min_distance, ttc_seconds, idx) of closest ego-VRU approach over horizon."""
    d = np.linalg.norm(ego_traj - vru_traj, axis=1)
    idx = int(np.argmin(d))
    return float(d[idx]), float(idx * dt), idx


def interaction_risk(min_dist: float, ttc: float, params=VRU_RISK) -> float:
    """Combine proximity and imminence into a risk in [0, 1]."""
    d_term = np.clip((params.d_warn - min_dist) / (params.d_warn - params.d_critical), 0, 1)
    imminence = np.clip((params.ttc_max - ttc) / params.ttc_max, 0, 1)
    return float(d_term * imminence)


# ---------------------------------------------------------------------------
# scene-level assessment
# ---------------------------------------------------------------------------
@dataclass
class VRUInteraction:
    track_id: str
    object_type: str
    risk: float
    min_distance: float
    ttc: float
    near_miss: bool


@dataclass
class VRUAssessment:
    interactions: list[VRUInteraction]

    @property
    def max_risk(self) -> float:
        return max((i.risk for i in self.interactions), default=0.0)

    @property
    def mean_risk(self) -> float:
        return float(np.mean([i.risk for i in self.interactions])) if self.interactions else 0.0

    @property
    def near_misses(self) -> int:
        return sum(1 for i in self.interactions if i.near_miss)


def _scene_grid(scene: DrivingScene):
    """Common time grid (from the ego clock) + resampled ego pos/vel."""
    if not scene.ego:
        return None
    grid = np.array([s.t for s in scene.ego], dtype=np.float64)
    epos = np.stack([[s.x, s.y] for s in scene.ego])
    evel = np.stack([[(s.vx or 0.0), (s.vy or 0.0)] for s in scene.ego])
    return grid, epos, evel


def assess_scene_vru(scene: DrivingScene, stride: int = 3,
                     horizon_s: float = VRU_HORIZON_S,
                     forecaster=None) -> VRUAssessment:
    """Worst-case ego-VRU interaction risk per VRU over the scene.

    For each VRU, slide assessment points over time (every `stride` grid steps); at each
    point project the ego at constant velocity and forecast the VRU. The risk metric uses
    a CONSTANT-VELOCITY VRU forecast by default — the conservative collision-course
    surrogate (standard TTC / minimum-distance safety measure: assume no one yields). Pass
    `forecaster=sfm_rollout` (or the SFM+LSTM model) for the realistic, reaction-aware
    forecast instead. Keep the worst (highest-risk / closest) interaction per VRU.
    """
    if forecaster is None:
        forecaster = constant_velocity_rollout
    g = _scene_grid(scene)
    if g is None:
        return VRUAssessment([])
    grid, epos, evel = g
    G = len(grid)
    if G < 2:
        return VRUAssessment([])
    dt = float(np.median(np.diff(grid)))
    horizon = max(1, int(round(horizon_s / dt)))

    vru_tracks = scene.vru_agents
    others = [a for a in scene.agents]      # neighbours = all agents (self excluded below)
    resampled = {a.track_id: _resample(a, grid) for a in others}

    interactions: list[VRUInteraction] = []
    for vru in vru_tracks:
        vpos, vvel, vvalid = resampled[vru.track_id]
        best = None
        for k in range(0, G - 1, stride):
            if not vvalid[k]:
                continue
            h = min(horizon, G - 1 - k)
            if h < 1:
                break
            # ego: constant-velocity projection from k
            ego_future = epos[k][None, :] + evel[k][None, :] * (np.arange(1, h + 1) * dt)[:, None]
            # neighbours at t0=k (exclude self)
            npos, nvel = [], []
            for a in others:
                if a.track_id == vru.track_id:
                    continue
                p, v, val = resampled[a.track_id]
                if val[k]:
                    npos.append(p[k]); nvel.append(v[k])
            npos = np.array(npos) if npos else np.zeros((0, 2))
            nvel = np.array(nvel) if nvel else np.zeros((0, 2))

            if forecaster is not None:
                vru_traj = forecaster(vpos[k], vvel[k], ego_future, npos, nvel, h, dt)
            else:
                vru_traj = sfm_rollout(vpos[k], vvel[k], ego_future, npos, nvel, h, dt)

            min_d, ttc, _ = closest_approach(ego_future, vru_traj, dt)
            risk = interaction_risk(min_d, ttc)
            if best is None or risk > best[0]:
                best = (risk, min_d, ttc)
        if best is not None:
            risk, min_d, ttc = best
            interactions.append(VRUInteraction(
                track_id=vru.track_id, object_type=vru.object_type, risk=risk,
                min_distance=min_d, ttc=ttc, near_miss=min_d < VRU_RISK.d_nearmiss))
    return VRUAssessment(interactions)


if __name__ == "__main__":
    from sdiq.data_loader import iter_nuscenes_scenes

    print("=== nuScenes ego-VRU interaction risk (constant-velocity collision-course) ===")
    for sc in iter_nuscenes_scenes():
        a = assess_scene_vru(sc)
        flags = ("N" if sc.is_night else "-") + ("R" if sc.is_rain else "-")
        print(f"  {sc.scene_id} [{flags}] vru={len(sc.vru_agents):3d} "
              f"max_risk={a.max_risk:.3f} mean_risk={a.mean_risk:.3f} near_misses={a.near_misses}")
