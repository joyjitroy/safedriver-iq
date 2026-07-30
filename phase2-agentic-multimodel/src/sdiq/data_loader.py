"""M1 — unified data loaders for the Phase 2 POC.

Both nuScenes mini and Argoverse 2 are normalized into the SAME small set of
dataclasses so every downstream model (kinematic LSTM, VRU social-force, scenario
summary) is dataset-agnostic:

    DrivingScene
      ├── ego:    list[EgoState]      (ordered by time)
      └── agents: list[AgentTrack]
                    └── states: list[AgentState]

Conventions shared across sources:
  * positions are in a metric frame (meters), self-consistent within a scene
    (nuScenes = global map frame; AV2 = city frame). We never compare across scenes.
  * timestamps are floats in **seconds**, zeroed to the scene start (t0 = 0.0).
  * headings are radians; velocities are m/s in the scene frame.
  * object_type is normalized to: vehicle | pedestrian | cyclist | motorcyclist | other.
  * VRU == object_type in {pedestrian, cyclist, motorcyclist} (PMD/scooter -> pedestrian).

nuScenes annotations carry no velocity, so per-state velocity/heading are derived
from neighbouring annotations of the same instance (central difference).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from sdiq import config
from sdiq.nuscenes_mini import NuScenesMini

# VRU = vulnerable road users.
VRU_TYPES = frozenset({"pedestrian", "cyclist", "motorcyclist"})


# ---------------------------------------------------------------------------
# Common dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AgentState:
    t: float                       # seconds since scene start
    x: float                       # meters (scene frame)
    y: float
    heading: float | None = None   # radians
    vx: float | None = None        # m/s
    vy: float | None = None

    @property
    def speed(self) -> float | None:
        if self.vx is None or self.vy is None:
            return None
        return math.hypot(self.vx, self.vy)


@dataclass
class AgentTrack:
    track_id: str
    object_type: str               # normalized type
    states: list[AgentState] = field(default_factory=list)
    is_focal: bool = False         # AV2 "focal" agent; always False for nuScenes

    @property
    def is_vru(self) -> bool:
        return self.object_type in VRU_TYPES


# Ego shares the AgentState shape; alias keeps call sites self-documenting.
EgoState = AgentState


@dataclass
class DrivingScene:
    scene_id: str
    source: str                    # "nuscenes" | "argoverse2"
    ego: list[EgoState]
    agents: list[AgentTrack]
    sample_rate_hz: float
    city: str | None = None
    description: str | None = None  # nuScenes scene text; None for AV2
    is_night: bool = False
    is_rain: bool = False

    @property
    def vru_agents(self) -> list[AgentTrack]:
        return [a for a in self.agents if a.is_vru]

    @property
    def duration_s(self) -> float:
        ts = [s.t for s in self.ego] + [s.t for a in self.agents for s in a.states]
        return (max(ts) - min(ts)) if ts else 0.0

    def summary(self) -> str:
        n_vru = len(self.vru_agents)
        flags = []
        if self.is_night:
            flags.append("night")
        if self.is_rain:
            flags.append("rain")
        tag = f" [{', '.join(flags)}]" if flags else ""
        return (f"{self.source}:{self.scene_id}{tag} | {len(self.agents)} agents "
                f"({n_vru} VRU) | {self.duration_s:.1f}s @ {self.sample_rate_hz:g}Hz"
                f"{(' | ' + self.city) if self.city else ''}")


# ---------------------------------------------------------------------------
# nuScenes mini
# ---------------------------------------------------------------------------
_NUSC_PED_PREFIX = "human.pedestrian"


def _nusc_normalize_type(category_name: str) -> str:
    if category_name.startswith(_NUSC_PED_PREFIX):
        return "pedestrian"          # includes personal_mobility (PMD/scooter)
    if category_name == "vehicle.bicycle":
        return "cyclist"
    if category_name == "vehicle.motorcycle":
        return "motorcyclist"
    if category_name.startswith("vehicle."):
        return "vehicle"
    return "other"                   # movable_object.*, static_object.*, animal


def _quat_to_yaw(q: list[float]) -> float:
    """nuScenes quaternion [w, x, y, z] -> yaw (rotation about z), radians."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _detect_conditions(description: str) -> tuple[bool, bool]:
    d = description.lower()
    is_night = "night" in d
    is_rain = "rain" in d or "wet" in d
    return is_night, is_rain


def load_nuscenes_scene(nusc: NuScenesMini, scene: dict) -> DrivingScene:
    # Ordered keyframe samples for this scene (follow the linked list).
    samples: list[dict] = []
    token = scene["first_sample_token"]
    while token:
        s = nusc.get("sample", token)
        samples.append(s)
        token = s["next"]

    t0 = samples[0]["timestamp"] / 1e6  # microseconds -> seconds

    # Ego trajectory: take each keyframe's LIDAR_TOP ego_pose (canonical sample pose).
    ego: list[EgoState] = []
    for s in samples:
        sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        ego.append(EgoState(
            t=pose["timestamp"] / 1e6 - t0,
            x=pose["translation"][0],
            y=pose["translation"][1],
            heading=_quat_to_yaw(pose["rotation"]),
        ))
    _fill_velocity(ego)

    # Agents: group annotations by instance, ordered by sample time.
    sample_time = {s["token"]: s["timestamp"] / 1e6 - t0 for s in samples}
    by_instance: dict[str, list[tuple[float, dict]]] = {}
    for s in samples:
        for ann_token in s["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            by_instance.setdefault(ann["instance_token"], []).append(
                (sample_time[s["token"]], ann))

    agents: list[AgentTrack] = []
    for inst_token, items in by_instance.items():
        items.sort(key=lambda it: it[0])
        inst = nusc.get("instance", inst_token)
        cat = nusc.get("category", inst["category_token"])["name"]
        otype = _nusc_normalize_type(cat)
        if otype == "other":
            continue  # barriers, cones, debris, animals — not road agents
        states = [
            AgentState(t=t, x=ann["translation"][0], y=ann["translation"][1],
                       heading=_quat_to_yaw(ann["rotation"]))
            for t, ann in items
        ]
        _fill_velocity(states)
        agents.append(AgentTrack(track_id=inst_token, object_type=otype, states=states))

    is_night, is_rain = _detect_conditions(scene["description"])
    return DrivingScene(
        scene_id=scene["name"], source="nuscenes", ego=ego, agents=agents,
        sample_rate_hz=2.0, city=None, description=scene["description"],
        is_night=is_night, is_rain=is_rain,
    )


def _fill_velocity(states: list[AgentState]) -> None:
    """Central-difference velocity (and heading from motion if absent)."""
    n = len(states)
    for i in range(n):
        lo = states[max(0, i - 1)]
        hi = states[min(n - 1, i + 1)]
        dt = hi.t - lo.t
        if dt > 1e-6:
            states[i].vx = (hi.x - lo.x) / dt
            states[i].vy = (hi.y - lo.y) / dt
        else:
            states[i].vx = states[i].vy = 0.0


def iter_nuscenes_scenes(nusc: NuScenesMini | None = None) -> Iterator[DrivingScene]:
    nusc = nusc or NuScenesMini(config.NUSCENES_DATAROOT, config.NUSCENES_VERSION)
    for scene in nusc.scene:
        yield load_nuscenes_scene(nusc, scene)


# ---------------------------------------------------------------------------
# Argoverse 2 (motion forecasting val) — read parquet directly (devkit-free,
# robust). Schema columns: track_id, object_type, object_category, timestep,
# position_x/y, heading, velocity_x/y, city, scenario_id, focal_track_id, observed.
# ---------------------------------------------------------------------------
_AV2_TYPE_MAP = {
    "vehicle": "vehicle", "bus": "vehicle", "motorcyclist": "motorcyclist",
    "pedestrian": "pedestrian", "cyclist": "cyclist",
}
_AV2_DT = 0.1  # 10 Hz


def load_av2_scenario(parquet_path: str | Path) -> DrivingScene:
    import pandas as pd

    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)
    scenario_id = str(df["scenario_id"].iloc[0])
    city = str(df["city"].iloc[0]) if "city" in df.columns else None
    focal = str(df["focal_track_id"].iloc[0]) if "focal_track_id" in df.columns else None

    ego: list[EgoState] = []
    agents: list[AgentTrack] = []
    for track_id, g in df.groupby("track_id", sort=False):
        g = g.sort_values("timestep")
        raw_type = str(g["object_type"].iloc[0])
        states = [
            AgentState(
                t=float(row.timestep) * _AV2_DT,
                x=float(row.position_x), y=float(row.position_y),
                heading=float(row.heading) if "heading" in g.columns else None,
                vx=float(row.velocity_x) if "velocity_x" in g.columns else None,
                vy=float(row.velocity_y) if "velocity_y" in g.columns else None,
            )
            for row in g.itertuples(index=False)
        ]
        if str(track_id) == "AV":
            ego = states
            continue
        otype = _AV2_TYPE_MAP.get(raw_type, "other")
        if otype == "other":
            continue  # static, background, riderless_bicycle, construction, unknown
        agents.append(AgentTrack(
            track_id=str(track_id), object_type=otype, states=states,
            is_focal=(focal is not None and str(track_id) == focal),
        ))

    return DrivingScene(
        scene_id=scenario_id, source="argoverse2", ego=ego, agents=agents,
        sample_rate_hz=10.0, city=city, description=None,
        is_night=False, is_rain=False,
    )


def iter_av2_scenarios(root: str | Path | None = None,
                       limit: int | None = None) -> Iterator[DrivingScene]:
    root = Path(root) if root else config.AV2_VAL_ROOT
    count = 0
    for parquet in root.rglob("*.parquet"):
        yield load_av2_scenario(parquet)
        count += 1
        if limit is not None and count >= limit:
            return


if __name__ == "__main__":
    print("=== nuScenes mini ===")
    for sc in iter_nuscenes_scenes():
        print("  " + sc.summary())
    print("=== Argoverse 2 (first 3) ===")
    for sc in iter_av2_scenarios(limit=3):
        print("  " + sc.summary())
