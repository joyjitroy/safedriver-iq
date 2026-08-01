"""Waymo Open Motion Dataset loader using tfrecord + protobuf (no TensorFlow)."""
from __future__ import annotations

import os
import glob
from typing import List, Optional
import numpy as np
import tfrecord

from .driving_scene import DrivingScene, Agent, SceneAttributes
from .tf_example_proto import parse_example


# Waymo tf.example feature names (WOMD v1.2)
WAYMO_FEATURES = {
    "id": "scenario/id",
    "state_id": "state/id",
    "state_type": "state/type",
    "is_sdc": "state/is_sdc",
    "past_x": "state/past/x",
    "past_y": "state/past/y",
    "past_vx": "state/past/velocity_x",
    "past_vy": "state/past/velocity_y",
    "past_yaw": "state/past/bbox_yaw",
    "past_valid": "state/past/valid",
    "past_length": "state/past/length",
    "past_width": "state/past/width",
    "current_x": "state/current/x",
    "current_y": "state/current/y",
    "current_vx": "state/current/velocity_x",
    "current_vy": "state/current/velocity_y",
    "current_yaw": "state/current/bbox_yaw",
    "current_valid": "state/current/valid",
    "current_length": "state/current/length",
    "current_width": "state/current/width",
    "future_x": "state/future/x",
    "future_y": "state/future/y",
    "future_vx": "state/future/velocity_x",
    "future_vy": "state/future/velocity_y",
    "future_yaw": "state/future/bbox_yaw",
    "future_valid": "state/future/valid",
    "future_length": "state/future/length",
    "future_width": "state/future/width",
}


def _classify_by_size(lengths: np.ndarray, widths: np.ndarray, valid: np.ndarray) -> str:
    """Infer object type from median dimensions."""
    if valid.sum() == 0:
        return "other"
    med_len = float(np.median(lengths[valid > 0]))
    med_wid = float(np.median(widths[valid > 0]))
    if med_len >= 3.0 or med_wid >= 1.5:
        return "vehicle"
    if med_len < 1.0 and med_wid < 1.0:
        return "pedestrian"
    return "cyclist"


def _concat_state(past: np.ndarray, current: np.ndarray, future: np.ndarray) -> np.ndarray:
    """Concatenate past [N,10], current [N,1], future [N,80] -> [N,91]."""
    return np.concatenate([past, current, future], axis=1)


class WaymoLoader:
    """Load Waymo WOMD tf.example scenarios into DrivingScene objects."""

    def __init__(self, waymo_root: str):
        self.waymo_root = waymo_root

    def _find_tfrecords(self, split: str = "validation") -> List[str]:
        base = os.path.join(self.waymo_root, "tf_example_datasets", split)
        pattern = os.path.join(base, "*.tfrecord*")
        return sorted(glob.glob(pattern))

    def _parse_example(self, raw: bytes) -> Optional[DrivingScene]:
        """Parse one tf.example record into DrivingScene."""
        parsed = parse_example(raw)
        scenario_id = parsed[WAYMO_FEATURES["id"]][0].decode("utf-8")

        state_id = parsed[WAYMO_FEATURES["state_id"]].astype(np.int64)
        state_type = parsed[WAYMO_FEATURES["state_type"]].astype(np.int64)
        is_sdc = parsed[WAYMO_FEATURES["is_sdc"]].astype(np.int64)
        n_agents = len(state_id)
        if n_agents == 0:
            return None

        def _load(prefix: str):
            x = parsed[WAYMO_FEATURES[f"{prefix}_x"]].reshape(n_agents, -1)
            y = parsed[WAYMO_FEATURES[f"{prefix}_y"]].reshape(n_agents, -1)
            vx = parsed[WAYMO_FEATURES[f"{prefix}_vx"]].reshape(n_agents, -1)
            vy = parsed[WAYMO_FEATURES[f"{prefix}_vy"]].reshape(n_agents, -1)
            yaw = parsed[WAYMO_FEATURES[f"{prefix}_yaw"]].reshape(n_agents, -1)
            valid = parsed[WAYMO_FEATURES[f"{prefix}_valid"]].reshape(n_agents, -1).astype(bool)
            length = parsed[WAYMO_FEATURES[f"{prefix}_length"]].reshape(n_agents, -1)
            width = parsed[WAYMO_FEATURES[f"{prefix}_width"]].reshape(n_agents, -1)
            return x, y, vx, vy, yaw, valid, length, width

        past = _load("past")
        current = _load("current")
        future = _load("future")

        X = _concat_state(past[0], current[0], future[0])
        Y = _concat_state(past[1], current[1], future[1])
        VX = _concat_state(past[2], current[2], future[2])
        VY = _concat_state(past[3], current[3], future[3])
        Yaw = _concat_state(past[4], current[4], future[4])
        Valid = _concat_state(past[5], current[5], future[5]).astype(bool)
        L = _concat_state(past[6], current[6], future[6])
        W = _concat_state(past[7], current[7], future[7])

        n_steps = X.shape[1]
        dt = 0.1
        timestamps = np.arange(n_steps) * dt

        # Find ego (self-driving car)
        sdc_mask = is_sdc == 1
        ego_idx = int(np.argmax(sdc_mask)) if np.any(sdc_mask) else 0

        agents = {}

        def _track_to_agent(idx: int) -> Optional[Agent]:
            if not Valid[idx].any():
                return None
            otype = _classify_by_size(L[idx], W[idx], Valid[idx].astype(np.int64))
            if otype == "other":
                return None
            return Agent(
                agent_id=str(state_id[idx]),
                agent_type=otype,
                positions=np.column_stack([X[idx], Y[idx]]),
                velocities=np.column_stack([VX[idx], VY[idx]]),
                timestamps=timestamps,
                yaw=Yaw[idx],
                width=float(np.median(W[idx][Valid[idx]])),
                length=float(np.median(L[idx][Valid[idx]])),
            )

        ego_agent = _track_to_agent(ego_idx)
        if ego_agent is not None:
            ego_agent.agent_type = "ego"
            ego_agent.agent_id = "ego"
            agents["ego"] = ego_agent

        for i in range(n_agents):
            if i == ego_idx:
                continue
            agent = _track_to_agent(i)
            if agent is not None:
                agents[str(state_id[i])] = agent

        return DrivingScene(
            scene_id=scenario_id,
            dataset="waymo",
            agents=agents,
            attributes=SceneAttributes(
                weather="clear",
                lighting="day",
                road_condition="dry",
                location="waymo",
            ),
            frames=n_steps,
            sampling_rate=10.0,
        )

    def load_scenes(
        self,
        split: str = "validation",
        max_scenes: Optional[int] = None,
    ) -> List[DrivingScene]:
        """Load Waymo scenarios from tf.example TFRecord files."""
        files = self._find_tfrecords(split)
        if not files:
            return []
        scenes = []
        for path in files:
            for raw in tfrecord.reader.tfrecord_iterator(path):
                try:
                    scene = self._parse_example(raw)
                except Exception:
                    continue
                if scene is not None:
                    scenes.append(scene)
                if max_scenes is not None and len(scenes) >= max_scenes:
                    return scenes
        return scenes

    def make_synthetic_scene(self, scene_id: str = "waymo_synthetic_001", n_frames: int = 91) -> DrivingScene:
        """Create a synthetic Waymo-style scene for testing."""
        t = np.arange(n_frames) / 10.0
        ego_x = np.linspace(0.0, 50.0, n_frames)
        ego_y = np.zeros(n_frames)
        ego_vx = np.full(n_frames, 5.0)
        ego_vy = np.zeros(n_frames)
        ped_x = np.full(n_frames, 8.0)
        ped_y = np.linspace(0.0, 5.0, n_frames)
        agents = {
            "ego": Agent(
                agent_id="ego",
                agent_type="ego",
                positions=np.column_stack([ego_x, ego_y]),
                velocities=np.column_stack([ego_vx, ego_vy]),
                timestamps=t,
                yaw=np.zeros(n_frames),
                width=1.8,
                length=4.5,
            ),
            "ped": Agent(
                agent_id="ped",
                agent_type="pedestrian",
                positions=np.column_stack([ped_x, ped_y]),
                velocities=np.column_stack([np.zeros(n_frames), np.full(n_frames, 0.05)]),
                timestamps=t,
                width=0.5,
                length=0.5,
            ),
        }
        return DrivingScene(
            scene_id=scene_id,
            dataset="waymo",
            agents=agents,
            attributes=SceneAttributes(weather="clear", lighting="day", road_condition="dry", location="synthetic"),
            frames=n_frames,
            sampling_rate=10.0,
        )
