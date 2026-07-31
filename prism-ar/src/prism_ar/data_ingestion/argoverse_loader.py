"""Argoverse 2 loader using pyarrow/pandas (no av2/torch dependency)."""
from __future__ import annotations

import os
import glob
from typing import List, Optional
import numpy as np
import pandas as pd

from .driving_scene import DrivingScene, Agent, SceneAttributes


# Argoverse 2 parquet schema (discovered from local data):
# observed, track_id, object_type, object_category, timestep,
# position_x, position_y, heading, velocity_x, velocity_y,
# scenario_id, start_timestamp, end_timestamp, num_timestamps, focal_track_id, city

OBJECT_TYPE_MAP = {
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "motorcyclist": "vehicle",
    "cyclist": "cyclist",
    "bus": "vehicle",
    "large_vehicle": "vehicle",
    "static": "other",
    "background": "other",
    "construction": "other",
    "riderless_bicycle": "cyclist",
    "unknown": "other",
}


class ArgoverseLoader:
    """Load Argoverse 2 motion forecasting scenarios into DrivingScene objects."""

    def __init__(self, argoverse_root: str):
        self.argoverse_root = argoverse_root

    def _find_parquet_files(self) -> List[str]:
        pattern = os.path.join(self.argoverse_root, "val", "*", "scenario_*.parquet")
        return sorted(glob.glob(pattern))

    def _parquet_to_scene(self, path: str) -> DrivingScene:
        """Parse a single Argoverse 2 scenario parquet into DrivingScene."""
        df = pd.read_parquet(path)
        scenario_id = str(df["scenario_id"].iloc[0])
        focal_track_id = int(df["focal_track_id"].iloc[0])
        city = str(df["city"].iloc[0])

        # Determine ego agent: choose the longest vehicle track as the ego AV.
        # The focal track is the prediction target, not necessarily the ego.
        vehicle_tracks = []
        for track_id, track_df in df.groupby("track_id"):
            otype = str(track_df["object_type"].iloc[0]).lower()
            if OBJECT_TYPE_MAP.get(otype, "other") == "vehicle" and track_df["observed"].any():
                vehicle_tracks.append((track_id, track_df))
        if vehicle_tracks:
            # Pick the vehicle with the most observed timesteps as ego
            ego_track_id, ego_df = max(vehicle_tracks, key=lambda t: t[1]["observed"].sum())
        else:
            ego_df = df[df["observed"]].sort_values("timestep")

        timesteps = ego_df["timestep"].values
        n_frames = len(timesteps)
        if n_frames == 0:
            return None

        timestamps = timesteps / 10.0  # Argoverse 2 is 10 Hz

        agents = {}

        def _track_to_agent(track_df: pd.DataFrame) -> Optional[Agent]:
            track_df = track_df.sort_values("timestep")
            if track_df.empty:
                return None
            # Interpolate to ego timesteps to align frames
            tx = np.interp(timesteps, track_df["timestep"].values, track_df["position_x"].values)
            ty = np.interp(timesteps, track_df["timestep"].values, track_df["position_y"].values)
            vx = np.interp(timesteps, track_df["timestep"].values, track_df["velocity_x"].values)
            vy = np.interp(timesteps, track_df["timestep"].values, track_df["velocity_y"].values)
            yaw = np.interp(timesteps, track_df["timestep"].values, track_df["heading"].values)
            otype = str(track_df["object_type"].iloc[0]).lower()
            return Agent(
                agent_id=str(track_df["track_id"].iloc[0]),
                agent_type=OBJECT_TYPE_MAP.get(otype, "other"),
                positions=np.column_stack([tx, ty]),
                velocities=np.column_stack([vx, vy]),
                timestamps=timestamps,
                yaw=yaw,
                width=float(track_df["velocity_x"].abs().median()) * 0.1 + 1.0,
                length=float(track_df["velocity_x"].abs().median()) * 0.2 + 2.0,
            )

        # Ego
        ego_agent = _track_to_agent(ego_df)
        if ego_agent is not None:
            ego_agent.agent_type = "ego"
            ego_agent.agent_id = "ego"
            agents["ego"] = ego_agent

        # Other agents
        for track_id, track_df in df.groupby("track_id"):
            if track_id == focal_track_id:
                continue
            agent = _track_to_agent(track_df)
            if agent is not None and agent.agent_type in ("vehicle", "pedestrian", "cyclist"):
                agents[str(track_id)] = agent

        return DrivingScene(
            scene_id=scenario_id,
            dataset="argoverse",
            agents=agents,
            attributes=SceneAttributes(
                weather="clear",
                lighting="day",
                road_condition="dry",
                location=city,
            ),
            frames=n_frames,
            sampling_rate=10.0,
        )

    def load_scenes(self, max_scenes: Optional[int] = None) -> List[DrivingScene]:
        """Load Argoverse 2 scenes from the validation split."""
        files = self._find_parquet_files()
        if not files:
            return []

        scenes = []
        for path in files:
            scene = self._parquet_to_scene(path)
            if scene is not None:
                scenes.append(scene)
            if max_scenes is not None and len(scenes) >= max_scenes:
                break
        return scenes

    def make_synthetic_scene(self, scene_id: str = "av2_synthetic_001", n_frames: int = 110) -> DrivingScene:
        """Create a synthetic Argoverse-style scene for testing."""
        t = np.arange(n_frames) / 10.0
        ego_x = np.linspace(0.0, 60.0, n_frames)
        ego_y = np.zeros(n_frames)
        ego_vx = np.full(n_frames, 5.5)
        ego_vy = np.zeros(n_frames)
        cyc_x = np.full(n_frames, 6.0)
        cyc_y = np.linspace(-2.0, 8.0, n_frames)
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
            "cyc": Agent(
                agent_id="cyc",
                agent_type="cyclist",
                positions=np.column_stack([cyc_x, cyc_y]),
                velocities=np.column_stack([np.zeros(n_frames), np.full(n_frames, 0.09)]),
                timestamps=t,
                width=0.6,
                length=1.7,
            ),
        }
        return DrivingScene(
            scene_id=scene_id,
            dataset="argoverse",
            agents=agents,
            attributes=SceneAttributes(weather="clear", lighting="day", road_condition="dry", location="synthetic"),
            frames=n_frames,
            sampling_rate=10.0,
        )
