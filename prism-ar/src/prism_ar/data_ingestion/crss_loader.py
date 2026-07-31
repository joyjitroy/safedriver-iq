"""CRSS loader: converts crash records into synthetic DrivingScene objects."""
from __future__ import annotations

import os
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

from .driving_scene import DrivingScene, Agent, SceneAttributes


# Mapping from CRSS weather codes to labels
WEATHER_MAP = {
    0: "no_additional_atmospheric_conditions",
    1: "clear",
    2: "rain",
    3: "sleet",
    4: "snow",
    5: "fog",
    6: "smoke",
    7: "dust",
    10: "cloudy",
    11: "blowing_sand",
    12: "severe_crosswinds",
}

LIGHTING_MAP = {
    1: "daylight",
    2: "dark",
    3: "dark_lighted",
    4: "dawn",
    5: "dusk",
    6: "dark_unknown",
}

ROAD_CONDITION_MAP = {
    1: "dry",
    2: "wet",
    3: "snow",
    4: "ice",
    5: "sand",
    6: "mud",
    7: "oil",
    10: "slush",
    11: "water",
    12: "debris",
}


def _load_accident_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _get_crss_paths(crss_root: str) -> Dict[str, List[str]]:
    """Return paths to CRSS ACCIDENT.csv files by year."""
    paths = {}
    if not os.path.exists(crss_root):
        return paths
    for year in range(2016, 2024):
        year_dir = os.path.join(crss_root, str(year))
        accident_path = os.path.join(year_dir, "ACCIDENT.CSV")
        if os.path.exists(accident_path):
            paths.setdefault(str(year), []).append(accident_path)
    return paths


def crss_record_to_scene(
    row: pd.Series,
    scene_id: str,
    n_frames: int = 20,
    sampling_rate: float = 10.0,
) -> DrivingScene:
    """Convert a single CRSS accident record into a synthetic DrivingScene."""
    weather = WEATHER_MAP.get(row.get("WEATHER", 1), "clear")
    lighting = LIGHTING_MAP.get(row.get("LGT_COND", 1), "daylight")
    road_condition = ROAD_CONDITION_MAP.get(row.get("SUR_COND", 1), "dry")
    speed_limit = float(row.get("VSPD_LIM", 30)) if pd.notna(row.get("VSPD_LIM", None)) else 30.0

    attributes = SceneAttributes(
        weather=weather,
        lighting=lighting,
        road_condition=road_condition,
        speed_limit=speed_limit,
        time_of_day=str(row.get("HOUR", 12)),
        location="CRSS",
    )

    # Synthetic trajectory: ego braking toward a pedestrian
    t = np.arange(n_frames) / sampling_rate
    ego_speed = max(0.0, float(row.get("TRAV_SP", 30)) if pd.notna(row.get("TRAV_SP", None)) else 30.0)
    decel = 0.5
    ego_x = np.maximum(0.0, ego_speed * t - 0.5 * decel * t**2)
    ego_y = np.zeros(n_frames)
    ego_vel_x = np.maximum(0.0, ego_speed - decel * t)
    ego_vel_y = np.zeros(n_frames)

    agents = {
        "ego": Agent(
            agent_id="ego",
            agent_type="ego",
            positions=np.column_stack([ego_x, ego_y]),
            velocities=np.column_stack([ego_vel_x, ego_vel_y]),
            timestamps=t,
            yaw=np.zeros(n_frames),
            width=1.8,
            length=4.5,
        )
    }

    # Add a pedestrian if the record involves a pedestrian
    person_type = str(row.get("PER_TYP", ""))
    if person_type in ("1", "2", "3") or "PED" in str(row.get("PBPTYPE", "")).upper():
        ped_x = np.full(n_frames, 5.0)
        ped_y = np.linspace(0.0, 3.0, n_frames)
        agents["ped"] = Agent(
            agent_id="ped",
            agent_type="pedestrian",
            positions=np.column_stack([ped_x, ped_y]),
            velocities=np.column_stack([np.zeros(n_frames), np.full(n_frames, 0.15)]),
            timestamps=t,
            width=0.5,
            length=0.5,
        )

    return DrivingScene(
        scene_id=scene_id,
        dataset="crss",
        agents=agents,
        attributes=attributes,
        frames=n_frames,
        sampling_rate=sampling_rate,
    )


class CRSSLoader:
    """Load CRSS crash records into DrivingScene objects."""

    def __init__(self, crss_root: str):
        self.crss_root = crss_root

    def load_scenes(
        self,
        max_scenes: Optional[int] = None,
        n_frames: int = 20,
        sampling_rate: float = 10.0,
    ) -> List[DrivingScene]:
        """Load CRSS records as synthetic DrivingScene objects."""
        paths = _get_crss_paths(self.crss_root)
        if not paths:
            return []

        scenes = []
        counter = 0
        for year, file_list in paths.items():
            for path in file_list:
                df = _load_accident_csv(path)
                for _, row in df.iterrows():
                    scene_id = f"crss_{year}_{counter:06d}"
                    scene = crss_record_to_scene(row, scene_id, n_frames, sampling_rate)
                    scenes.append(scene)
                    counter += 1
                    if max_scenes is not None and len(scenes) >= max_scenes:
                        return scenes
        return scenes
