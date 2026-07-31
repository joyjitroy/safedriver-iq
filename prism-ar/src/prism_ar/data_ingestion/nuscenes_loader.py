"""nuScenes loader using JSON files (no nuscenes-devkit)."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np

from .driving_scene import DrivingScene, Agent, SceneAttributes


CATEGORY_MAP = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "vehicle.bicycle": "cyclist",
    "vehicle.motorcycle": "vehicle",
    "vehicle.car": "vehicle",
    "vehicle.truck": "vehicle",
    "vehicle.bus.bendy": "vehicle",
    "vehicle.bus.rigid": "vehicle",
    "vehicle.construction": "vehicle",
    "vehicle.emergency.ambulance": "vehicle",
    "vehicle.emergency.police": "vehicle",
    "vehicle.trailer": "vehicle",
    "movable_object.barrier": "other",
    "movable_object.debris": "other",
    "movable_object.pushable_pullable": "other",
    "movable_object.trafficcone": "other",
    "static_object.bicycle_rack": "other",
    "animal": "other",
}


def _quaternion_to_yaw(w, x, y, z) -> float:
    """Convert quaternion to yaw angle (z-axis rotation)."""
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class NuScenesLoader:
    """Load nuScenes mini scenes into DrivingScene objects from JSON files."""

    def __init__(self, nuscenes_root: str):
        self.nuscenes_root = nuscenes_root
        self.v1_dir = os.path.join(nuscenes_root, "v1.0-mini")
        self._tables_loaded = False
        if os.path.exists(self.v1_dir):
            self._load_tables()
            self._tables_loaded = True

    def _load_json(self, name: str):
        path = os.path.join(self.v1_dir, f"{name}.json")
        with open(path, "r") as f:
            return json.load(f)

    def _load_tables(self):
        self.scenes = {s["token"]: s for s in self._load_json("scene")}
        self.samples = {s["token"]: s for s in self._load_json("sample")}
        self.sample_data = self._load_json("sample_data")
        self.ego_poses = {ep["token"]: ep for ep in self._load_json("ego_pose")}
        self.annotations = self._load_json("sample_annotation")
        self.instances = {i["token"]: i for i in self._load_json("instance")}
        self.categories = {c["token"]: c for c in self._load_json("category")}
        self.logs = {l["token"]: l for l in self._load_json("log")}

        # Map sample -> lidar sample_data -> ego_pose
        self.sample_to_ego_pose = {}
        for sd in self.sample_data:
            if sd.get("channel") == "LIDAR_TOP" or "LIDAR_TOP" in sd.get("filename", ""):
                self.sample_to_ego_pose[sd["sample_token"]] = sd["ego_pose_token"]
        # Fallback: use any sample_data for the sample
        for sd in self.sample_data:
            if sd["sample_token"] not in self.sample_to_ego_pose:
                self.sample_to_ego_pose[sd["sample_token"]] = sd["ego_pose_token"]

        # Map sample -> annotations
        self.sample_to_annotations = defaultdict(list)
        for ann in self.annotations:
            self.sample_to_annotations[ann["sample_token"]].append(ann)

        # Build scene -> ordered samples
        self.scene_samples = defaultdict(list)
        for sample in self.samples.values():
            self.scene_samples[sample["scene_token"]].append(sample)
        for scene_token in self.scene_samples:
            self.scene_samples[scene_token].sort(key=lambda s: s["timestamp"])

    def _get_ego_pose(self, sample_token: str):
        pose_token = self.sample_to_ego_pose.get(sample_token)
        if pose_token is None:
            return None
        return self.ego_poses[pose_token]

    def _scene_to_drivingscene(self, scene_token: str) -> Optional[DrivingScene]:
        scene = self.scenes[scene_token]
        samples = self.scene_samples[scene_token]
        if not samples:
            return None

        n_frames = len(samples)
        timestamps = np.array([s["timestamp"] for s in samples])
        # Normalize to seconds from first frame
        timestamps_s = (timestamps - timestamps[0]) / 1e6
        sampling_rate = float(n_frames / timestamps_s[-1]) if timestamps_s[-1] > 0 else 2.0

        # Ego trajectory
        ego_positions = []
        ego_velocities = []
        ego_yaws = []
        for s in samples:
            pose = self._get_ego_pose(s["token"])
            if pose is None:
                ego_positions.append([0.0, 0.0])
                ego_velocities.append([0.0, 0.0])
                ego_yaws.append(0.0)
                continue
            trans = pose["translation"]
            rot = pose["rotation"]
            ego_positions.append([trans[0], trans[1]])
            ego_yaws.append(_quaternion_to_yaw(rot[0], rot[1], rot[2], rot[3]))
            # Compute velocity from position differences
            if len(ego_positions) > 1:
                dt = timestamps_s[len(ego_positions) - 1] - timestamps_s[len(ego_positions) - 2]
                if dt > 0:
                    dx = (ego_positions[-1][0] - ego_positions[-2][0]) / dt
                    dy = (ego_positions[-1][1] - ego_positions[-2][1]) / dt
                    ego_velocities[-1] = [dx, dy]
            ego_velocities.append([0.0, 0.0])
        # Make velocities same length as positions
        if len(ego_velocities) < len(ego_positions):
            ego_velocities.append([0.0, 0.0])
        ego_velocities = np.array(ego_velocities[:len(ego_positions)])

        agents = {
            "ego": Agent(
                agent_id="ego",
                agent_type="ego",
                positions=np.array(ego_positions),
                velocities=ego_velocities,
                timestamps=timestamps_s,
                yaw=np.array(ego_yaws),
                width=1.8,
                length=4.5,
            )
        }

        # Track other agents over frames by instance_token
        instance_tracks = defaultdict(lambda: {"positions": [], "yaws": [], "sizes": []})
        for i, s in enumerate(samples):
            for ann in self.sample_to_annotations.get(s["token"], []):
                inst_token = ann["instance_token"]
                cat_name = self.categories[self.instances[inst_token]["category_token"]]["name"]
                if cat_name not in CATEGORY_MAP:
                    continue
                if CATEGORY_MAP[cat_name] == "other":
                    continue
                trans = ann["translation"]
                size = ann["size"]
                rot = ann["rotation"]
                instance_tracks[inst_token]["positions"].append([trans[0], trans[1], i])
                instance_tracks[inst_token]["yaws"].append(_quaternion_to_yaw(rot[0], rot[1], rot[2], rot[3]))
                instance_tracks[inst_token]["sizes"].append(size)
                instance_tracks[inst_token]["category"] = CATEGORY_MAP[cat_name]

        # Convert tracks to Agent objects, interpolating missing frames
        for inst_token, track in instance_tracks.items():
            positions = np.array(track["positions"])
            observed_frames = positions[:, 2].astype(int)
            all_x = np.interp(np.arange(n_frames), observed_frames, positions[:, 0])
            all_y = np.interp(np.arange(n_frames), observed_frames, positions[:, 1])
            all_yaw = np.interp(np.arange(n_frames), observed_frames, np.array(track["yaws"]))
            # Velocity from position differences
            vx = np.gradient(all_x, timestamps_s)
            vy = np.gradient(all_y, timestamps_s)
            sizes = np.array(track["sizes"])
            width = float(np.median(sizes[:, 0]))
            length = float(np.median(sizes[:, 1]))
            agents[str(inst_token)] = Agent(
                agent_id=str(inst_token),
                agent_type=track["category"],
                positions=np.column_stack([all_x, all_y]),
                velocities=np.column_stack([vx, vy]),
                timestamps=timestamps_s,
                yaw=all_yaw,
                width=width,
                length=length,
            )

        # Get log info for scene attributes
        log = self.logs.get(scene.get("log_token"), {})
        description = scene.get("description", "").lower()
        is_night = "night" in description or log.get("time_of_day", "").lower() == "night"
        is_rain = "rain" in description or "wet" in description

        return DrivingScene(
            scene_id=scene["name"],
            dataset="nuscenes",
            agents=agents,
            attributes=SceneAttributes(
                weather="rain" if is_rain else "clear",
                lighting="night" if is_night else "day",
                road_condition="wet" if is_rain else "dry",
                location=log.get("location", "unknown"),
            ),
            frames=n_frames,
            sampling_rate=sampling_rate,
        )

    def load_scenes(self, max_scenes: Optional[int] = None) -> List[DrivingScene]:
        """Load nuScenes mini scenes."""
        if not self._tables_loaded:
            return []
        scenes = []
        for scene_token in self.scenes:
            scene = self._scene_to_drivingscene(scene_token)
            if scene is not None:
                scenes.append(scene)
            if max_scenes is not None and len(scenes) >= max_scenes:
                break
        return scenes

    def make_synthetic_scene(
        self,
        scene_id: str = "nuscenes_synthetic_001",
        n_frames: int = 40,
        lighting: str = "night",
        weather: str = "rain",
    ) -> DrivingScene:
        """Create a synthetic nuScenes-style scene for testing (adverse conditions)."""
        t = np.arange(n_frames) / 2.0
        ego_x = np.linspace(0.0, 20.0, n_frames)
        ego_y = np.zeros(n_frames)
        ego_vx = np.full(n_frames, 3.0)
        ego_vy = np.zeros(n_frames)
        ped_x = np.full(n_frames, 5.0)
        ped_y = np.linspace(0.0, 4.0, n_frames)
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
                velocities=np.column_stack([np.zeros(n_frames), np.full(n_frames, 0.1)]),
                timestamps=t,
                width=0.5,
                length=0.5,
            ),
        }
        return DrivingScene(
            scene_id=scene_id,
            dataset="nuscenes",
            agents=agents,
            attributes=SceneAttributes(
                weather=weather,
                lighting=lighting,
                road_condition="wet",
                location="synthetic",
            ),
            frames=n_frames,
            sampling_rate=2.0,
        )
