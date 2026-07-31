"""Extract AR-relevant VRU interaction clips from full DrivingScene objects.

Criteria:
- at least one VRU present
- ego-VRU distance decreasing at some point
- minimum distance below a threshold
- TTC computable
- optional: adverse conditions (night/rain)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from prism_ar.data_ingestion.driving_scene import DrivingScene, Agent


@dataclass
class ExtractionConfig:
    """Configuration for AR-relevant scenario extraction."""
    max_distance_m: float = 20.0      # VRU must come within this distance
    min_clip_duration_s: float = 3.0    # minimum clip length
    max_clip_duration_s: float = 10.0 # maximum clip length
    min_vru_frames: int = 5             # VRU must be visible for this many frames
    require_decreasing_distance: bool = True
    adverse_conditions_bonus: bool = True  # prefer night/rain
    near_miss_mode: bool = False         # if True, extract only the closest/lowest-TTC windows
    near_miss_distance_m: float = 10.0   # window must come within this distance
    near_miss_ttc_s: float = 3.0         # window must have TTC below this at least once


def _compute_ego_vru_distances(scene: DrivingScene) -> dict:
    """Return dict of {vru_id: distance_array} from ego to each VRU."""
    ego = scene.get_ego()
    if ego is None:
        return {}
    distances = {}
    for vru in scene.get_vrus():
        # Interpolate VRU positions to ego timestamps if needed
        if np.allclose(vru.timestamps, ego.timestamps):
            vru_pos = vru.positions
        else:
            x = np.interp(ego.timestamps, vru.timestamps, vru.positions[:, 0])
            y = np.interp(ego.timestamps, vru.timestamps, vru.positions[:, 1])
            vru_pos = np.column_stack([x, y])
        dist = np.linalg.norm(vru_pos - ego.positions, axis=1)
        distances[vru.agent_id] = dist
    return distances


def _compute_ego_vru_ttc(scene: DrivingScene) -> dict:
    """Return dict of {vru_id: ttc_array} for each VRU."""
    ego = scene.get_ego()
    if ego is None:
        return {}
    ttcs = {}
    for vru in scene.get_vrus():
        rel_pos = vru.positions - ego.positions
        rel_vel = vru.velocities - ego.velocities
        rel_speed = np.linalg.norm(rel_vel, axis=1)
        rel_distance = np.linalg.norm(rel_pos, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(rel_speed > 0.01, rel_distance / rel_speed, np.inf)
        ttcs[vru.agent_id] = ttc
    return ttcs


def _find_interaction_windows(
    distances: np.ndarray,
    ttc: np.ndarray,
    timestamps: np.ndarray,
    config: ExtractionConfig,
) -> List[tuple]:
    """Find time windows where distance is below threshold and decreasing."""
    threshold = config.near_miss_distance_m if config.near_miss_mode else config.max_distance_m
    close_mask = distances < threshold
    if not np.any(close_mask):
        return []

    if config.near_miss_mode:
        # Only keep windows where TTC drops below threshold at least once
        ttc_mask = ttc < config.near_miss_ttc_s
        close_mask = close_mask & (ttc_mask | np.roll(ttc_mask, 1) | np.roll(ttc_mask, -1))
        if not np.any(close_mask):
            return []

    # Find contiguous windows
    windows = []
    start = None
    for i, is_close in enumerate(close_mask):
        if is_close and start is None:
            start = i
        elif not is_close and start is not None:
            windows.append((start, i))
            start = None
    if start is not None:
        windows.append((start, len(close_mask)))

    # Filter by duration and decreasing distance
    valid_windows = []
    for s, e in windows:
        duration = timestamps[e - 1] - timestamps[s]
        if duration < config.min_clip_duration_s:
            continue
        if config.require_decreasing_distance:
            # Check if distance decreases in at least one sub-window
            window_dist = distances[s:e]
            if not np.any(np.diff(window_dist) < -0.01):
                continue
        # Trim to max duration centered around minimum distance
        min_idx = s + int(np.argmin(distances[s:e]))
        half = int(config.max_clip_duration_s * (len(timestamps) / (timestamps[-1] - timestamps[0])) / 2)
        half = max(5, half)
        s2 = max(0, min_idx - half)
        e2 = min(len(timestamps), min_idx + half)
        valid_windows.append((s2, e2))
    return valid_windows


def _extract_subscene(scene: DrivingScene, start_idx: int, end_idx: int, clip_id: str) -> DrivingScene:
    """Extract a sub-scene from start_idx to end_idx."""
    ego = scene.get_ego()
    new_agents = {}
    if ego is not None:
        new_agents["ego"] = Agent(
            agent_id="ego",
            agent_type="ego",
            positions=ego.positions[start_idx:end_idx].copy(),
            velocities=ego.velocities[start_idx:end_idx].copy(),
            timestamps=ego.timestamps[start_idx:end_idx].copy() - ego.timestamps[start_idx],
            yaw=ego.yaw[start_idx:end_idx].copy() if ego.yaw is not None else None,
            width=ego.width,
            length=ego.length,
        )

    for agent_id, agent in scene.agents.items():
        if agent_id == "ego":
            continue
        # Interpolate agent to ego timestamps if needed
        if np.allclose(agent.timestamps, ego.timestamps):
            pos = agent.positions[start_idx:end_idx].copy()
            vel = agent.velocities[start_idx:end_idx].copy()
            yaw = agent.yaw[start_idx:end_idx].copy() if agent.yaw is not None else None
        else:
            t = ego.timestamps[start_idx:end_idx]
            x = np.interp(t, agent.timestamps, agent.positions[:, 0])
            y = np.interp(t, agent.timestamps, agent.positions[:, 1])
            pos = np.column_stack([x, y])
            vx = np.interp(t, agent.timestamps, agent.velocities[:, 0])
            vy = np.interp(t, agent.timestamps, agent.velocities[:, 1])
            vel = np.column_stack([vx, vy])
            yaw = np.interp(t, agent.timestamps, agent.yaw) if agent.yaw is not None else None
        new_agents[agent_id] = Agent(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            positions=pos,
            velocities=vel,
            timestamps=ego.timestamps[start_idx:end_idx].copy() - ego.timestamps[start_idx],
            yaw=yaw,
            width=agent.width,
            length=agent.length,
        )

    return DrivingScene(
        scene_id=clip_id,
        dataset=scene.dataset,
        agents=new_agents,
        attributes=scene.attributes,
        frames=end_idx - start_idx,
        sampling_rate=scene.sampling_rate,
    )


class ScenarioExtractor:
    """Extract AR-relevant VRU interaction clips from full DrivingScene objects."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()

    def extract(self, scenes: List[DrivingScene]) -> List[DrivingScene]:
        """Extract AR-relevant clips from a list of scenes."""
        clips = []
        for scene in scenes:
            ego = scene.get_ego()
            if ego is None:
                continue
            vrus = scene.get_vrus()
            if not vrus:
                continue

            distances = _compute_ego_vru_distances(scene)
            ttcs = _compute_ego_vru_ttc(scene)
            for vru_id, dist in distances.items():
                ttc = ttcs.get(vru_id, np.full_like(dist, np.inf))
                windows = _find_interaction_windows(dist, ttc, ego.timestamps, self.config)
                for i, (s, e) in enumerate(windows):
                    # Check if VRU is visible in the window
                    vru = scene.agents[vru_id]
                    if np.allclose(vru.timestamps, ego.timestamps):
                        vru_visible = np.ones(e - s, dtype=bool)
                    else:
                        t = ego.timestamps[s:e]
                        vru_visible = np.array([np.any(np.abs(vru.timestamps - ti) < 0.5 / scene.sampling_rate) for ti in t])
                    if vru_visible.sum() < self.config.min_vru_frames:
                        continue

                    clip_id = f"{scene.scene_id}_{vru_id}_clip{i:03d}"
                    clip = _extract_subscene(scene, s, e, clip_id)
                    clips.append(clip)
        return clips

    def extract_by_condition(
        self,
        scenes: List[DrivingScene],
        condition: str,
        max_clips: Optional[int] = None,
    ) -> List[DrivingScene]:
        """Extract clips matching a condition: 'night', 'rain', 'day', 'all'."""
        clips = self.extract(scenes)
        if condition == "all":
            filtered = clips
        elif condition == "night":
            filtered = [c for c in clips if c.attributes.lighting == "night"]
        elif condition == "rain":
            filtered = [c for c in clips if "rain" in c.attributes.weather.lower()]
        elif condition == "day":
            filtered = [c for c in clips if c.attributes.lighting == "day"]
        else:
            filtered = clips

        # Sort by adversity/diversity if bonus enabled
        if self.config.adverse_conditions_bonus and condition == "all":
            filtered = sorted(
                filtered,
                key=lambda c: (
                    c.attributes.lighting == "night",
                    "rain" in c.attributes.weather.lower(),
                ),
                reverse=True,
            )
        if max_clips is not None:
            filtered = filtered[:max_clips]
        return filtered
