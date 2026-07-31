"""Unified DrivingScene format for PRISM-AR."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import copy
import numpy as np


@dataclass
class Agent:
    """A traffic agent (vehicle, pedestrian, cyclist) in a scene."""
    agent_id: str
    agent_type: str  # 'ego', 'vehicle', 'pedestrian', 'cyclist'
    positions: np.ndarray  # (T, 2) array of [x, y]
    velocities: np.ndarray  # (T, 2)
    timestamps: np.ndarray  # (T,)
    yaw: Optional[np.ndarray] = None  # (T,)
    width: float = 1.0
    length: float = 1.0

    def copy(self) -> "Agent":
        return Agent(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            timestamps=self.timestamps.copy(),
            yaw=self.yaw.copy() if self.yaw is not None else None,
            width=self.width,
            length=self.length,
        )


@dataclass
class SceneAttributes:
    """Environmental and scene-level attributes."""
    weather: str = "clear"
    lighting: str = "day"
    road_condition: str = "dry"
    time_of_day: str = "12:00"
    speed_limit: float = 30.0  # mph or km/h
    location: str = "unknown"


@dataclass
class DrivingScene:
    """Unified representation of a driving scene across datasets."""
    scene_id: str
    dataset: str  # 'crss', 'waymo', 'argoverse', 'nuscenes'
    agents: Dict[str, Agent] = field(default_factory=dict)
    attributes: SceneAttributes = field(default_factory=SceneAttributes)
    frames: int = 0
    sampling_rate: float = 10.0  # Hz

    def copy(self) -> "DrivingScene":
        return DrivingScene(
            scene_id=self.scene_id,
            dataset=self.dataset,
            agents={k: v.copy() for k, v in self.agents.items()},
            attributes=copy.copy(self.attributes),
            frames=self.frames,
            sampling_rate=self.sampling_rate,
        )

    def get_ego(self) -> Optional[Agent]:
        """Return the ego vehicle if present."""
        for agent in self.agents.values():
            if agent.agent_type == "ego":
                return agent
        return None

    def get_vrus(self) -> List[Agent]:
        """Return all vulnerable road users."""
        return [a for a in self.agents.values() if a.agent_type in ("pedestrian", "cyclist")]

    def duration_seconds(self) -> float:
        """Return scene duration in seconds."""
        return self.frames / self.sampling_rate if self.sampling_rate else 0.0

    def summary(self) -> Dict:
        """Return a JSON-serializable summary of the scene."""
        ego = self.get_ego()
        vrus = self.get_vrus()
        return {
            "scene_id": self.scene_id,
            "dataset": self.dataset,
            "frames": self.frames,
            "duration_s": self.duration_seconds(),
            "num_agents": len(self.agents),
            "num_vrus": len(vrus),
            "has_ego": ego is not None,
            "weather": self.attributes.weather,
            "lighting": self.attributes.lighting,
            "road_condition": self.attributes.road_condition,
        }
