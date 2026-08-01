"""Simplified PRISM risk engine for DrivingScene objects.

This is a reference implementation that fuses environmental, trajectory,
and VRU-interaction risk into a continuous 0--100 safety score and a
four-tier risk level. In a production system, the weights below would be
replaced by the trained PRISM models (Random Forest, LSTM, SFM + LSTM, DQN).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from prism_ar.data_ingestion.driving_scene import DrivingScene, Agent

# Optional: trained PRISM/SafeDriver-IQ model bridge
try:
    from prism_ar.prism.trained_model_bridge import TrainedPRISMBridge
    _TRAINED_BRIDGE_AVAILABLE = True
except Exception:
    _TRAINED_BRIDGE_AVAILABLE = False
    TrainedPRISMBridge = None

from prism_ar.prism.crss_env_mapper import crss_env_risk_multiplier


RISK_TIERS = {
    "emergency": (0.0, 20.0),
    "intervention": (20.0, 40.0),
    "advisory": (40.0, 70.0),
    "silent": (70.0, 100.0),
}


def tier_from_score(score: float) -> str:
    for tier, (low, high) in RISK_TIERS.items():
        if low <= score < high:
            return tier
    return "silent" if score >= 100.0 else "emergency"


def _environmental_risk(scene: DrivingScene) -> float:
    """Return a 0--1 environmental crash likelihood based on scene attributes."""
    risk = 0.0
    weather = scene.attributes.weather.lower()
    lighting = scene.attributes.lighting.lower()
    road = scene.attributes.road_condition.lower()
    if any(w in weather for w in ["rain", "snow", "sleet", "fog"]):
        risk += 0.25
    if any(l in lighting for l in ["dark", "dusk", "dawn"]):
        risk += 0.20
    if any(r in road for r in ["wet", "ice", "snow", "slush"]):
        risk += 0.20
    return min(1.0, risk)


def _compute_ttc(ego: Agent, vru: Agent) -> np.ndarray:
    """Compute a simplified time-to-collision array (frames)."""
    rel_pos = vru.positions - ego.positions
    rel_vel = vru.velocities - ego.velocities
    rel_speed = np.linalg.norm(rel_vel, axis=1)
    rel_distance = np.linalg.norm(rel_pos, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(rel_speed > 0.01, rel_distance / rel_speed, np.inf)
    return ttc


def _trajectory_risk(scene: DrivingScene) -> np.ndarray:
    """Return a 0--1 trajectory risk array based on TTC and relative speed."""
    ego = scene.get_ego()
    if ego is None:
        return np.zeros(scene.frames)
    ttc = np.full(scene.frames, np.inf)
    for vru in scene.get_vrus():
        ttc = np.minimum(ttc, _compute_ttc(ego, vru))
    # Saturate risk for imminent collision: TTC < 1.0s -> ~1.0, TTC < 2.0s -> high
    risk = np.where(
        ttc < 1.0,
        0.99,
        np.where(ttc < 2.0, 0.95 * np.exp(-(ttc - 1.0) / 0.7), np.exp(-ttc / 2.0)),
    )
    risk = np.clip(risk, 0.0, 1.0)
    return risk


def _vru_risk(scene: DrivingScene) -> np.ndarray:
    """Return a 0--1 VRU proximity risk array based on distance to ego."""
    ego = scene.get_ego()
    if ego is None:
        return np.zeros(scene.frames)
    min_dist = np.full(scene.frames, np.inf)
    for vru in scene.get_vrus():
        dist = np.linalg.norm(vru.positions - ego.positions, axis=1)
        min_dist = np.minimum(min_dist, dist)
    # Saturate risk for actual overlap / very close proximity: < 2m -> ~1.0
    risk = np.where(
        min_dist < 2.0,
        0.99,
        np.where(min_dist < 4.0, 0.95 * np.exp(-(min_dist - 2.0) / 1.5), np.exp(-min_dist / 4.0)),
    )
    risk = np.clip(risk, 0.0, 1.0)
    return risk


@dataclass
class PRISMOutput:
    """PRISM risk output per frame."""
    scene_id: str
    scores: np.ndarray  # continuous 0--100 safety score
    tiers: list  # per-frame tier strings
    env_risk: float  # scalar 0--1
    traj_risk: np.ndarray  # per-frame 0--1
    vru_risk: np.ndarray  # per-frame 0--1
    top_factor: str

    def to_dict(self) -> Dict:
        return {
            "scene_id": self.scene_id,
            "mean_score": float(np.mean(self.scores)),
            "min_score": float(np.min(self.scores)),
            "tier_distribution": {tier: self.tiers.count(tier) for tier in set(self.tiers)},
            "env_risk": float(self.env_risk),
            "mean_traj_risk": float(np.mean(self.traj_risk)),
            "mean_vru_risk": float(np.mean(self.vru_risk)),
            "top_factor": self.top_factor,
        }


class PRISMRiskEngine:
    """Reference PRISM risk engine."""

    def __init__(self, env_weight: float = 0.15, traj_weight: float = 0.40, vru_weight: float = 0.45):
        self.env_weight = env_weight
        self.traj_weight = traj_weight
        self.vru_weight = vru_weight

    def score(self, scene: DrivingScene) -> PRISMOutput:
        """Compute PRISM score and tier for a DrivingScene."""
        env_risk = _environmental_risk(scene)
        traj_risk = _trajectory_risk(scene)
        vru_risk = _vru_risk(scene)

        # Combine risks per frame
        fused_risk = (
            self.env_weight * env_risk
            + self.traj_weight * traj_risk
            + self.vru_weight * vru_risk
        )
        fused_risk = np.clip(fused_risk, 0.0, 1.0)
        # Convert to safety score (inverse crash probability)
        scores = (1.0 - fused_risk) * 100.0
        tiers = [tier_from_score(float(s)) for s in scores]

        # Top factor by scene attributes
        top_factor = "environmental"
        if np.mean(traj_risk) > env_risk and np.mean(traj_risk) > np.mean(vru_risk):
            top_factor = "trajectory"
        if np.mean(vru_risk) > env_risk and np.mean(vru_risk) > np.mean(traj_risk):
            top_factor = "vru_proximity"

        return PRISMOutput(
            scene_id=scene.scene_id,
            scores=scores,
            tiers=tiers,
            env_risk=env_risk,
            traj_risk=traj_risk,
            vru_risk=vru_risk,
            top_factor=top_factor,
        )


class TrainedPRISMRiskEngine:
    """PRISM engine using the trained CRSS model for environmental risk.

    Optionally blends a lightweight CRSS-derived environmental risk multiplier
    with the trained model's environmental score to account for raw CRSS
    environmental attributes (weather, lighting, road, time-of-day).
    """

    def __init__(
        self,
        env_weight: float = 0.40,
        traj_weight: float = 0.30,
        vru_weight: float = 0.30,
        use_crss_env: bool = True,
        crss_env_weight: float = 0.05,
    ):
        if not _TRAINED_BRIDGE_AVAILABLE:
            raise ImportError("TrainedPRISMBridge is not available. Install safedriver-iq-model dependencies.")
        self.bridge = TrainedPRISMBridge()
        self.env_weight = env_weight
        self.traj_weight = traj_weight
        self.vru_weight = vru_weight
        self.use_crss_env = use_crss_env
        self.crss_env_weight = crss_env_weight

    def set_weights(self, env_weight: float, traj_weight: float, vru_weight: float):
        """Update fusion weights for ablation studies."""
        self.env_weight = env_weight
        self.traj_weight = traj_weight
        self.vru_weight = vru_weight

    def score(self, scene: DrivingScene) -> PRISMOutput:
        """Compute PRISM score using the trained environmental model."""
        traj_risk = _trajectory_risk(scene)
        vru_risk = _vru_risk(scene)

        model_env_risk = self.bridge.environmental_risk(
            weather=scene.attributes.weather,
            lighting=scene.attributes.lighting,
            road_condition=scene.attributes.road_condition,
            time_of_day=scene.attributes.time_of_day,
            num_vrus=len(scene.get_vrus()),
        )

        if self.use_crss_env:
            crss_env_risk = crss_env_risk_multiplier(scene.attributes)
            env_risk = (
                (1.0 - self.crss_env_weight) * model_env_risk
                + self.crss_env_weight * crss_env_risk
            )
        else:
            env_risk = model_env_risk

        fused_risk = (
            self.env_weight * env_risk
            + self.traj_weight * traj_risk
            + self.vru_weight * vru_risk
        )
        fused_risk = np.clip(fused_risk, 0.0, 1.0)
        scores = (1.0 - fused_risk) * 100.0
        tiers = [tier_from_score(float(s)) for s in scores]

        top_factor = "environmental"
        if np.mean(traj_risk) > env_risk and np.mean(traj_risk) > np.mean(vru_risk):
            top_factor = "trajectory"
        if np.mean(vru_risk) > env_risk and np.mean(vru_risk) > np.mean(traj_risk):
            top_factor = "vru_proximity"

        return PRISMOutput(
            scene_id=scene.scene_id,
            scores=scores,
            tiers=tiers,
            env_risk=env_risk,
            traj_risk=traj_risk,
            vru_risk=vru_risk,
            top_factor=top_factor,
        )
