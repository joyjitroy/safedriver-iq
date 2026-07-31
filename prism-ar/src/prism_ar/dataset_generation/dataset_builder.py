"""Build the PRISM-AR dataset: paired images + CSV annotations.

Follows the PRISM-AR proposal (Section 4):
- Generate ~200 controlled scenarios across risk conditions
- Render static AR and adaptive AR overlays
- Save per-frame annotations with PRISM score, tier, cue type, under_warning, lead_time
"""
from __future__ import annotations

import os
from typing import List, Dict
import pandas as pd
import numpy as np

from prism_ar.data_ingestion.driving_scene import DrivingScene
from prism_ar.prism.risk_engine import PRISMRiskEngine
from prism_ar.ar_overlay.cue_mapper import AdaptiveARCueMapper, StaticARCueMapper
from prism_ar.evaluation.metrics import (
    compute_warning_lead_time,
    compute_under_warning_rate,
    compute_over_warning_rate,
    _tier_level,
)
from .scenario_generator import generate_scenarios
from .ar_renderer import TopDownARRenderer


def _frame_to_row(
    scene: DrivingScene,
    frame_idx: int,
    prism_output,
    static_cue,
    adaptive_cue,
    min_distances: np.ndarray,
    first_warning_idx: int,
) -> Dict:
    """Create a CSV annotation row for one frame."""
    ego = scene.get_ego()
    vrus = scene.get_vrus()
    ego_speed = float(np.linalg.norm(ego.velocities[frame_idx])) if ego else 0.0
    vru_distance = float(min_distances[frame_idx]) if min_distances.size else -1.0
    timestamp = float(ego.timestamps[frame_idx]) if ego else frame_idx / scene.sampling_rate

    under_warning = (
        prism_output.scores[frame_idx] < 40.0
        and _tier_level(adaptive_cue.tier) < _tier_level(static_cue.tier)
    )

    lead_time = 0.0
    if first_warning_idx >= 0:
        lead_time = timestamp - ego.timestamps[first_warning_idx]

    return {
        "scenario_id": scene.scene_id,
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "ego_speed_ms": ego_speed,
        "vru_distance_m": vru_distance,
        "lighting": scene.attributes.lighting,
        "weather": scene.attributes.weather,
        "road_condition": scene.attributes.road_condition,
        "prism_score": float(prism_output.scores[frame_idx]),
        "prism_tier": prism_output.tiers[frame_idx],
        "top_factor": prism_output.top_factor,
        "static_cue": static_cue.tier,
        "adaptive_cue": adaptive_cue.tier,
        "under_warning": int(under_warning),
        "lead_time_s": float(lead_time),
    }


class PRISMARDatasetBuilder:
    """Generate the PRISM-AR dataset."""

    def __init__(
        self,
        output_dir: str,
        n_per_template: int = 5,
        seed: int = 42,
    ):
        self.output_dir = output_dir
        self.n_per_template = n_per_template
        self.seed = seed
        self.risk_engine = PRISMRiskEngine()
        self.adaptive_mapper = AdaptiveARCueMapper()
        self.static_mapper = StaticARCueMapper()
        self.renderer = TopDownARRenderer()

        self.images_dir = os.path.join(output_dir, "images")
        self.annotations_dir = os.path.join(output_dir, "annotations")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)

    def build(self) -> str:
        """Generate dataset and return path to annotations CSV."""
        scenes = generate_scenarios(n_per_template=self.n_per_template, seed=self.seed)
        rows = []

        for scene in scenes:
            prism_output = self.risk_engine.score(scene)
            adaptive_cues = self.adaptive_mapper.map(prism_output)
            static_cues = self.static_mapper.map(prism_output)

            ego = scene.get_ego()
            vrus = scene.get_vrus()
            if vrus and ego:
                min_distances = np.min(
                    [np.linalg.norm(v.positions - ego.positions, axis=1) for v in vrus],
                    axis=0,
                )
            else:
                min_distances = np.full(scene.frames, np.inf)

            # First red-tier warning index
            first_warning_idx = -1
            for i, cue in enumerate(adaptive_cues):
                if cue.tier in ("intervention", "emergency"):
                    first_warning_idx = i
                    break

            # Render static images
            static_dir = os.path.join(self.images_dir, "static", scene.scene_id)
            self.renderer.render_sequence(scene, static_cues, "static", static_dir)

            # Render adaptive images
            adaptive_dir = os.path.join(self.images_dir, "adaptive", scene.scene_id)
            self.renderer.render_sequence(scene, adaptive_cues, "adaptive", adaptive_dir)

            for i in range(scene.frames):
                rows.append(
                    _frame_to_row(
                        scene,
                        i,
                        prism_output,
                        static_cues[i],
                        adaptive_cues[i],
                        min_distances,
                        first_warning_idx,
                    )
                )

        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.annotations_dir, "prism_ar_dataset.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    def close(self):
        self.renderer.close()
