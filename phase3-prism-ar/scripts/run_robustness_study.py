"""PRISM-AR robustness study.

Tests the effect of:
- Gaussian noise on ego and VRU positions
- Frame delay (skip first N frames)
- Frame drop (randomly remove frames)

Outputs a robustness CSV and summary table.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from prism_ar.data_ingestion.waymo_loader import WaymoLoader
from prism_ar.data_ingestion.argoverse_loader import ArgoverseLoader
from prism_ar.data_ingestion.nuscenes_loader import NuScenesLoader
from prism_ar.dataset_generation.scenario_extractor import ScenarioExtractor
from prism_ar.dataset_generation.near_miss_generator import generate_near_miss_scenarios
from prism_ar.prism.risk_engine import PRISMRiskEngine, TrainedPRISMRiskEngine
from prism_ar.ar_overlay.cue_mapper import AdaptiveARCueMapper, StaticARCueMapper
from prism_ar.evaluation.metrics import evaluate_scenario, compute_min_ttc_per_frame


DATA_PATHS = {
    "waymo": r"C:\data_prismar\waymo",
    "argoverse": r"C:\data_prismar\argoverse2",
    "nuscenes": r"C:\data_prismar\nuscenes",
}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def add_position_noise(scene, sigma: float):
    """Add Gaussian noise to all agent positions."""
    for agent in scene.agents.values():
        noise = np.random.normal(0, sigma, agent.positions.shape)
        agent.positions = agent.positions + noise
    return scene


def delay_frames(scene, n_frames: int):
    """Skip the first n_frames of the scene."""
    if n_frames >= scene.frames:
        return scene
    for agent in scene.agents.values():
        agent.positions = agent.positions[n_frames:]
        agent.velocities = agent.velocities[n_frames:]
        agent.timestamps = agent.timestamps[n_frames:] - agent.timestamps[n_frames]
        if agent.yaw is not None:
            agent.yaw = agent.yaw[n_frames:]
    scene.frames = scene.frames - n_frames
    return scene


def drop_frames(scene, drop_prob: float):
    """Randomly drop frames with probability drop_prob."""
    keep = np.random.rand(scene.frames) > drop_prob
    if not np.any(keep):
        keep[0] = True
    for agent in scene.agents.values():
        agent.positions = agent.positions[keep]
        agent.velocities = agent.velocities[keep]
        agent.timestamps = agent.timestamps[keep] - agent.timestamps[0]
        if agent.yaw is not None:
            agent.yaw = agent.yaw[keep]
    scene.frames = int(keep.sum())
    return scene


def compute_min_distances(scene):
    ego = scene.get_ego()
    vrus = scene.get_vrus()
    if not vrus or ego is None:
        return np.full(scene.frames, np.inf)
    return np.min(
        [np.linalg.norm(v.positions - ego.positions, axis=1) for v in vrus],
        axis=0,
    )


def evaluate_clips(clips, engine, label):
    adaptive_mapper = AdaptiveARCueMapper()
    static_mapper = StaticARCueMapper()
    rows = []
    for clip in clips:
        output = engine.score(clip)
        min_distances = compute_min_distances(clip)
        min_ttc = compute_min_ttc_per_frame(clip)
        ego = clip.get_ego()
        timestamps = ego.timestamps if ego is not None else np.arange(clip.frames) / clip.sampling_rate
        adaptive_cues = adaptive_mapper.map(output)
        static_cues = static_mapper.map(output)
        result = evaluate_scenario(clip.scene_id, adaptive_cues, static_cues, output, timestamps, min_distances, min_ttc)
        row = result.to_dict()
        row["condition"] = label
        row["dataset"] = clip.dataset
        row["mean_score"] = output.to_dict()["mean_score"]
        rows.append(row)
    return rows


def load_clips(max_scenes: int):
    scenes = []
    scenes += WaymoLoader(DATA_PATHS["waymo"]).load_scenes(split="validation", max_scenes=max_scenes)
    scenes += ArgoverseLoader(DATA_PATHS["argoverse"]).load_scenes(max_scenes=max_scenes)
    scenes += NuScenesLoader(DATA_PATHS["nuscenes"]).load_scenes(max_scenes=max_scenes)
    extractor = ScenarioExtractor()
    clips = extractor.extract(scenes)
    return clips


def main():
    np.random.seed(42)
    clips = load_clips(max_scenes=10)
    print(f"Loaded {len(clips)} real clips for robustness study")
    near_miss = generate_near_miss_scenarios(n_per_type=3)
    clips.extend(near_miss)
    print(f"  + {len(near_miss)} synthetic near-miss clips -> {len(clips)} total")

    try:
        engine = TrainedPRISMRiskEngine()
        print("Using trained PRISM/SafeDriver-IQ model for environmental risk.")
    except Exception as e:
        print(f"Trained model not available ({e}); using reference engine.")
        engine = PRISMRiskEngine()

    conditions = {
        "clean": lambda c: c,
        "noise_0.1": lambda c: add_position_noise(c.copy(), 0.1),
        "noise_0.5": lambda c: add_position_noise(c.copy(), 0.5),
        "delay_5": lambda c: delay_frames(c.copy(), 5),
        "delay_10": lambda c: delay_frames(c.copy(), 10),
        "drop_0.1": lambda c: drop_frames(c.copy(), 0.1),
        "drop_0.3": lambda c: drop_frames(c.copy(), 0.3),
    }

    all_rows = []
    for label, transform in conditions.items():
        transformed = [transform(c) for c in clips]
        rows = evaluate_clips(transformed, engine, label)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "robustness_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved robustness CSV: {csv_path}")

    summary = df.groupby("condition")[[
        "mean_score", "under_warning_rate", "over_warning_rate",
        "warning_lead_time_s", "cue_flicker_hz", "visual_clutter",
    ]].mean()
    print("\nRobustness Summary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
