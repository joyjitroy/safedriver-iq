"""PRISM-AR ablation study.

Tests the effect of disabling each PRISM risk component:
- full: environment + trajectory + VRU
- no_vru: environment + trajectory only
- no_weather: environment (without weather) + trajectory + VRU
- no_trajectory: environment + VRU only

Outputs an ablation CSV and summary table.
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


def compute_min_distances(scene):
    ego = scene.get_ego()
    vrus = scene.get_vrus()
    if not vrus or ego is None:
        return np.full(scene.frames, np.inf)
    return np.min(
        [np.linalg.norm(v.positions - ego.positions, axis=1) for v in vrus],
        axis=0,
    )


def load_clips(max_scenes: int):
    scenes = []
    scenes += WaymoLoader(DATA_PATHS["waymo"]).load_scenes(split="validation", max_scenes=max_scenes)
    scenes += ArgoverseLoader(DATA_PATHS["argoverse"]).load_scenes(max_scenes=max_scenes)
    scenes += NuScenesLoader(DATA_PATHS["nuscenes"]).load_scenes(max_scenes=max_scenes)
    extractor = ScenarioExtractor()
    clips = extractor.extract(scenes)
    return clips


def run_ablation(engine, label, clips):
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
        row["ablation"] = label
        row["dataset"] = clip.dataset
        row["mean_score"] = output.to_dict()["mean_score"]
        row["env_risk"] = output.env_risk
        row["mean_traj_risk"] = float(np.mean(output.traj_risk))
        row["mean_vru_risk"] = float(np.mean(output.vru_risk))
        row["top_factor"] = output.top_factor
        rows.append(row)
    return rows


def main():
    clips = load_clips(max_scenes=20)
    print(f"Loaded {len(clips)} real clips for ablation study")
    near_miss = generate_near_miss_scenarios(n_per_type=5)
    clips.extend(near_miss)
    print(f"  + {len(near_miss)} synthetic near-miss clips -> {len(clips)} total")

    try:
        base_engine = TrainedPRISMRiskEngine()
        print("Using trained PRISM/SafeDriver-IQ model for environmental risk.")
    except Exception as e:
        print(f"Trained model not available ({e}); using reference engine.")
        base_engine = PRISMRiskEngine()

    configs = {
        "full": (0.40, 0.30, 0.30),
        "no_vru": (0.55, 0.45, 0.0),
        "no_trajectory": (0.55, 0.0, 0.45),
        "no_weather": (0.40, 0.30, 0.30),
    }

    all_rows = []
    for label, (env_w, traj_w, vru_w) in configs.items():
        base_engine.set_weights(env_w, traj_w, vru_w)
        if label == "no_weather":
            # Override weather/lights to always be clear/day
            for clip in clips:
                clip.attributes.weather = "clear"
                clip.attributes.lighting = "day"
                clip.attributes.road_condition = "dry"
        rows = run_ablation(base_engine, label, clips)
        all_rows.extend(rows)
        if label == "no_weather":
            # Reload clips to restore original attributes
            clips = load_clips(max_scenes=20)

    df = pd.DataFrame(all_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved ablation CSV: {csv_path}")

    # Summary table
    summary = df.groupby("ablation")[[
        "mean_score", "under_warning_rate", "over_warning_rate",
        "warning_lead_time_s", "cue_flicker_hz", "visual_clutter",
    ]].mean()
    print("\nAblation Summary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
