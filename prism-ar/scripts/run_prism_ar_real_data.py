"""End-to-end PRISM-AR evaluation on real datasets.

Usage:
    C:\prismar_venv\Scripts\python.exe run_prism_ar_real_data.py --max_scenes 50 --output_dir results/prism_ar_real

Steps:
1. Load Waymo, Argoverse 2, and nuScenes mini through short-path junctions.
2. Extract AR-relevant VRU interaction clips.
3. Run PRISM reference engine on each clip.
4. Generate paired overlays (no-AR, static, adaptive, oracle).
5. Compute evaluation metrics.
6. Save results CSV, summary tables, and sample overlay images.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from prism_ar.data_ingestion.waymo_loader import WaymoLoader
from prism_ar.data_ingestion.argoverse_loader import ArgoverseLoader
from prism_ar.data_ingestion.nuscenes_loader import NuScenesLoader
from prism_ar.dataset_generation.scenario_extractor import ScenarioExtractor
from prism_ar.dataset_generation.near_miss_generator import generate_near_miss_scenarios
from prism_ar.prism.risk_engine import PRISMRiskEngine, TrainedPRISMRiskEngine
from prism_ar.ar_overlay.cue_mapper import (
    AdaptiveARCueMapper,
    StaticARCueMapper,
    NoARCueMapper,
    OracleARCueMapper,
)
from prism_ar.dataset_generation.ar_renderer import TopDownARRenderer
from prism_ar.evaluation.metrics import evaluate_scenario, compute_min_ttc_per_frame


# Short-path junctions created by setup
DATA_PATHS = {
    "waymo": r"C:\data_prismar\waymo",
    "argoverse": r"C:\data_prismar\argoverse2",
    "nuscenes": r"C:\data_prismar\nuscenes",
}


def load_real_scenes(max_scenes: int) -> Dict[str, List]:
    """Load scenes from all real datasets."""
    scenes_by_dataset = {}
    print("Loading Waymo WOMD...")
    scenes_by_dataset["waymo"] = WaymoLoader(DATA_PATHS["waymo"]).load_scenes(
        split="validation", max_scenes=max_scenes
    )
    print(f"  -> {len(scenes_by_dataset['waymo'])} scenes")

    print("Loading Argoverse 2...")
    scenes_by_dataset["argoverse"] = ArgoverseLoader(DATA_PATHS["argoverse"]).load_scenes(
        max_scenes=max_scenes
    )
    print(f"  -> {len(scenes_by_dataset['argoverse'])} scenes")

    print("Loading nuScenes mini...")
    scenes_by_dataset["nuscenes"] = NuScenesLoader(DATA_PATHS["nuscenes"]).load_scenes(
        max_scenes=max_scenes
    )
    print(f"  -> {len(scenes_by_dataset['nuscenes'])} scenes")

    return scenes_by_dataset


def compute_min_distances(scene):
    ego = scene.get_ego()
    vrus = scene.get_vrus()
    if not vrus or ego is None:
        return np.full(scene.frames, np.inf)
    return np.min(
        [np.linalg.norm(v.positions - ego.positions, axis=1) for v in vrus],
        axis=0,
    )


def run_pipeline(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 1. Load datasets
    scenes_by_dataset = load_real_scenes(args.max_scenes)
    all_scenes = [s for scenes in scenes_by_dataset.values() for s in scenes]

    # 2. Extract AR-relevant clips
    print("Extracting AR-relevant clips...")
    extractor = ScenarioExtractor()
    clips = extractor.extract(all_scenes)
    print(f"  -> {len(clips)} real clips")

    # Add controlled synthetic near-miss clips to ensure all risk tiers are represented
    if args.near_miss_clips > 0:
        print(f"Generating {args.near_miss_clips} synthetic near-miss clips...")
        near_miss_scenes = generate_near_miss_scenarios(
            n_per_type=max(1, args.near_miss_clips // 6), seed=42
        )
        # Limit to requested number
        near_miss_scenes = near_miss_scenes[:args.near_miss_clips]
        clips.extend(near_miss_scenes)
        print(f"  -> {len(clips)} total clips ({len(clips) - len(near_miss_scenes)} real + {len(near_miss_scenes)} synthetic)")

    if len(clips) == 0:
        print("No clips extracted. Exiting.")
        return

    # Optional: limit total clips
    if args.max_clips is not None and len(clips) > args.max_clips:
        clips = clips[:args.max_clips]

    # 3. Initialize PRISM and cue mappers
    try:
        engine = TrainedPRISMRiskEngine()
        print("Using trained PRISM/SafeDriver-IQ model for environmental risk.")
    except Exception as e:
        print(f"Trained model not available ({e}); using reference PRISM engine.")
        engine = PRISMRiskEngine()
    adaptive_mapper = AdaptiveARCueMapper()
    static_mapper = StaticARCueMapper()
    no_ar_mapper = NoARCueMapper()
    oracle_mapper = OracleARCueMapper()
    renderer = TopDownARRenderer()

    # 4. Run pipeline per clip
    print("Running PRISM + AR evaluation...")
    rows = []
    start_time = time.time()
    for i, clip in enumerate(clips):
        prism_output = engine.score(clip)
        min_distances = compute_min_distances(clip)
        min_ttc = compute_min_ttc_per_frame(clip)
        ego = clip.get_ego()
        timestamps = ego.timestamps if ego is not None else np.arange(clip.frames) / clip.sampling_rate

        adaptive_cues = adaptive_mapper.map(prism_output)
        static_cues = static_mapper.map(prism_output)
        no_ar_cues = no_ar_mapper.map(prism_output)
        oracle_cues = oracle_mapper.map(prism_output)

        # Render a few example frames
        if i < args.num_overlay_examples:
            for mode, cues in [
                ("no_ar", no_ar_cues),
                ("static", static_cues),
                ("adaptive", adaptive_cues),
                ("oracle", oracle_cues),
            ]:
                frame_dir = os.path.join(images_dir, clip.scene_id, mode)
                os.makedirs(frame_dir, exist_ok=True)
                # Render middle frame
                mid = len(cues) // 2
                renderer.save_frame(clip, mid, cues[mid], os.path.join(frame_dir, f"frame_{mid:04d}.png"))

        adaptive_eval = evaluate_scenario(
            scene_id=clip.scene_id,
            adaptive_cues=adaptive_cues,
            static_cues=static_cues,
            prism_output=prism_output,
            timestamps=timestamps,
            min_distances=min_distances,
            min_ttc=min_ttc,
        )

        row = adaptive_eval.to_dict()
        row["dataset"] = clip.dataset
        row["frames"] = clip.frames
        row["duration_s"] = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        row["num_vrus"] = len(clip.get_vrus())
        row["lighting"] = clip.attributes.lighting
        row["weather"] = clip.attributes.weather
        row["location"] = clip.attributes.location
        row["env_risk"] = prism_output.env_risk
        row["trajectory_risk"] = float(np.mean(prism_output.traj_risk))
        row["vru_risk"] = float(np.mean(prism_output.vru_risk))
        row["min_distance_m"] = float(np.min(min_distances))
        row["mean_ttc"] = float(np.mean(compute_min_distances(clip) / np.maximum(np.linalg.norm(clip.get_ego().velocities, axis=1), 0.01))) if clip.get_ego() is not None else 0.0
        rows.append(row)

    end_time = time.time()
    total_runtime = end_time - start_time

    # 5. Save results
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "prism_ar_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results CSV: {csv_path}")

    # 6. Summary tables
    def _to_native(obj):
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        return obj

    from collections import Counter
    tier_counter = Counter()
    for r in df["tier_distribution"]:
        for k, v in r.items():
            tier_counter[k] += int(v)
    tier_counts = dict(tier_counter)
    summary = {
        "total_clips": len(clips),
        "total_runtime_s": total_runtime,
        "clips_per_second": len(clips) / total_runtime if total_runtime > 0 else 0,
        "mean_latency_ms": (total_runtime / len(clips)) * 1000 if clips else 0,
        "dataset_counts": _to_native(df["dataset"].value_counts().to_dict()),
        "tier_counts": _to_native(tier_counts),
        "mean_by_dataset": _to_native(df.groupby("dataset")[[
            "mean_score", "under_warning_rate", "over_warning_rate",
            "warning_lead_time_s", "cue_flicker_hz", "visual_clutter"
        ]].mean().to_dict()),
    }
    json_path = os.path.join(output_dir, "prism_ar_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {json_path}")

    # 7. Print summary
    print("\n=== PRISM-AR Real-Data Summary ===")
    print(f"Total clips: {summary['total_clips']}")
    print(f"Total runtime: {total_runtime:.1f}s")
    print(f"Mean latency: {summary['mean_latency_ms']:.1f}ms per clip")
    print("\nDataset breakdown:")
    print(df.groupby("dataset")[["mean_score", "under_warning_rate", "over_warning_rate"]].mean().to_string())

    renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Run PRISM-AR on real datasets")
    parser.add_argument("--max_scenes", type=int, default=50, help="Max scenes per dataset")
    parser.add_argument("--max_clips", type=int, default=None, help="Max total clips to process")
    parser.add_argument("--num_overlay_examples", type=int, default=5, help="Number of clips to render overlays for")
    parser.add_argument("--near_miss_clips", type=int, default=60, help="Number of synthetic near-miss clips to add")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    args.output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    run_pipeline(args)


if __name__ == "__main__":
    main()
