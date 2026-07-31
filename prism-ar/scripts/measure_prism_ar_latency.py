"""Measure per-layer latency of the PRISM-AR pipeline.

Outputs a JSON file with timing breakdown per layer in milliseconds per frame.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from prism_ar.data_ingestion.waymo_loader import WaymoLoader
from prism_ar.data_ingestion.argoverse_loader import ArgoverseLoader
from prism_ar.data_ingestion.nuscenes_loader import NuScenesLoader
from prism_ar.dataset_generation.scenario_extractor import ScenarioExtractor, ExtractionConfig
from prism_ar.dataset_generation.near_miss_generator import generate_near_miss_scenarios
from prism_ar.prism.risk_engine import (
    PRISMRiskEngine,
    _environmental_risk,
    _trajectory_risk,
    _vru_risk,
)
from prism_ar.ar_overlay.cue_mapper import AdaptiveARCueMapper


DATA_PATHS = {
    "waymo": r"C:\data_prismar\waymo",
    "argoverse": r"C:\data_prismar\argoverse2",
    "nuscenes": r"C:\data_prismar\nuscenes",
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_clips(max_scenes: int = 5):
    waymo_loader = WaymoLoader(DATA_PATHS["waymo"])
    argo_loader = ArgoverseLoader(DATA_PATHS["argoverse"])
    nuscenes_loader = NuScenesLoader(DATA_PATHS["nuscenes"])

    config = ExtractionConfig(
        max_distance_m=25.0,
        near_miss_mode=True,
        near_miss_distance_m=15.0,
        near_miss_ttc_s=5.0,
    )
    extractor = ScenarioExtractor(config)

    clips = []
    for loader, split in [
        (waymo_loader, "validation"),
        (argo_loader, "val"),
        (nuscenes_loader, "v1.0-mini"),
    ]:
        try:
            scenes = loader.load_scenes(split=split, max_scenes=max_scenes)
            for ds in scenes:
                try:
                    clips.extend(extractor.extract(ds))
                except Exception as e:
                    print(f"Skipping scene {ds.scene_id}: {e}")
        except Exception as e:
            print(f"Skipping loader: {e}")
    clips.extend(generate_near_miss_scenarios(n_per_type=1))
    return clips


def measure_latency(clips, engine, n_repeats: int = 3):
    mapper = AdaptiveARCueMapper()
    timings = []
    for clip in clips:
        env_times = []
        traj_times = []
        vru_times = []
        fuse_times = []
        cue_times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _environmental_risk(clip)
            t1 = time.perf_counter()
            _trajectory_risk(clip)
            t2 = time.perf_counter()
            _vru_risk(clip)
            t3 = time.perf_counter()
            prism_output = engine.score(clip)
            t4 = time.perf_counter()
            mapper.map(prism_output)
            t5 = time.perf_counter()

            env_times.append((t1 - t0) * 1000.0)
            traj_times.append((t2 - t1) * 1000.0)
            vru_times.append((t3 - t2) * 1000.0)
            fuse_times.append((t4 - t3) * 1000.0)
            cue_times.append((t5 - t4) * 1000.0)

        n_frames = clip.frames
        timings.append(
            {
                "scene_id": clip.scene_id,
                "frames": n_frames,
                "env_ms_per_frame": np.mean(env_times) / n_frames,
                "traj_ms_per_frame": np.mean(traj_times) / n_frames,
                "vru_ms_per_frame": np.mean(vru_times) / n_frames,
                "fuse_ms_per_frame": np.mean(fuse_times) / n_frames,
                "cue_ms_per_frame": np.mean(cue_times) / n_frames,
                "total_ms_per_frame": (
                    np.mean(env_times)
                    + np.mean(traj_times)
                    + np.mean(vru_times)
                    + np.mean(fuse_times)
                    + np.mean(cue_times)
                ) / n_frames,
            }
        )
    return timings


def main():
    engine = PRISMRiskEngine()
    clips = load_clips(max_scenes=10)
    print(f"Measuring latency on {len(clips)} clips...")
    timings = measure_latency(clips, engine, n_repeats=3)

    summary = {
        "n_clips": len(timings),
        "mean_ms_per_frame": {
            "environmental": float(np.mean([t["env_ms_per_frame"] for t in timings])),
            "trajectory": float(np.mean([t["traj_ms_per_frame"] for t in timings])),
            "vru": float(np.mean([t["vru_ms_per_frame"] for t in timings])),
            "fusion": float(np.mean([t["fuse_ms_per_frame"] for t in timings])),
            "cue_mapping": float(np.mean([t["cue_ms_per_frame"] for t in timings])),
            "total": float(np.mean([t["total_ms_per_frame"] for t in timings])),
        },
        "per_clip": timings,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "latency_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved latency results: {out_path}")
    print(json.dumps(summary["mean_ms_per_frame"], indent=2))


if __name__ == "__main__":
    main()
