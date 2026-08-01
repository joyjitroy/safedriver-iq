"""Generate all paper outputs: CSV summary + plots."""
import os
os.environ["SDIQ_AV2_VAL_ROOT"] = "C:\\sdiq\\argoverse2-val\\val"
os.environ["SDIQ_NUSCENES_DATAROOT"] = "C:\\sdiq\\nuscenes-mini"
os.environ["PYTHONIOENCODING"] = "utf-8"

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sdiq.data_loader import iter_nuscenes_scenes
from sdiq.main import AgenticPipeline
from sdiq.agentic_layer import TIER_NAMES

pipeline = AgenticPipeline()
scenes = list(iter_nuscenes_scenes())


def scene_speed_stats(sc):
    speeds = [s.speed for s in sc.ego if s.speed is not None]
    if not speeds:
        return 0.0, 0.0
    return float(np.mean(speeds)), float(np.max(speeds))


rows = []
for sc in scenes:
    r = pipeline.process(sc, narrate=False, explain=False)
    s = r.summary
    mean_speed, max_speed = scene_speed_stats(sc)
    rows.append({
        "scenario_id": sc.scene_id,
        "city": sc.city or "unknown",
        "vehicles": sum(1 for a in sc.agents if a.object_type == "vehicle"),
        "pedestrians": sum(1 for a in sc.agents if a.object_type == "pedestrian"),
        "cyclists": sum(1 for a in sc.agents if a.object_type == "cyclist"),
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "min_vru_distance": s.min_distance,
        "near_miss": s.vru_near_misses,
        "safedriver_iq_score": s.env_safety_score,
        "final_safety_score": s.combined_safety_score,
        "intervention_level": r.tier_name,
        "env_multiplier": s.env_multiplier,
        "trajectory_risk": s.trajectory_risk,
        "vru_risk": s.vru_risk,
        "is_night": s.is_night,
        "is_rain": s.is_rain,
    })

df = pd.DataFrame(rows)
df.to_csv("C:/paper_results/scenario_summary.csv", index=False)
print(f"Saved scenario_summary.csv with {len(df)} rows")

# Plot 1: Score distribution
plt.figure(figsize=(8, 5))
plt.hist(df["final_safety_score"], bins=20, color="steelblue", edgecolor="black")
plt.xlabel("Final Safety Score")
plt.ylabel("Number of Scenes")
plt.title("Distribution of Final Safety Scores (nuScenes mini)")
plt.tight_layout()
plt.savefig("C:/paper_results/score_distribution.png")
plt.close()

# Plot 2: Near-miss rate by city
plt.figure(figsize=(8, 5))
city_nm = df.groupby("city")["near_miss"].mean().sort_values(ascending=False)
city_nm.plot(kind="bar", color="coral", edgecolor="black")
plt.ylabel("Near-Miss Rate")
plt.title("Near-Miss Rate by City")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("C:/paper_results/nearmiss_by_city.png")
plt.close()

# Plot 3: VRU distance histogram
plt.figure(figsize=(8, 5))
plt.hist(df["min_vru_distance"].replace(float("inf"), 50), bins=20, color="seagreen", edgecolor="black")
plt.xlabel("Minimum VRU Distance (m)")
plt.ylabel("Number of Scenes")
plt.title("VRU Distance Histogram")
plt.tight_layout()
plt.savefig("C:/paper_results/vru_distance_histogram.png")
plt.close()

print("Plots saved: score_distribution.png, nearmiss_by_city.png, vru_distance_histogram.png")
