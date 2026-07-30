"""Run Phase 2 pipeline over Waymo WOMD validation scenarios."""
import os
os.environ["SDIQ_NUSCENES_DATAROOT"] = "C:\\sdiq\\nuscenes-mini"
os.environ["SDIQ_AV2_VAL_ROOT"] = "C:\\sdiq\\argoverse2-val\\val"
os.environ["PYTHONIOENCODING"] = "utf-8"

import json
import numpy as np
import pandas as pd
from collections import Counter

from sdiq.main import AgenticPipeline
from sdiq.waymo_loader import iter_waymo_scenarios

WAYMO_ROOT = "C:/Personal/EB1A/1. Project Description/American Center for Mobility Project/03_Conference_ASCE2027/safedriver-iq-main/waymo/motion_dataset"

pipeline = AgenticPipeline()

print("Processing Waymo validation scenarios...")
rows = []
for i, sc in enumerate(iter_waymo_scenarios(WAYMO_ROOT, split="validation")):
    if i >= 500:
        break
    r = pipeline.process(sc, narrate=False, explain=False)
    s = r.summary
    speeds = [st.speed for st in sc.ego if st.speed is not None]
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    max_speed = float(np.max(speeds)) if speeds else 0.0
    rows.append({
        "scenario_id": sc.scene_id,
        "source": sc.source,
        "vehicles": sum(1 for a in sc.agents if a.object_type == "vehicle"),
        "pedestrians": sum(1 for a in sc.agents if a.object_type == "pedestrian"),
        "cyclists": sum(1 for a in sc.agents if a.object_type == "cyclist"),
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "min_vru_distance": s.min_distance,
        "near_miss": s.vru_near_misses,
        "safedriver_iq_score": s.env_safety_score,
        "final_safety_score": s.combined_safety_score,
        "crash_probability": 100 - s.combined_safety_score,
        "intervention_level": r.tier_name,
        "env_multiplier": s.env_multiplier,
        "trajectory_risk": s.trajectory_risk,
        "vru_risk": s.vru_risk,
    })
    if (i + 1) % 50 == 0:
        print(f"  processed {i + 1} scenes")

df = pd.DataFrame(rows)
df.to_csv("C:/paper_results/waymo_scenario_summary.csv", index=False)

results = {
    "total_scenes": len(df),
    "mean_crash_probability": float(df["crash_probability"].mean()),
    "mean_final_safety_score": float(df["final_safety_score"].mean()),
    "mean_trajectory_risk": float(df["trajectory_risk"].mean()),
    "mean_vru_risk": float(df["vru_risk"].mean()),
    "near_miss_rate": float(df["near_miss"].mean()),
    "tier_distribution": dict(Counter(df["intervention_level"])),
    "avg_vehicles": float(df["vehicles"].mean()),
    "avg_pedestrians": float(df["pedestrians"].mean()),
    "avg_cyclists": float(df["cyclists"].mean()),
}

with open("C:/paper_results/waymo_validation_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps(results, indent=2, default=str))
print(f"Saved waymo_scenario_summary.csv with {len(df)} rows")
