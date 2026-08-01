"""Waymo validation runner — detects missing data and falls back to AV2 scaling."""
import os
os.environ["SDIQ_AV2_VAL_ROOT"] = "C:\\sdiq\\argoverse2-val\\val"
os.environ["SDIQ_NUSCENES_DATAROOT"] = "C:\\sdiq\\nuscenes-mini"
os.environ["PYTHONIOENCODING"] = "utf-8"

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sdiq.main import AgenticPipeline
from sdiq.data_loader import iter_av2_scenarios

pipeline = AgenticPipeline()

waymo_dir = Path("C:/Personal/EB1A/1. Project Description/American Center for Mobility Project/03_Conference_ASCE2027/safedriver-iq-main/waymo/motion_dataset")
waymo_val_files = list((waymo_dir / "tf_example_datasets" / "validation").glob("*.tfrecord*")) if waymo_dir.exists() else []
waymo_has_data = any(f.stat().st_size > 1000 for f in waymo_val_files)

results = {"waymo_available": waymo_has_data, "waymo_files_found": len(waymo_val_files)}

if not waymo_has_data:
    results["note"] = "Waymo WOMD validation files are placeholders (< 1 KB). Running scaled AV2 validation instead."

# Run 1000 AV2 scenes as Waymo substitute
print("Running scaled AV2 validation (1000 scenes) since Waymo data is not present ...")
rows = []
for i, sc in enumerate(iter_av2_scenarios()):
    if i >= 1000:
        break
    r = pipeline.process(sc, narrate=False, explain=False)
    s = r.summary
    speeds = [st.speed for st in sc.ego if st.speed is not None]
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    max_speed = float(np.max(speeds)) if speeds else 0.0
    rows.append({
        "scenario_id": sc.scene_id,
        "city": sc.city or "unknown",
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

df = pd.DataFrame(rows)
df.to_csv("C:/paper_results/av2_validation_1000.csv", index=False)

# Mean crash probability by city
by_city = df.groupby("city").agg(
    n=("scenario_id", "count"),
    mean_crash_probability=("crash_probability", "mean"),
    mean_final_safety_score=("final_safety_score", "mean"),
    near_miss_rate=("near_miss", "mean"),
).reset_index().sort_values("mean_crash_probability", ascending=False)
by_city.to_csv("C:/paper_results/av2_validation_by_city.csv", index=False)

# Overall tier distribution
results["total_scenes"] = len(df)
results["tier_distribution"] = dict(Counter(df["intervention_level"]))
results["mean_crash_probability"] = float(df["crash_probability"].mean())
results["mean_final_safety_score"] = float(df["final_safety_score"].mean())
results["near_miss_rate"] = float(df["near_miss"].mean())
results["by_city"] = by_city.to_dict("records")

with open("C:/paper_results/waymo_validation_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps(results, indent=2, default=str))
print("Saved: av2_validation_1000.csv, av2_validation_by_city.csv, waymo_validation_results.json")
