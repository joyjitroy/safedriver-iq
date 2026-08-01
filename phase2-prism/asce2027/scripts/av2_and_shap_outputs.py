"""Generate AV2 summary and SHAP top-10 feature plot."""
import os
os.environ["SDIQ_AV2_VAL_ROOT"] = "C:\\sdiq\\argoverse2-val\\val"
os.environ["SDIQ_NUSCENES_DATAROOT"] = "C:\\sdiq\\nuscenes-mini"
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sdiq.data_loader import iter_av2_scenarios, iter_nuscenes_scenes
from sdiq.main import AgenticPipeline
from sdiq.scenario_summary import VECTOR_FIELDS

pipeline = AgenticPipeline()

# --- AV2 evaluation (limited for speed) ---
print("Processing Argoverse 2 scenes ...")
av2_rows = []
for i, sc in enumerate(iter_av2_scenarios()):
    if i >= 500:
        break
    r = pipeline.process(sc, narrate=False, explain=True)
    s = r.summary
    speeds = [st.speed for st in sc.ego if st.speed is not None]
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    max_speed = float(np.max(speeds)) if speeds else 0.0
    av2_rows.append({
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
    })

av2_df = pd.DataFrame(av2_rows)
av2_df.to_csv("C:/paper_results/av2_scenario_summary.csv", index=False)
print(f"Saved av2_scenario_summary.csv with {len(av2_df)} rows")

# --- SHAP top-10 over all nuScenes scenes ---
print("Collecting SHAP attributions over nuScenes ...")
shap_counter = Counter()
for sc in iter_nuscenes_scenes():
    summary = pipeline.summarizer.summarize(sc)
    decision = pipeline.reasoner.decide(summary, explain=True)
    for feat, val in decision.shap.items():
        shap_counter[feat] += abs(float(val))

# Map feature names to readable labels
feature_labels = {
    "env_risk": "Env risk",
    "env_multiplier": "Env multiplier",
    "trajectory_risk": "Trajectory risk",
    "vru_risk": "VRU risk",
    "proximity": "VRU proximity",
    "imminence": "VRU imminence",
    "is_night": "Night",
    "is_rain": "Rain",
}

labels = [feature_labels.get(f, f) for f, _ in shap_counter.most_common(10)]
values = [v for _, v in shap_counter.most_common(10)]

plt.figure(figsize=(8, 5))
plt.barh(labels[::-1], values[::-1], color="teal", edgecolor="black")
plt.xlabel("Mean |SHAP value|")
plt.title("Top 10 SHAP Feature Attributions (nuScenes mini)")
plt.tight_layout()
plt.savefig("C:/paper_results/shap_top10.png")
plt.close()
print("Saved shap_top10.png")

# --- SHAP top-10 over AV2 (sample) ---
print("Collecting SHAP attributions over AV2 sample ...")
shap_counter_av2 = Counter()
for i, sc in enumerate(iter_av2_scenarios()):
    if i >= 100:
        break
    summary = pipeline.summarizer.summarize(sc)
    decision = pipeline.reasoner.decide(summary, explain=True)
    for feat, val in decision.shap.items():
        shap_counter_av2[feat] += abs(float(val))

labels_av2 = [feature_labels.get(f, f) for f, _ in shap_counter_av2.most_common(10)]
values_av2 = [v for _, v in shap_counter_av2.most_common(10)]

plt.figure(figsize=(8, 5))
plt.barh(labels_av2[::-1], values_av2[::-1], color="darkorange", edgecolor="black")
plt.xlabel("Mean |SHAP value|")
plt.title("Top 10 SHAP Feature Attributions (Argoverse 2 sample)")
plt.tight_layout()
plt.savefig("C:/paper_results/shap_top10_av2.png")
plt.close()
print("Saved shap_top10_av2.png")
print("Done")
