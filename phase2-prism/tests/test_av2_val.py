from pathlib import Path
from collections import Counter

from av2.datasets.motion_forecasting.scenario_serialization import (
    load_argoverse_scenario_parquet,
)

root = Path.home() / "av-safety-poc/datasets/argoverse2/motion-forecasting/val"

scenario_path = next(root.rglob("*.parquet"))
scenario = load_argoverse_scenario_parquet(scenario_path)

print("Scenario file:", scenario_path)
print("Scenario ID:", scenario.scenario_id)
print("City:", scenario.city_name)
print("Number of timestamps:", len(scenario.timestamps_ns))
print("Number of tracks:", len(scenario.tracks))

object_counts = Counter(track.object_type.value for track in scenario.tracks)
print("Object type counts:", object_counts)

vru_tracks = [
    track for track in scenario.tracks
    if track.object_type.value in ["pedestrian", "cyclist"]
]

print("VRU tracks:", len(vru_tracks))

if vru_tracks:
    t = vru_tracks[0]
    print("Example VRU type:", t.object_type.value)
    print("Number of states:", len(t.object_states))
    print("First state:", t.object_states[0])
