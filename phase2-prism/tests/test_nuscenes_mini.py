from pathlib import Path
from nuscenes.nuscenes import NuScenes

dataroot = Path.home() / "av-safety-poc/datasets/nuscenes"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(dataroot),
    verbose=True
)

print("Number of scenes:", len(nusc.scene))
print("Number of samples:", len(nusc.sample))
print("Number of annotations:", len(nusc.sample_annotation))

scene = nusc.scene[0]
print("First scene name:", scene["name"])
print("First scene description:", scene["description"])

sample = nusc.get("sample", scene["first_sample_token"])
print("Available sensor channels:", list(sample["data"].keys()))

front_camera_token = sample["data"]["CAM_FRONT"]
front_camera = nusc.get("sample_data", front_camera_token)

print("CAM_FRONT file:", front_camera["filename"])
print("Timestamp:", sample["timestamp"])
