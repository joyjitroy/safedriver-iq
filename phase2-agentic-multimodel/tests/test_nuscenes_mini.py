"""M0 sanity test — nuScenes mini loads and the expected scene/sensor structure is present.

Config-driven (no hard-coded ~/av-safety-poc paths) and devkit-free (uses the POC's
NuScenesMini reader, avoiding the nuscenes-devkit numpy<2 conflict). Runs as a plain
script or under pytest.
"""
from __future__ import annotations

from sdiq import config
from sdiq.nuscenes_mini import NuScenesMini


def test_nuscenes_mini_loads():
    assert not config.validate(require_phase1_model=False), config.validate(False)

    nusc = NuScenesMini(config.NUSCENES_DATAROOT, config.NUSCENES_VERSION, verbose=True)

    # The mini split is a fixed, known size — assert exact expected counts.
    assert len(nusc.scene) == 10, f"expected 10 scenes, got {len(nusc.scene)}"
    assert len(nusc.sample) == 404, f"expected 404 samples, got {len(nusc.sample)}"
    assert len(nusc.sample_annotation) == 18538, \
        f"expected 18538 annotations, got {len(nusc.sample_annotation)}"

    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])

    # 12 sensors: 6 cameras + LIDAR_TOP + 5 radars.
    channels = set(sample["data"].keys())
    assert "CAM_FRONT" in channels
    assert "LIDAR_TOP" in channels
    assert len(channels) == 12, f"expected 12 sensor channels, got {sorted(channels)}"

    front = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    assert front["filename"].endswith(".jpg")
    assert config.NUSCENES_DATAROOT.joinpath(front["filename"]).is_file(), \
        "CAM_FRONT image file referenced by the table is missing on disk"


if __name__ == "__main__":
    test_nuscenes_mini_loads()
    nusc = NuScenesMini(config.NUSCENES_DATAROOT, config.NUSCENES_VERSION)
    s = nusc.scene[0]
    print("Number of scenes:", len(nusc.scene))
    print("Number of samples:", len(nusc.sample))
    print("Number of annotations:", len(nusc.sample_annotation))
    print("First scene name:", s["name"])
    print("First scene description:", s["description"])
    samp = nusc.get("sample", s["first_sample_token"])
    print("Available sensor channels:", list(samp["data"].keys()))
    print("CAM_FRONT file:", nusc.get("sample_data", samp["data"]["CAM_FRONT"])["filename"])
    print("Timestamp:", samp["timestamp"])
    print("\nPASS: nuScenes mini sanity test")
