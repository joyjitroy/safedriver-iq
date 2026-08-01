"""Tests for data ingestion and DrivingScene format."""
import numpy as np

from prism_ar.data_ingestion.driving_scene import DrivingScene, Agent, SceneAttributes
from prism_ar.data_ingestion.waymo_loader import WaymoLoader
from prism_ar.data_ingestion.argoverse_loader import ArgoverseLoader
from prism_ar.data_ingestion.nuscenes_loader import NuScenesLoader


def test_driving_scene_summary():
    scene = DrivingScene(
        scene_id="test_001",
        dataset="synthetic",
        agents={
            "ego": Agent(
                agent_id="ego",
                agent_type="ego",
                positions=np.array([[0, 0], [1, 0], [2, 0]]),
                velocities=np.array([[1, 0], [1, 0], [1, 0]]),
                timestamps=np.array([0.0, 0.1, 0.2]),
            ),
            "ped": Agent(
                agent_id="ped",
                agent_type="pedestrian",
                positions=np.array([[5, 0], [5, 1], [5, 2]]),
                velocities=np.array([[0, 1], [0, 1], [0, 1]]),
                timestamps=np.array([0.0, 0.1, 0.2]),
            ),
        },
        attributes=SceneAttributes(),
        frames=3,
        sampling_rate=10.0,
    )
    summary = scene.summary()
    assert summary["scene_id"] == "test_001"
    assert summary["num_vrus"] == 1
    assert summary["has_ego"]


def test_waymo_synthetic_scene():
    loader = WaymoLoader("dummy_path")
    scene = loader.make_synthetic_scene()
    assert scene.dataset == "waymo"
    assert scene.get_ego() is not None
    assert len(scene.get_vrus()) == 1


def test_argoverse_synthetic_scene():
    loader = ArgoverseLoader("dummy_path")
    scene = loader.make_synthetic_scene()
    assert scene.dataset == "argoverse"
    assert scene.get_ego() is not None
    assert len(scene.get_vrus()) == 1


def test_nuscenes_synthetic_scene():
    loader = NuScenesLoader("dummy_path")
    scene = loader.make_synthetic_scene(lighting="night", weather="rain")
    assert scene.dataset == "nuscenes"
    assert scene.attributes.lighting == "night"
    assert scene.attributes.weather == "rain"
