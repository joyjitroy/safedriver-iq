"""Tests for PRISM risk engine."""
import numpy as np

from prism_ar.data_ingestion.waymo_loader import WaymoLoader
from prism_ar.prism.risk_engine import PRISMRiskEngine, tier_from_score


def test_tier_from_score():
    assert tier_from_score(85.0) == "silent"
    assert tier_from_score(55.0) == "advisory"
    assert tier_from_score(30.0) == "intervention"
    assert tier_from_score(10.0) == "emergency"


def test_risk_engine_outputs():
    loader = WaymoLoader("dummy_path")
    scene = loader.make_synthetic_scene(n_frames=20)
    engine = PRISMRiskEngine()
    output = engine.score(scene)
    assert output.scores.shape[0] == scene.frames
    assert len(output.tiers) == scene.frames
    assert 0.0 <= np.min(output.scores) <= 100.0
    assert output.to_dict()["mean_score"] > 0.0


def test_environmental_risk():
    loader = WaymoLoader("dummy_path")
    scene = loader.make_synthetic_scene()
    scene.attributes.weather = "rain"
    scene.attributes.lighting = "dark"
    scene.attributes.road_condition = "wet"
    engine = PRISMRiskEngine()
    output = engine.score(scene)
    assert output.env_risk > 0.0
