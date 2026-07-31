"""Tests for AR overlay mapping."""
import numpy as np

from prism_ar.data_ingestion.waymo_loader import WaymoLoader
from prism_ar.prism.risk_engine import PRISMRiskEngine
from prism_ar.ar_overlay.cue_mapper import AdaptiveARCueMapper, StaticARCueMapper


def test_adaptive_mapper():
    loader = WaymoLoader("dummy_path")
    scene = loader.make_synthetic_scene(n_frames=20)
    engine = PRISMRiskEngine()
    output = engine.score(scene)
    mapper = AdaptiveARCueMapper()
    cues = mapper.map(output)
    assert len(cues) == scene.frames
    assert cues[0].tier in output.tiers


def test_static_mapper():
    loader = WaymoLoader("dummy_path")
    scene = loader.make_synthetic_scene(n_frames=20)
    engine = PRISMRiskEngine()
    output = engine.score(scene)
    mapper = StaticARCueMapper()
    cues = mapper.map(output)
    assert len(cues) == scene.frames
    assert all(c.tier == "advisory" for c in cues)
