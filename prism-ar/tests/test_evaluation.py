"""Tests for evaluation metrics."""
import numpy as np

from prism_ar.data_ingestion.waymo_loader import WaymoLoader
from prism_ar.prism.risk_engine import PRISMRiskEngine
from prism_ar.ar_overlay.cue_mapper import AdaptiveARCueMapper, StaticARCueMapper
from prism_ar.evaluation.metrics import evaluate_scenario, compute_min_ttc_per_frame


def test_evaluate_scenario():
    loader = WaymoLoader("dummy_path")
    scene = loader.make_synthetic_scene(n_frames=20)
    engine = PRISMRiskEngine()
    output = engine.score(scene)

    adaptive = AdaptiveARCueMapper().map(output)
    static = StaticARCueMapper().map(output)

    ego = scene.get_ego()
    vrus = scene.get_vrus()
    if vrus:
        min_distances = np.linalg.norm(vrus[0].positions - ego.positions, axis=1)
    else:
        min_distances = np.full(scene.frames, np.inf)
    min_ttc = compute_min_ttc_per_frame(scene)

    result = evaluate_scenario(
        scene_id=scene.scene_id,
        adaptive_cues=adaptive,
        static_cues=static,
        prism_output=output,
        timestamps=ego.timestamps,
        min_distances=min_distances,
        min_ttc=min_ttc,
    )
    assert 0.0 <= result.under_warning_rate <= 1.0
    assert 0.0 <= result.over_warning_rate <= 1.0
    assert result.mean_score > 0.0
    assert result.to_dict()["scene_id"] == scene.scene_id
