"""Tests for dataset generation."""
import os
import tempfile

from prism_ar.dataset_generation.scenario_generator import generate_scenarios, ScenarioParams, generate_scenario
from prism_ar.dataset_generation.dataset_builder import PRISMARDatasetBuilder


def test_scenario_generator():
    scenes = generate_scenarios(n_per_template=2, seed=42)
    assert len(scenes) > 0
    assert all(s.dataset == "prism_ar_synthetic" for s in scenes)


def test_dataset_builder():
    with tempfile.TemporaryDirectory() as tmp:
        builder = PRISMARDatasetBuilder(
            output_dir=tmp,
            n_per_template=1,
            seed=42,
        )
        csv_path = builder.build()
        assert os.path.exists(csv_path)
        assert os.path.exists(os.path.join(tmp, "images", "static"))
        assert os.path.exists(os.path.join(tmp, "images", "adaptive"))
        builder.close()
