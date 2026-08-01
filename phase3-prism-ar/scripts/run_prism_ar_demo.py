"""End-to-end PRISM-AR demo on synthetic scenes.

This script:
1. Generates synthetic DrivingScene objects for each dataset
2. Runs the PRISM risk engine
3. Maps risk to adaptive and static AR cues
4. Computes evaluation metrics
5. Prints a summary
"""
import json
import numpy as np

from prism_ar.data_ingestion.waymo_loader import WaymoLoader
from prism_ar.data_ingestion.argoverse_loader import ArgoverseLoader
from prism_ar.data_ingestion.nuscenes_loader import NuScenesLoader
from prism_ar.prism.risk_engine import PRISMRiskEngine
from prism_ar.ar_overlay.cue_mapper import AdaptiveARCueMapper, StaticARCueMapper
from prism_ar.evaluation.metrics import evaluate_scenario, compute_min_ttc_per_frame


def compute_min_distances(scene):
    ego = scene.get_ego()
    vrus = scene.get_vrus()
    if not vrus or ego is None:
        return np.full(scene.frames, np.inf)
    return np.min(
        [np.linalg.norm(v.positions - ego.positions, axis=1) for v in vrus],
        axis=0,
    )


def main():
    engine = PRISMRiskEngine()
    adaptive_mapper = AdaptiveARCueMapper()
    static_mapper = StaticARCueMapper()

    loaders = {
        "waymo": WaymoLoader("dummy").make_synthetic_scene(),
        "argoverse": ArgoverseLoader("dummy").make_synthetic_scene(),
        "nuscenes_day": NuScenesLoader("dummy").make_synthetic_scene(lighting="day", weather="clear"),
        "nuscenes_night_rain": NuScenesLoader("dummy").make_synthetic_scene(lighting="night", weather="rain"),
    }

    results = []
    for name, scene in loaders.items():
        output = engine.score(scene)
        adaptive_cues = adaptive_mapper.map(output)
        static_cues = static_mapper.map(output)
        min_distances = compute_min_distances(scene)
        min_ttc = compute_min_ttc_per_frame(scene)
        ego = scene.get_ego()
        timestamps = ego.timestamps if ego is not None else np.arange(scene.frames) / scene.sampling_rate

        result = evaluate_scenario(
            scene_id=name,
            adaptive_cues=adaptive_cues,
            static_cues=static_cues,
            prism_output=output,
            timestamps=timestamps,
            min_distances=min_distances,
            min_ttc=min_ttc,
        )
        results.append(result.to_dict())

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
