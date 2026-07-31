"""Generate controlled PRISM-AR scenarios.

Scenarios are parameterised by:
- weather (clear, rain, snow, fog)
- lighting (day, dusk, night)
- road_condition (dry, wet, ice)
- ego_speed (m/s)
- pedestrian behaviour (waiting, walking, running)
- pedestrian start distance
- presence/absence of VRU

These synthetic scenes follow the PRISM-AR proposal plan (Section 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from prism_ar.data_ingestion.driving_scene import DrivingScene, Agent, SceneAttributes


@dataclass
class ScenarioParams:
    """Parameters for a single PRISM-AR scenario."""
    scenario_id: str
    weather: str
    lighting: str
    road_condition: str
    ego_speed: float
    ped_behaviour: str  # 'none', 'waiting', 'walking', 'running'
    ped_start_distance: float
    duration_s: float = 4.0
    sampling_rate: float = 10.0


def generate_scenario(params: ScenarioParams) -> DrivingScene:
    """Create a synthetic top-down crossing scene from parameters."""
    n_frames = int(params.duration_s * params.sampling_rate)
    t = np.arange(n_frames) / params.sampling_rate

    # Ego moves from x=-ego_speed*duration to x=0 (passing the crosswalk)
    ego_x = np.linspace(-params.ego_speed * params.duration_s, 0.0, n_frames)
    ego_y = np.zeros(n_frames)
    ego_vx = np.full(n_frames, params.ego_speed)
    ego_vy = np.zeros(n_frames)

    agents = {
        "ego": Agent(
            agent_id="ego",
            agent_type="ego",
            positions=np.column_stack([ego_x, ego_y]),
            velocities=np.column_stack([ego_vx, ego_vy]),
            timestamps=t,
            yaw=np.zeros(n_frames),
            width=1.8,
            length=4.5,
        )
    }

    if params.ped_behaviour != "none":
        # Pedestrian is at the curb (x=0) and crosses the road
        ped_x = np.full(n_frames, params.ped_start_distance)
        if params.ped_behaviour == "waiting":
            ped_y = np.full(n_frames, -2.0)
            ped_vy = np.zeros(n_frames)
        elif params.ped_behaviour == "walking":
            speed = 1.0
            ped_y = np.linspace(-2.0, 2.0, n_frames)
            ped_vy = np.full(n_frames, speed)
        elif params.ped_behaviour == "running":
            speed = 2.5
            ped_y = np.linspace(-2.0, 2.0, n_frames)
            ped_vy = np.full(n_frames, speed)
        else:
            raise ValueError(f"Unknown ped_behaviour: {params.ped_behaviour}")

        agents["ped"] = Agent(
            agent_id="ped",
            agent_type="pedestrian",
            positions=np.column_stack([ped_x, ped_y]),
            velocities=np.column_stack([np.zeros(n_frames), ped_vy]),
            timestamps=t,
            width=0.5,
            length=0.5,
        )

    return DrivingScene(
        scene_id=params.scenario_id,
        dataset="prism_ar_synthetic",
        agents=agents,
        attributes=SceneAttributes(
            weather=params.weather,
            lighting=params.lighting,
            road_condition=params.road_condition,
            speed_limit=params.ego_speed * 2.237,  # m/s to mph
            location="synthetic",
        ),
        frames=n_frames,
        sampling_rate=params.sampling_rate,
    )


# Scenario templates matching the PRISM-AR proposal
SCENARIO_TEMPLATES = [
    ("emergency", "rain", "night", "wet", 12.0, "running", 4.0),
    ("intervention", "clear", "dusk", "dry", 8.0, "walking", 6.0),
    ("intervention", "clear", "day", "dry", 6.0, "walking", 5.0),
    ("advisory", "clear", "day", "dry", 5.0, "waiting", 8.0),
    ("advisory", "clear", "day", "dry", 4.0, "walking", 10.0),
    ("silent", "clear", "day", "dry", 5.0, "none", 0.0),
    ("silent", "clear", "day", "dry", 3.0, "waiting", 15.0),
]


def generate_scenarios(
    n_per_template: int = 5,
    seed: int = 42,
) -> List[DrivingScene]:
    """Generate a balanced set of PRISM-AR scenarios.

    Default: 7 templates * 5 variations = 35 scenarios.
    Increase n_per_template to reach the ~200 target.
    """
    rng = np.random.default_rng(seed)
    scenes = []
    counter = 0
    for expected_tier, weather, lighting, road, speed, ped, dist in SCENARIO_TEMPLATES:
        for i in range(n_per_template):
            # Add small random variation to speed and distance
            v = speed * rng.uniform(0.9, 1.1)
            d = dist * rng.uniform(0.8, 1.2)
            params = ScenarioParams(
                scenario_id=f"prism_ar_{expected_tier}_{counter:04d}",
                weather=weather,
                lighting=lighting,
                road_condition=road,
                ego_speed=v,
                ped_behaviour=ped,
                ped_start_distance=d,
            )
            scenes.append(generate_scenario(params))
            counter += 1
    return scenes
