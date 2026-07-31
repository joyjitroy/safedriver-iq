"""Generate controlled near-miss / collision-proximity synthetic scenarios.

These are designed to produce intervention and emergency PRISM tiers so the
evaluation metrics can exercise the full cue spectrum. They are mixed with
real dataset clips as a controlled stress supplement.
"""
from __future__ import annotations

from typing import List
import numpy as np

from prism_ar.data_ingestion.driving_scene import DrivingScene, Agent, SceneAttributes


def generate_near_miss_scenarios(n_per_type: int = 10, seed: int = 42) -> List[DrivingScene]:
    """Generate extreme near-miss scenarios for intervention/emergency tiers."""
    rng = np.random.default_rng(seed)
    scenes = []
    counter = 0

    # Templates: (weather, lighting, road, ego_speed, ped_speed, ped_start_x, ped_start_y, collision_frame)
    templates = [
        # Ego faster, ped crosses late -> emergency (collision overlap)
        ("rain", "night", "wet", 12.0, 1.2, 0.0, -3.0, 20),
        ("clear", "dusk", "dry", 10.0, 1.0, 0.0, -2.5, 22),
        ("clear", "day", "dry", 10.0, 1.5, 0.0, -3.0, 18),
        # Ego and ped approaching each other -> intervention
        ("clear", "day", "dry", 7.0, 1.0, 0.0, -3.0, 28),
        ("rain", "day", "wet", 6.0, 0.8, 0.0, -2.5, 30),
        # Ped suddenly runs into road from behind -> emergency
        ("clear", "night", "dry", 8.0, 2.5, 0.0, -4.0, 15),
    ]

    for weather, lighting, road, ego_speed, ped_speed, ped_x, ped_y, collision_frame in templates:
        for i in range(n_per_type):
            duration_s = 4.0
            sampling_rate = 10.0
            n_frames = int(duration_s * sampling_rate)
            t = np.arange(n_frames) / sampling_rate

            # Ego trajectory
            ego_x = np.linspace(-ego_speed * duration_s, 0.0, n_frames)
            ego_y = np.zeros(n_frames)
            ego_vx = np.full(n_frames, ego_speed)
            ego_vy = np.zeros(n_frames)

            # Pedestrian and ego meet at collision_frame
            ego_x_at_collision = -ego_speed * duration_s + ego_speed * (collision_frame / sampling_rate)
            ped_x_arr = np.full(n_frames, ego_x_at_collision)

            ped_start_y = ped_y
            ped_end_y = -ped_y
            # Piecewise linear: y goes from start_y to 0 at collision_frame, then to end_y
            ped_y_arr = np.empty(n_frames)
            ped_y_arr[:collision_frame] = np.linspace(ped_start_y, 0.0, collision_frame, endpoint=False)
            ped_y_arr[collision_frame:] = np.linspace(0.0, ped_end_y, n_frames - collision_frame)

            ped_vx = np.zeros(n_frames)
            ped_vy = np.full(n_frames, ped_speed)

            # Add small random perturbation
            ego_speed *= rng.uniform(0.95, 1.05)
            ped_speed *= rng.uniform(0.9, 1.1)
            ego_vx = np.full(n_frames, ego_speed)
            ped_vy = np.full(n_frames, ped_speed)

            scene = DrivingScene(
                scene_id=f"near_miss_{counter:04d}",
                dataset="prism_ar_synthetic",
                agents={
                    "ego": Agent(
                        agent_id="ego",
                        agent_type="ego",
                        positions=np.column_stack([ego_x, ego_y]),
                        velocities=np.column_stack([ego_vx, ego_vy]),
                        timestamps=t,
                        yaw=np.zeros(n_frames),
                        width=1.8,
                        length=4.5,
                    ),
                    "ped": Agent(
                        agent_id="ped",
                        agent_type="pedestrian",
                        positions=np.column_stack([ped_x_arr, ped_y_arr]),
                        velocities=np.column_stack([ped_vx, ped_vy]),
                        timestamps=t,
                        width=0.5,
                        length=0.5,
                    ),
                },
                attributes=SceneAttributes(
                    weather=weather,
                    lighting=lighting,
                    road_condition=road,
                    time_of_day="20:00" if "night" in lighting else "18:00" if "dusk" in lighting else "12:00",
                    location="synthetic",
                ),
                frames=n_frames,
                sampling_rate=sampling_rate,
            )
            scenes.append(scene)
            counter += 1

    return scenes
