"""2D top-down AR overlay renderer using pygame.

This is a lightweight, CARLA-free renderer that produces the paired
(static vs adaptive) images and per-frame annotations required by the
PRISM-AR dataset generation plan.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pygame

from prism_ar.data_ingestion.driving_scene import DrivingScene, Agent
from prism_ar.ar_overlay.cue_mapper import ARCue


# Colors (RGBA)
COLOR_ROAD = (50, 50, 50)
COLOR_LANE = (200, 200, 200)
COLOR_SIDEWALK = (150, 150, 150)
COLOR_EGO = (0, 120, 255)
COLOR_PED = (255, 200, 0)
COLOR_SAFE_ZONE = (0, 255, 0, 80)
COLOR_DANGER_ZONE = (255, 0, 0, 100)
COLOR_STOP_LINE = (255, 255, 0)
COLOR_NO_CROSS = (255, 0, 0)


@dataclass
class RenderFrame:
    """A single rendered frame with annotations."""
    timestamp: float
    ego_speed: float
    vru_distance: float
    lighting: str
    weather: str
    prism_score: float
    prism_tier: str
    top_factor: str
    static_cue: str
    adaptive_cue: str
    under_warning: bool


class TopDownARRenderer:
    """Render top-down road scenes with optional AR overlays."""

    def __init__(self, width: int = 800, height: int = 600, scale: float = 20.0):
        self.width = width
        self.height = height
        self.scale = scale  # pixels per meter
        pygame.init()
        self.surface = pygame.Surface((width, height))

    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters, x right, y up) to screen pixels."""
        screen_x = int(self.width / 2 + x * self.scale)
        screen_y = int(self.height / 2 - y * self.scale)
        return screen_x, screen_y

    def _draw_road(self, scene: DrivingScene):
        self.surface.fill(COLOR_SIDEWALK)
        # Road surface
        road_rect = pygame.Rect(0, self.height // 2 - 40, self.width, 80)
        pygame.draw.rect(self.surface, COLOR_ROAD, road_rect)
        # Center line
        pygame.draw.line(self.surface, COLOR_LANE, (0, self.height // 2), (self.width, self.height // 2), 2)
        # Crosswalk area at x=0
        crosswalk = pygame.Rect(self.width // 2 - 50, self.height // 2 - 50, 100, 100)
        pygame.draw.rect(self.surface, (220, 220, 220), crosswalk)

    def _draw_agent(self, agent: Agent, frame_idx: int, color: Tuple[int, int, int]):
        x, y = agent.positions[frame_idx]
        sx, sy = self._world_to_screen(x, y)
        w = int(agent.width * self.scale)
        l = int(agent.length * self.scale)
        rect = pygame.Rect(sx - l // 2, sy - w // 2, l, w)
        pygame.draw.rect(self.surface, color, rect)

    def _draw_overlay(self, cue: ARCue, scene: DrivingScene, frame_idx: int):
        ego = scene.get_ego()
        if ego is None:
            return
        ex, ey = ego.positions[frame_idx]
        esx, esy = self._world_to_screen(ex, ey)

        if cue.show_danger_zone:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            danger_rect = pygame.Rect(esx - 30, self.height // 2 - 50, 60, 100)
            pygame.draw.rect(overlay, COLOR_DANGER_ZONE, danger_rect)
            self.surface.blit(overlay, (0, 0))

        if cue.show_stop_line:
            stop_x = esx + 40
            pygame.draw.line(self.surface, COLOR_STOP_LINE, (stop_x, self.height // 2 - 50), (stop_x, self.height // 2 + 50), 4)

        if cue.show_no_cross:
            pygame.draw.line(self.surface, COLOR_NO_CROSS, (self.width // 2 - 50, self.height // 2 - 55), (self.width // 2 + 50, self.height // 2 - 55), 6)

        if cue.show_pedestrian_indicator:
            vrus = scene.get_vrus()
            if vrus:
                vx, vy = vrus[0].positions[frame_idx]
                vsx, vsy = self._world_to_screen(vx, vy)
                pygame.draw.circle(self.surface, (255, 255, 255), (vsx, vsy), 6)

    def render_frame(
        self,
        scene: DrivingScene,
        frame_idx: int,
        cue: ARCue,
    ) -> np.ndarray:
        """Render one frame and return RGB array."""
        self._draw_road(scene)
        ego = scene.get_ego()
        if ego:
            self._draw_agent(ego, frame_idx, COLOR_EGO)
        for vru in scene.get_vrus():
            self._draw_agent(vru, frame_idx, COLOR_PED)
        self._draw_overlay(cue, scene, frame_idx)
        return pygame.surfarray.array3d(self.surface).swapaxes(0, 1)

    def save_frame(self, scene: DrivingScene, frame_idx: int, cue: ARCue, path: str):
        """Render frame and save to disk."""
        rgb = self.render_frame(scene, frame_idx, cue)
        # Save via pygame
        pygame_image = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        pygame.image.save(pygame_image, path)

    def render_sequence(
        self,
        scene: DrivingScene,
        cues: List[ARCue],
        mode: str,
        output_dir: str,
    ) -> List[str]:
        """Render all frames of a scene and save images. Returns paths."""
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i, cue in enumerate(cues):
            path = os.path.join(output_dir, f"{scene.scene_id}_{mode}_{i:04d}.png")
            self.save_frame(scene, i, cue, path)
            paths.append(path)
        return paths

    def close(self):
        pygame.quit()
