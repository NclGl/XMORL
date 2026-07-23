from os import path
from typing import List, Optional, Tuple

import numpy as np
import pygame
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle

from xmorl.envs.dst.deep_sea_treasure import (
    CONCAVE_FRONT,
    CONCAVE_MAP,
    CONVEX_FRONT,
    DEFAULT_MAP,
    MIRRORED_MAP,
    DeepSeaTreasure,
)

# same probability everywhere within the radius
def constant_in_radius(distance, radius, base_prob):
    return base_prob

# probability halves for each additional Manhattan distance away from the bomb
def half_per_distance(distance, radius, base_prob):
    return base_prob * (0.5**distance)

class StochasticDeepSeaTreasure(DeepSeaTreasure):
    """Deep Sea Treasure with stochastic bomb termination.

    The environment behaves like DeepSeaTreasure with static bombs
    that can terminate an episode when the submarine enters their radius
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        dst_map=DEFAULT_MAP,
        float_state=False,
        bomb_positions: Optional[List[Tuple[int, int]]] = None,
        bomb_explosion_prob=0.5,
        bomb_radius=1,
        bomb_penalty=-20.0,
        bomb_explosion_prob_fn=constant_in_radius,
        max_steps=200,
    ):
        super().__init__(
            render_mode=render_mode,
            dst_map=dst_map,
            float_state=float_state,
        )
        EzPickle.__init__(
            self,
            render_mode,
            dst_map,
            float_state,
            bomb_positions,
            bomb_explosion_prob,
            bomb_radius,
            bomb_penalty,
            bomb_explosion_prob_fn,
            max_steps,
        )

        self.reward_space = Box(
            low=np.array([min(bomb_penalty, 0.0), -1], dtype=np.float32),
            high=np.array([np.max(self.sea_map), -1], dtype=np.float32),
            dtype=np.float32,
        )
        self.preview_terminal_reward_vector = np.array(
            [float(np.max(self.sea_map)), -1.0],
            dtype=np.float64,
        )
        self.preview_stochasticity = float(bomb_explosion_prob)

        self.bomb_positions = {
            tuple(int(value) for value in position)
            for position in (bomb_positions or [])
        }
        self.bomb_explosion_prob = bomb_explosion_prob
        self.bomb_radius = bomb_radius
        self.bomb_penalty = bomb_penalty
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self._validate_bombs()

        self.bomb_explosion_prob_fn = bomb_explosion_prob_fn
        self.explosion_cells = set()
        self.step_count = 0
        self.bomb_img = None
        self.explosion_img = None

    def _validate_bombs(self):
        for pos in self.bomb_positions:
            if not self._is_in_bounds(np.array(pos, dtype=np.int32)):
                raise ValueError(f"Bomb out of bounds: {pos}")
            if self._get_map_value(pos) == -10:
                raise ValueError(f"Bomb on rock: {pos}")

    def _is_in_bounds(self, state):
        if self.map_name == "mirrored":
            return 0 <= state[0] <= 10 and 0 <= state[1] <= 19
        return 0 <= state[0] <= 10 and 0 <= state[1] <= 10

    def _is_bomb(self, state):
        return tuple(state) in self.bomb_positions

    def _bomb_explodes(self):
        return self.np_random.random() < self.bomb_explosion_prob

    def _bombs_in_range(self, state):
        if not self.bomb_positions:
            return []

        sx, sy = state
        in_range = []
        for bx, by in self.bomb_positions:
            distance = abs(sx - bx) + abs(sy - by)
            if distance <= self.bomb_radius:
                in_range.append((bx, by, distance))
        return in_range

    def _explosion_overlay(self, center):
        cx, cy = center
        cells = set()
        for dx in range(-self.bomb_radius, self.bomb_radius + 1):
            remaining = self.bomb_radius - abs(dx)
            for dy in range(-remaining, remaining + 1):
                x = cx + dx
                y = cy + dy
                if self._is_in_bounds([x, y]):
                    cells.add((x, y))
        return cells

    def _load_additional_render_assets(self):
        if self.bomb_img is None:
            filename = path.join(path.dirname(__file__), "assets", "bomb.png")
            self.bomb_img = pygame.transform.scale(
                pygame.image.load(filename),
                self.pix_square_size,
            )
        if self.explosion_img is None:
            filename = path.join(path.dirname(__file__), "assets", "explosion.png")
            self.explosion_img = pygame.transform.scale(
                pygame.image.load(filename),
                self.pix_square_size,
            )

    def _draw_before_agent(self):
        for bx, by in self.bomb_positions:
            bomb_rect = pygame.Rect(
                by * self.pix_square_size[0] + 8,
                bx * self.pix_square_size[1] + 8,
                self.pix_square_size[0] - 16,
                self.pix_square_size[1] - 16,
            )
            self.window.blit(self.bomb_img, bomb_rect)

    def _draw_after_agent(self):
        for ex, ey in self.explosion_cells:
            explosion_rect = pygame.Rect(
                ey * self.pix_square_size[0] + 8,
                ex * self.pix_square_size[1] + 8,
                self.pix_square_size[0] - 16,
                self.pix_square_size[1] - 16,
            )
            self.window.blit(self.explosion_img, explosion_rect)

    def set_state_for_preview(self, observation):
        """Set a state for GUI one-step previews without resetting dynamics."""
        state = np.asarray(observation, dtype=np.float32)
        if self.float_state:
            scale = np.maximum(
                np.asarray(self.sea_map.shape, dtype=np.float32) - 1,
                1,
            )
            state = np.rint(state * scale)
        state = state.astype(np.int32)
        if not self._is_in_bounds(state):
            raise ValueError(f"Preview state is outside the map: {tuple(state)}")
        self.current_state = state

    def reset(self, seed=None, **kwargs):
        self.explosion_cells = set()
        self.step_count = 0
        return super().reset(seed=seed, **kwargs)

    def step(self, action):
        next_state = self.current_state + self.dir[int(action)]
        if self._is_valid_state(next_state):
            self.current_state = next_state

        self.step_count += 1
        self.explosion_cells = set()
        exploded_center = None
        for bx, by, distance in self._bombs_in_range(self.current_state):
            explosion_prob = self.bomb_explosion_prob_fn(
                distance,
                self.bomb_radius,
                self.bomb_explosion_prob,
            )
            if self.np_random.random() < explosion_prob:
                exploded_center = (bx, by)
                break

        terminal_reason = None
        if exploded_center is not None:
            treasure_value = self.bomb_penalty
            terminated = True
            self.explosion_cells = self._explosion_overlay(exploded_center)
            terminal_reason = "explosion"
        else:
            treasure_value = self._get_map_value(self.current_state)
            if treasure_value == 0 or treasure_value == -10:
                treasure_value = 0.0
                terminated = False
            else:
                terminated = True
                terminal_reason = "treasure"

        truncated = False
        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            terminal_reason = "timeout"

        vec_reward = np.array([treasure_value, -1.0], dtype=np.float32)
        state = self._get_state()
        info = {
            "terminal_reason": terminal_reason,
            "step_count": self.step_count,
        }
        if self.render_mode == "human":
            self.render()
        return state, vec_reward, terminated, truncated, info