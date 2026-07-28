from __future__ import annotations

import os
import random
from typing import Any, Iterable

import numpy as np

# TODO: add a flag to keep all the raw weights to normalize whenever needed/shown

# FEEDBACK_TOKEN_MAP = {
#     "O1S": ("objective 1", "small"),
#     "O1B": ("objective 1", "big"),
#     "O1H": ("objective 1", "huge"),
#     "O2S": ("objective 2", "small"),
#     "O2B": ("objective 2", "big"),
#     "O2H": ("objective 2", "huge"),
# }

def set_global_seed(seed: int) -> None:
    """
    Seed Python and NumPy for reproducibility

    :param seed: Seed shared by Python and NumPy
    :return: None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def reset_env_with_seed(env: Any, seed: int):
    """
    Reset an environment and seed its sampling spaces consistently

    :param env: Gymnasium-compatible environment
    :param seed: Reset and sampling seed
    :return: Initial observation and reset information
    """
    observation, info = env.reset(seed=seed)
    for space_name in ("action_space", "observation_space"):
        space = getattr(env, space_name, None)
        if space is not None:
            try:
                space.seed(seed)
            except (AttributeError, NotImplementedError):
                pass
    return observation, info

def obs_to_state(observation: Any) -> tuple[int | float, ...]:
    """
    Convert an observation into a tabular state

    :param observation: Environment observation
    :return: Flattened state
    """
    values = np.asarray(observation).reshape(-1)
    return tuple(int(value) if float(value).is_integer() else float(value) for value in values)

def linear_utility(reward: Any, weights: Iterable[float]) -> float:
    """
    Scalarize a reward vector with the current objective weights

    :param reward: Objective reward vector
    :param weights: Objective importance weights
    :return: Weighted scalar utility
    """
    return float(np.dot(np.asarray(tuple(weights), dtype=np.float64), np.asarray(reward, dtype=np.float64)))

def normalize_weights(weights: Iterable[float]) -> tuple[float, ...]:
    """
    Validate nonnegative preferences and normalize them to sum to one

    :param weights: Raw objective weights
    :return: Normalized objective weights
    """
    values = tuple(float(value) for value in weights)
    if values[0] <= 0 or any(value < 0 for value in values[1:]):
        raise ValueError("Weights must satisfy: first objective > 0 and other objectives >= 0")
    return tuple(float(value) / sum(values) for value in values)

def feedback_schedule(total_episodes: int, num_moments: int, explicit: Iterable[int] | None = None) -> set[int]:
    """
    Use explicit feedback episodes or spread moments evenly through training

    :param total_episodes: Number of training episodes
    :param num_moments: Number of evenly spaced feedback moments
    :param explicit: Explicit feedback episode numbers
    :return: Episodes after which feedback is requested
    """
    if explicit is not None:
        return {int(episode) for episode in explicit}
    if num_moments <= 0:
        return set()
    step = max(1, total_episodes // (num_moments + 1))
    return {index * step for index in range(1, num_moments + 1)}

# NOTE: (maybe later TODO) change this so that the config can change the values
def stage_ratio_map(feedback_number: int) -> dict[str, float]:
    """
    Return diminishing feedback multipliers for the current checkpoint

    :param feedback_number: Feedback checkpoint number
    :return: Strength names mapped to weight multipliers
    """
    if feedback_number <= 1:
        return {"small": 5.0, "big": 10.0, "huge": 25.0}
    if feedback_number == 2:
        return {"small": 3.0, "big": 5.0, "huge": 10.0}
    return {"small": 2.0, "big": 3.0, "huge": 5.0}

def apply_preference_update(
    current_weights: Iterable[float], direction: str, objective: int, strength: str, feedback_number: int
) -> tuple[tuple[float, ...], str]:
    """
    Increase one objective by the chosen ratio and renormalize all weights

    :param current_weights: Current objective weights
    :param direction: Keep choice or selected objective label
    :param objective: Index of the objective to prioritize
    :param strength: Requested multiplier strength
    :param feedback_number: Feedback checkpoint number
    :return: Updated weights and explanation
    """
    obj_weights = normalize_weights(current_weights)
    if direction == "keep":
        return (obj_weights), "Keeping current weights."
    if strength not in stage_ratio_map(feedback_number):
        raise ValueError(f"Unknown feedback strength: {strength}")
    multiplier = stage_ratio_map(feedback_number)[strength]
    changed_weight = obj_weights[objective] * multiplier
    new_weights = list(obj_weights)
    new_weights[objective] = changed_weight
    message = f"Prioritizing {direction} more ({strength}, ratio x{multiplier:g})."
    return normalize_weights(new_weights), message

# NOTE: this might be used later if we want to keep tokens
# def encode_feedback_token(direction: str, strength: str) -> str:
#     if direction == "keep":
#         return "K"
#     for token, choice in FEEDBACK_TOKEN_MAP.items():
#         if choice == (direction, strength):
#             return token
#     raise ValueError(f"Cannot encode feedback choice {direction}/{strength}")

def serialize_weights(weights: Iterable[float]) -> str:
    """
    Format objective weights compactly for display and logging

    :param weights: Objective weights
    :return: Compact tuple representation
    """
    values = tuple(weights)
    return "(" + ", ".join(f"{value:.10g}" for value in values) + ")"

def cell_to_pixel(cell, image_shape, shape):
    """
    Map a grid-cell center to its coordinates in a rendered frame

    :param cell: Grid row and column
    :param image_shape: Rendered image dimensions
    :param shape: Grid row and column counts
    :return: Pixel x and y coordinates
    """
    row, col = cell
    image_height, image_width = image_shape[:2]
    rows, cols = shape

    cell_width = image_width / cols
    cell_height = image_height / rows

    x = (col + 0.5) * cell_width
    y = (row + 0.5) * cell_height

    return x, y
