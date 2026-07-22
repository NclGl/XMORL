from __future__ import annotations

import os
import random
from typing import Any, Iterable

import numpy as np

# TODO: add a flag to keep all the raw weights to normalize whenever needed/shown

# map objectives and impact strengths to a single token with objective index and impact strength depending on the amount of objectives (e.g. "1S" = prioritize objective 1 with small impact)

FEEDBACK_TOKEN_MAP = {
    "TS": ("treasure", "small"),
    "TB": ("treasure", "big"),
    "TH": ("treasure", "huge"),
    "SS": ("speed", "small"),
    "SB": ("speed", "big"),
    "SH": ("speed", "huge"),
}

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def reset_env_with_seed(env: Any, seed: int):
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
    values = np.asarray(observation).reshape(-1)
    return tuple(int(value) if float(value).is_integer() else float(value) for value in values)

def linear_utility(reward: Any, weights: Iterable[float]) -> float:
    return float(np.dot(np.asarray(tuple(weights), dtype=np.float64), np.asarray(reward, dtype=np.float64)))

def normalize_weights(weights: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in weights)
    if values[0] <= 0 or any(value < 0 for value in values[1:]):
        raise ValueError("Weights must satisfy: first objective > 0 and other objectives >= 0")
    return tuple(float(value) / sum(values) for value in values)

def feedback_schedule(total_episodes: int, num_moments: int, explicit: Iterable[int] | None = None) -> set[int]:
    if explicit is not None:
        return {int(episode) for episode in explicit}
    if num_moments <= 0:
        return set()
    step = max(1, total_episodes // (num_moments + 1))
    return {index * step for index in range(1, num_moments + 1)}

def stage_ratio_map(feedback_number: int) -> dict[str, float]:
    if feedback_number <= 1:
        return {"small": 5.0, "big": 10.0, "huge": 25.0}
    if feedback_number == 2:
        return {"small": 3.0, "big": 5.0, "huge": 10.0}
    return {"small": 2.0, "big": 3.0, "huge": 5.0}

def apply_preference_update(
    current_weights: Iterable[float], direction: str, objective: int, strength: str, feedback_number: int
) -> tuple[tuple[float, ...], str]:
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

def encode_feedback_token(direction: str, strength: str) -> str:
    if direction == "keep":
        return "K"
    for token, choice in FEEDBACK_TOKEN_MAP.items():
        if choice == (direction, strength):
            return token
    raise ValueError(f"Cannot encode feedback choice {direction}/{strength}")

def serialize_weights(weights: Iterable[float]) -> str:
    values = tuple(weights)
    return "(" + ", ".join(f"{value:.10g}" for value in values) + ")"