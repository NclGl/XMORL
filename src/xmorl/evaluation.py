from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from xmorl.config import EnvironmentConfig
from xmorl.envs.factory import make_env, without_explosions
from xmorl.training import TabularModel, success_value
from xmorl.utils import linear_utility, obs_to_state, reset_env_with_seed, set_global_seed

@dataclass
class EpisodeResult:
    seed: int
    scalar_return: float
    vector_return: np.ndarray
    actions: list[int]
    length: int
    success: float
    exploded: float
    timed_out: float

def rollout(
    environment: EnvironmentConfig,
    model: TabularModel,
    weights: tuple[float, ...],
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    env = make_env(environment)
    try:
        observation, _ = reset_env_with_seed(env, seed)
        state = obs_to_state(observation)
        actions: list[int] = []
        vector_return = np.zeros(len(weights), dtype=np.float64)
        scalar_return = 0.0
        success = exploded = timed_out = 0.0
        for length in range(1, max_steps + 1):
            q_values, _, _ = model.values(state)
            action = int(np.argmax(q_values))
            actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            reward = np.asarray(reward, dtype=np.float64)
            vector_return += reward
            scalar_return += linear_utility(reward, weights)
            if terminated or truncated:
                success = success_value(reward, terminated, truncated, info)
                exploded = float(info.get("terminal_reason") == "explosion")
                timed_out = float(truncated or info.get("terminal_reason") == "timeout")
                break
            state = obs_to_state(observation)
        else:
            length = max_steps
            timed_out = 1.0
        return EpisodeResult(seed, scalar_return, vector_return, actions, length, success, exploded, timed_out)
    finally:
        env.close()

def evaluate(
    environment: EnvironmentConfig,
    model: TabularModel,
    weights: tuple[float, ...],
    seeds: Iterable[int],
    max_steps: int,
) -> tuple[list[EpisodeResult], dict]:
    results = [rollout(environment, model, weights, int(seed), max_steps) for seed in seeds]
    if not results:
        raise ValueError("Evaluation requires at least one seed")
    scalar = np.asarray([item.scalar_return for item in results])
    vectors = np.asarray([item.vector_return for item in results])
    summary = {
        "num_episodes": len(results),
        "mean_return_scalar": float(np.mean(scalar)),
        "std_return_scalar": float(np.std(scalar)),
        "mean_return_vec": np.mean(vectors, axis=0),
        "success_rate": float(np.mean([item.success for item in results])),
        "explosion_rate": float(np.mean([item.exploded for item in results])),
        "timeout_rate": float(np.mean([item.timed_out for item in results])),
        "mean_length": float(np.mean([item.length for item in results])),
        "num_successes": int(sum(item.success for item in results)),
    }
    return results, summary

def representative_episode(results: list[EpisodeResult], successful_only: bool = True) -> EpisodeResult | None:
    candidates = [item for item in results if item.success] if successful_only else list(results)
    if not candidates:
        return None
    mean_return = float(np.mean([item.scalar_return for item in candidates]))
    mean_length = float(np.mean([item.length for item in candidates]))
    return min(candidates, key=lambda item: (abs(item.scalar_return - mean_return), abs(item.length - mean_length), item.seed))

def best_episode(results: list[EpisodeResult], successful_only: bool = True) -> EpisodeResult | None:
    candidates = [item for item in results if item.success] if successful_only else list(results)
    return max(candidates, key=lambda item: item.scalar_return) if candidates else None

def build_replay(
    environment: EnvironmentConfig,
    model: TabularModel,
    weights: tuple[float, ...],
    episode: EpisodeResult,
    disable_explosions: bool,
) -> dict:
    replay_config = without_explosions(environment) if disable_explosions else environment
    env = make_env(replay_config, render_mode="rgb_array")
    entries = []
    cumulative_return = 0.0
    set_global_seed(episode.seed)
    try:
        observation, _ = reset_env_with_seed(env, episode.seed)
        state = obs_to_state(observation)
        for index, action in enumerate(episode.actions):
            q_values, p_values, _ = model.values(state)
            frame = np.asarray(env.render())
            next_observation, reward, terminated, truncated, info = env.step(action)
            reward = np.asarray(reward, dtype=np.float64)
            utility = linear_utility(reward, weights)
            next_state = obs_to_state(next_observation)
            entries.append({
                "index": index,
                "frame": frame,
                "state": state,
                "q_values": q_values.copy(),
                "p_values": p_values.copy(),
                "action": int(action),
                "reward_vec": reward,
                "utility_reward": utility,
                "cumulative_return": cumulative_return + utility,
                "next_state": next_state,
                "terminal_reason": info.get("terminal_reason"),
            })
            cumulative_return += utility
            state = next_state
            if terminated or truncated:
                break
        q_values, p_values, _ = model.values(state)
        entries.append({
            "index": len(entries), "frame": np.asarray(env.render()), "state": state,
            "q_values": q_values.copy(), "p_values": p_values.copy(), "action": None,
            "reward_vec": None, "utility_reward": None, "cumulative_return": cumulative_return,
            "next_state": None, "terminal_reason": entries[-1]["terminal_reason"] if entries else None,
        })
    finally:
        env.close()
    return {"seed": episode.seed, "entries": entries, "display_explosions_disabled": disable_explosions}


def format_summary(summary: dict, weights: tuple[float, ...]) -> str:
    return "\n".join((
        f"Current weights: {weights}",
        f"Mean utility return: {summary['mean_return_scalar']:.4f} ± {summary['std_return_scalar']:.4f}",
        f"Mean vector return: {np.asarray(summary['mean_return_vec']).tolist()}",
        f"Success rate: {summary['success_rate']:.1%}",
        f"Explosion rate: {summary['explosion_rate']:.1%}",
        f"Timeout rate: {summary['timeout_rate']:.1%}",
        f"Mean episode length: {summary['mean_length']:.2f}",
    ))