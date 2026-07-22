from __future__ import annotations

from typing import Any

import gymnasium as gym

from xmorl.config import EnvironmentConfig

def make_env(config: EnvironmentConfig, render_mode: str | None = None, **overrides: Any):
    import xmorl.envs  # noqa: F401 - registration side effect

    kwargs = dict(config.kwargs)
    kwargs.update(overrides)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    # Gymnasium's scalar-reward checker rejects valid multi-objective reward
    # vectors, so XMORL environments perform their own validation.
    return gym.make(config.id, disable_env_checker=True, **kwargs)

def without_explosions(config: EnvironmentConfig) -> EnvironmentConfig:
    kwargs = dict(config.kwargs)
    if "bomb_explosion_prob" in kwargs or config.id.startswith("stochastic-deep-sea-treasure"):
        kwargs["bomb_explosion_prob"] = 0.0
    return EnvironmentConfig(
        id=config.id,
        kwargs=kwargs,
        action_names=dict(config.action_names),
        objective_names=tuple(config.objective_names),
    )
