from __future__ import annotations

from typing import Any

class EnvironmentAdapter:
    """
    Adapt optional environment behavior without extending Gym.env
    """

    supports_state_preview = False

    def success_value(self, reward: Any, terminated: bool, truncated: bool, info: dict) -> float | None:
        """
        Return a success value with respect to the environment
        """
        return None

    def set_preview_state(self, env: Any, observation: Any) -> None:
        """
        Place an environment at a recorded state for a one-step preview
        """
        raise NotImplementedError

    def preview_stochasticity(self, env: Any) -> float:
        """
        Return how strongly preview scores should reflect random dynamics
        """
        return 0.0

    def get_shape(self, env: Any) -> tuple[int, ...] | None:
        """
        Infer grid bounds from a bounded observation space when possible
        """
        observation_space = getattr(env, "observation_space", None)
        if observation_space is None:
            return None
        try:
            return int(observation_space.low.min()), int(observation_space.high.max())
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            return None

class SDSTEnvironmentAdapter(EnvironmentAdapter):
    """
    Enable state previews for Stochastic Deep Sea Treasure
    """

    supports_state_preview = True

    def set_preview_state(self, env: Any, observation: Any) -> None:
        """
        Restore the submarine state and restart preview step counting
        """
        unwrapped = env.unwrapped
        unwrapped.set_state_for_preview(observation)
        unwrapped.step_count = 0

    def preview_stochasticity(self, env: Any) -> float:
        """
        Read the configured explosion probability used by preview scoring
        """
        value = getattr(env.unwrapped, "preview_stochasticity", 0.0)
        return 0.0 if value is None else float(value)

_ADAPTERS = {
    "default": EnvironmentAdapter(),
    "sdst": SDSTEnvironmentAdapter(),
}

def environment_adapter(name: str) -> EnvironmentAdapter:
    """
    Return the registered adapter named by the environment configuration
    """
    try:
        return _ADAPTERS[name]
    except KeyError as error:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unknown environment adapter '{name}'. Available adapters: {available}") from error

def environment_adapter_names() -> set[str]:
    """
    Return the adapter names accepted by configuration validation
    """
    return set(_ADAPTERS)
