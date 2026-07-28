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

        :param reward: Objective reward vector
        :param terminated: Whether a terminal state was reached
        :param truncated: Whether the episode ended by a limit
        :param info: Environment transition metadata
        :return: Success value or None to use configured rules
        """
        return None

    def set_preview_state(self, env: Any, observation: Any) -> None:
        """
        Place an environment at a recorded state for a one-step preview

        :param env: Environment to reposition
        :param observation: Recorded state observation
        :return: None
        """
        raise NotImplementedError

    def preview_stochasticity(self, env: Any) -> float:
        """
        Return how strongly preview scores should reflect random dynamics

        :param env: Environment being previewed
        :return: Stochasticity blend factor
        """
        return 0.0

    def get_shape(self, env: Any) -> tuple[int, ...] | None:
        """
        Infer grid bounds from a bounded observation space when possible

        :param env: Environment whose grid is inspected
        :return: Grid bounds when available
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

        :param env: Stochastic Deep Sea Treasure environment
        :param observation: Recorded submarine state
        :return: None
        """
        unwrapped = env.unwrapped
        unwrapped.set_state_for_preview(observation)
        unwrapped.step_count = 0

    def preview_stochasticity(self, env: Any) -> float:
        """
        Read the configured explosion probability used by preview scoring

        :param env: Stochastic Deep Sea Treasure environment
        :return: Configured preview stochasticity
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

    :param name: Registered adapter name
    :return: Matching adapter instance
    """
    try:
        return _ADAPTERS[name]
    except KeyError as error:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unknown environment adapter '{name}'. Available adapters: {available}") from error

def environment_adapter_names() -> set[str]:
    """
    Return the adapter names accepted by configuration validation

    :return: Registered adapter names
    """
    return set(_ADAPTERS)
