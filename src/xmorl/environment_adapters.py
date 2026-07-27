from __future__ import annotations

from typing import Any

class EnvironmentAdapter: # TODO: inherits from Gym.env
    supports_state_preview = False

    def success_value(self, reward: Any, terminated: bool, truncated: bool, info: dict) -> float | None:
        return None

    def set_preview_state(self, env: Any, observation: Any) -> None:
        raise NotImplementedError

    def preview_stochasticity(self, env: Any) -> float:
        return 0.0

    def get_shape(self, env: Any) -> tuple[int, ...] | None:
        # TODO
        pass
            # observation_space = getattr(env, "observation_space", None)
            # if observation_space is not None:
            #     print("================================")
            #     print(f"observation_space: {observation_space}") # returns Box(0, 5, (4,), int32) 
            #     print("================================")
            #     try:
            #         print("================================")
            #         print(f"shape: {observation_space.shape}")
            #         print("================================")
                    
            #         print("================================")
            #         print(f"low: {observation_space.low}")
            #         print("================================")
                    
            #         print("================================")
            #         print(f"high: {observation_space.high}")
            #         print("================================")

            #         input("")
                    
            #         return tuple(int(value) for value in shape)
            #     except (AttributeError, NotImplementedError):
            #         pass

class SDSTEnvironmentAdapter(EnvironmentAdapter):
    supports_state_preview = True

    def set_preview_state(self, env: Any, observation: Any) -> None:
        unwrapped = env.unwrapped
        unwrapped.set_state_for_preview(observation)
        unwrapped.step_count = 0

    def preview_stochasticity(self, env: Any) -> float:
        value = getattr(env.unwrapped, "preview_stochasticity", 0.0)
        return 0.0 if value is None else float(value)

_ADAPTERS = {
    "default": EnvironmentAdapter(),
    "sdst": SDSTEnvironmentAdapter(),
}

def environment_adapter(name: str) -> EnvironmentAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as error:
        available = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unknown environment adapter '{name}'. Available adapters: {available}") from error

def environment_adapter_names() -> set[str]:
    return set(_ADAPTERS)