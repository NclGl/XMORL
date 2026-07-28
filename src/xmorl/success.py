from __future__ import annotations

import numpy as np

from xmorl.config import EnvironmentConfig
from xmorl.environment_adapters import environment_adapter

def success_value(
    reward: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: dict,
    environment: EnvironmentConfig | None = None,
) -> float:
    """
    Resolve success through info, an adapter, or the configured fallback rule

    :param reward: Objective reward vector
    :param terminated: Whether the environment reached a terminal state
    :param truncated: Whether the episode ended by a limit
    :param info: Environment transition metadata
    :param environment: Optional success configuration
    :return: Binary success value
    """
    if "success" in info:
        return float(bool(info["success"]))
    if environment is None:
        return _automatic_success(reward, terminated, truncated)
    adapter_value = environment_adapter(environment.adapter).success_value(reward, terminated, truncated, info)
    if adapter_value is not None:
        return float(adapter_value)
    rule = environment.success
    if rule.type == "auto":
        return _automatic_success(reward, terminated, truncated)
    if rule.require_terminated and (truncated or not terminated):
        return 0.0
    if rule.type == "info":
        return float(bool(info.get(rule.key, False)))
    if rule.type == "terminal_reason":
        return float(info.get("terminal_reason") in rule.values)
    values = np.asarray(reward, dtype=np.float64).reshape(-1)
    invalid = [index for index in rule.indices if index >= values.size]
    if invalid:
        raise ValueError(f"Success reward indices outside the returned reward vector: {invalid}")
    selected = values[list(rule.indices)]
    if rule.condition == "any_positive":
        return float(np.any(selected > rule.threshold))
    if rule.condition == "all_positive":
        return float(np.all(selected > rule.threshold))
    return float(np.sum(selected) > rule.threshold)

def _automatic_success(reward: np.ndarray, terminated: bool, truncated: bool) -> float:
    """
    Treat the first reward value as a success indicator if the episode terminated normally

    :param reward: Objective reward vector
    :param terminated: Whether the environment reached a terminal state
    :param truncated: Whether the episode ended by a limit
    :return: Binary success value
    """
    if truncated or not terminated:
        return 0.0
    values = np.asarray(reward).reshape(-1)
    return float(values.size > 0 and values[0] > 0)
