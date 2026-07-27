import numpy as np

from xmorl.config import EnvironmentConfig, SuccessConfig, load_config
from xmorl.success import success_value

def test_resource_gathering_success_uses_returned_resources():
    environment = load_config("configs/resource.yaml").environment
    assert success_value(np.array([0, 1, 0]), True, False, {}, environment) == 1.0
    assert success_value(np.array([0, 0, 1]), True, False, {}, environment) == 1.0
    assert success_value(np.array([-1, 0, 0]), True, False, {}, environment) == 0.0
    assert success_value(np.array([0, 0, 0]), True, False, {}, environment) == 0.0

def test_four_room_success_requires_the_goal_reward():
    environment = load_config("configs/4room.yaml").environment
    assert success_value(np.array([1, 1, 1]), True, False, {}, environment) == 1.0
    assert success_value(np.array([1, 0, 0]), True, False, {}, environment) == 0.0

def test_sdst_success_distinguishes_treasure_from_explosion():
    environment = load_config("configs/sdst.yaml").environment
    assert success_value(np.array([0.7, -1]), True, False, {}, environment) == 1.0
    assert success_value(np.array([-20, -1]), True, False, {}, environment) == 0.0

def test_environment_success_info_is_authoritative():
    environment = load_config("configs/resource.yaml").environment
    reward = np.array([0, 1, 0])
    assert success_value(reward, True, False, {"success": False}, environment) == 0.0

def test_terminal_reason_and_custom_info_rules_support_other_environments():
    terminal_reason_environment = EnvironmentConfig(
        success=SuccessConfig(
            type="terminal_reason",
            values=("goal", "treasure"),
        ),
    )
    custom_info_environment = EnvironmentConfig(
        success=SuccessConfig(
            type="info",
            key="goal_reached",
            require_terminated=False,
        ),
    )
    reward = np.array([0])
    assert success_value(reward, True, False, {"terminal_reason": "goal"}, terminal_reason_environment) == 1.0
    assert success_value(reward, False, False, {"goal_reached": True}, custom_info_environment) == 1.0

def test_automatic_rule_preserves_the_previous_fallback():
    reward = np.array([1, 0])
    assert success_value(reward, True, False, {}) == 1.0
    assert success_value(reward, False, False, {}) == 0.0
