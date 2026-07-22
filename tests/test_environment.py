import numpy as np

from xmorl.config import load_config
from xmorl.envs.factory import make_env

def test_yaml_coordinates_are_accepted_as_bomb_positions():
    config = load_config("configs/sdst.yaml")
    env = make_env(config.environment)
    try:
        observation, _ = env.reset(seed=0)
        assert np.array_equal(observation, [0, 0])
        assert (0, 7) in env.unwrapped.bomb_positions
    finally:
        env.close()

def test_environment_times_out():
    config = load_config("configs/sdst.yaml")
    env = make_env(config.environment, bomb_explosion_prob=0.0, max_steps=2)
    try:
        env.reset(seed=0)
        env.step(2)
        _, _, terminated, truncated, info = env.step(2)
        assert not terminated
        assert truncated
        assert info["terminal_reason"] == "timeout"
    finally:
        env.close()

def test_float_observation_can_be_restored_for_preview():
    config = load_config("configs/sdst.yaml")
    env = make_env(config.environment, float_state=True, bomb_explosion_prob=0.0)
    try:
        env.reset(seed=0)
        observation, *_ = env.step(3)
        env.unwrapped.set_state_for_preview(observation)
        assert np.array_equal(env.unwrapped.current_state, [0, 1])
    finally:
        env.close()