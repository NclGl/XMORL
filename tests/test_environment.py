import numpy as np

from xmorl.config import load_config
from xmorl.envs.dst.deep_sea_treasure import DeepSeaTreasure
from xmorl.envs.factory import make_env
from xmorl.envs.sdst.stochastic_deep_sea_treasure import StochasticDeepSeaTreasure

def test_sdst_reuses_dst_and_preserves_explosion_priority():
    env = StochasticDeepSeaTreasure(
        bomb_positions=[(1, 0)],
        bomb_explosion_prob=1.0,
        bomb_radius=0,
    )
    try:
        assert isinstance(env, DeepSeaTreasure)
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(1)
        assert np.array_equal(reward, [-20.0, -1.0])
        assert terminated
        assert not truncated
        assert info["terminal_reason"] == "explosion"
    finally:
        env.close()

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
