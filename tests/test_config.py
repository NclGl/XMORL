import pytest

from xmorl.config import load_config

def test_default_config_loads():
    config = load_config("configs/sdst.yaml")
    assert config.environment.id == "stochastic-deep-sea-treasure-v0"
    assert config.environment.adapter == "sdst"
    assert config.environment.success.indices == (0,)
    assert config.environment.action_names[3] == "right"
    assert config.logging.wandb_mode == "offline"
    assert config.training.utility_weights == (0.5, 0.5)

def test_unknown_option_is_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("training:\n  num_episoeds: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="num_episoeds"):
        load_config(path)

def test_feedback_episode_range_is_validated(tmp_path):
    path = tmp_path / "invalid-feedback.yaml"
    path.write_text("training:\n  num_episodes: 4\nfeedback:\n  episodes: [5]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the training range"):
        load_config(path)

def test_unknown_environment_adapter_is_rejected(tmp_path):
    path = tmp_path / "invalid-adapter.yaml"
    path.write_text("environment:\n  adapter: unknown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="environment.adapter"):
        load_config(path)

def test_invalid_success_reward_index_is_rejected(tmp_path):
    path = tmp_path / "invalid-success.yaml"
    path.write_text(
        "environment:\n  success:\n    type: reward\n    indices: [2]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside the objective range"):
        load_config(path)