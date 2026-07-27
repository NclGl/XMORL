import numpy as np

from xmorl.config import EnvironmentConfig
from xmorl.environment_adapters import environment_adapter
from xmorl.envs.factory import without_explosions
from xmorl.gui import estimate_state_preview
from xmorl.training import TabularModel

def test_default_adapter_uses_learned_preview_without_creating_an_environment():
    fallback = np.array([0.2, 0.8])
    result = estimate_state_preview(
        EnvironmentConfig(id="not-registered", adapter="default"),
        TabularModel(num_actions=2),
        state=(0, 0),
        weights=(0.5, 0.5),
        gamma_q=0.99,
        seed=0,
        samples=2,
        fallback=fallback,
    )
    assert np.array_equal(result, fallback)
    assert result is not fallback

def test_sdst_adapter_restores_state_and_preview_dynamics():
    class PreviewEnvironment:
        def __init__(self):
            self.unwrapped = self
            self.step_count = 9
            self.preview_stochasticity = 0.25
            self.observation = None

        def set_state_for_preview(self, observation):
            self.observation = observation

    env = PreviewEnvironment()
    adapter = environment_adapter("sdst")
    adapter.set_preview_state(env, (2, 3))
    assert env.observation == (2, 3)
    assert env.step_count == 0
    assert adapter.preview_stochasticity(env) == 0.25

def test_environment_clone_preserves_adapter():
    config = EnvironmentConfig(
        id="stochastic-deep-sea-treasure-v0",
        adapter="sdst",
    )
    clone = without_explosions(config)
    assert clone.adapter == "sdst"
    assert clone.success == config.success
