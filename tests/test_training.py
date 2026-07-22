from dataclasses import replace

import numpy as np

from xmorl.config import load_config
from xmorl.evaluation import evaluate
from xmorl.training import train

class MemoryLogger:
    def __init__(self):
        self.events = []

    def log(self, metrics, step=None):
        self.events.append((step, metrics))

def test_short_training_and_evaluation_smoke():
    config = load_config("configs/default.yaml")
    config = replace(
        config,
        training=replace(config.training, num_episodes=8, progress_every=0),
        feedback=replace(config.feedback, enabled=False),
        logging=replace(config.logging, log_every=2, wandb_enabled=False),
    )
    logger = MemoryLogger()
    result = train(config, logger)

    assert result.scalar_returns.shape == (8,)
    assert result.vector_returns.shape == (8, 2)
    assert result.model.q
    assert logger.events

    episodes, summary = evaluate(
        config.environment,
        result.model,
        result.final_weights,
        seeds=(100, 101),
        max_steps=10,
    )
    assert len(episodes) == 2
    assert np.isfinite(summary["mean_return_scalar"])