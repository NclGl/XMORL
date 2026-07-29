from dataclasses import replace

import numpy as np

from xmorl.config import load_config
from xmorl.evaluation import evaluate
from xmorl.experiments.runner import CLIFeedbackScheduler
from xmorl.training import FeedbackChoice, train

class MemoryLogger:
    def __init__(self):
        self.events = []

    def log(self, metrics, step=None):
        self.events.append((step, metrics))

def test_short_training_and_evaluation_smoke():
    config = load_config("configs/sdst.yaml")
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

def test_training_accepts_runtime_feedback_episodes():
    config = load_config("configs/sdst.yaml")
    config = replace(
        config,
        training=replace(config.training, num_episodes=4, progress_every=0),
        feedback=replace(config.feedback, enabled=True, episodes=()),
        logging=replace(config.logging, log_every=2, wandb_enabled=False),
    )
    checkpoints = []

    def feedback_provider(checkpoint):
        checkpoints.append(checkpoint.episode)
        return FeedbackChoice()

    def episode_provider(current_episode):
        return {3} if current_episode == 1 else set()

    train(
        config,
        MemoryLogger(),
        feedback_provider=feedback_provider,
        feedback_episode_provider=episode_provider,
    )

    assert checkpoints == [3]

def test_cli_feedback_scheduler_rejects_nonfuture_and_invalid_episodes(capsys):
    scheduler = CLIFeedbackScheduler(total_episodes=10, scheduled={6})
    scheduler.submit("5")
    scheduler.submit("6")
    scheduler.submit("11")
    scheduler.submit("not-an-episode")
    scheduler.submit("8")
    scheduler.submit("stepback 3")
    scheduler.submit("stepback not-an-episode")

    assert scheduler.poll(current_episode=5) == {8}
    assert scheduler.poll_stepbacks() == [3]
    output = capsys.readouterr().out
    assert "Rejected episode 5" in output
    assert "already scheduled" in output
    assert "training ends at episode 10" in output
    assert "enter an integer episode" in output
    assert "feedback episode must be an integer" in output

def test_training_steps_back_to_before_previous_feedback():
    config = load_config("configs/sdst.yaml")
    config = replace(
        config,
        training=replace(config.training, num_episodes=5, progress_every=0),
        feedback=replace(config.feedback, enabled=True, episodes=(2, 4)),
        logging=replace(config.logging, log_every=2, wandb_enabled=False),
    )
    checkpoints = []
    stepped_back = False

    def feedback_provider(checkpoint):
        checkpoints.append((checkpoint.episode, checkpoint.feedback_number, checkpoint.weights))
        if len(checkpoints) == 1:
            return FeedbackChoice("treasure", 0, "small")
        return FeedbackChoice()

    def stepback_provider():
        nonlocal stepped_back
        if len(checkpoints) == 2 and not stepped_back:
            stepped_back = True
            return [2]
        return []

    result = train(
        config,
        MemoryLogger(),
        feedback_provider=feedback_provider,
        stepback_provider=stepback_provider,
    )

    assert [(episode, number) for episode, number, _ in checkpoints] == [(2, 1), (4, 2), (2, 1), (4, 2)]
    assert checkpoints[0][2] == checkpoints[2][2]
    assert [event["episode"] for event in result.feedback_history] == [2, 4]
    assert result.scalar_returns.shape == (5,)
