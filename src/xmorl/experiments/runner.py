from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Thread
from typing import Iterable

from xmorl.config import AppConfig
from xmorl.evaluation import best_episode, build_replay, evaluate, format_summary, representative_episode
from xmorl.gui import ReplayViewer
from xmorl.logging import RunLogger
from xmorl.training import FeedbackChoice, TrainingCheckpoint, train
from xmorl.utils import feedback_schedule

class CLIFeedbackScheduler:
    """
    Collect future feedback episodes from standard input without blocking training
    """

    def __init__(self, total_episodes: int, scheduled: Iterable[int] = ()):
        """
        Prepare a runtime schedule alongside configured feedback episodes

        :param total_episodes: Final training episode
        :param scheduled: Episodes already present in the configured schedule
        :return: None
        """
        self.total_episodes = total_episodes
        self._scheduled = set(scheduled)
        self._requests: Queue[int] = Queue()
        self._stepbacks: Queue[int] = Queue()
        self._stopped = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        """
        Start reading feedback episode numbers

        :return: None
        """
        print(
            "Runtime feedback scheduling is active. Enter a future episode number,"
            "or 'stepback' followed by the feedback episode number, and press Enter."
        )
        self._thread = Thread(target=self._read, name="xmorl-feedback-input", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Stop accepting new feedback episode numbers

        :return: None
        """
        self._stopped.set()

    def submit(self, value: str) -> None:
        """
        Parse and queue one command entered through the CLI

        :param value: Raw CLI input
        :return: None
        """
        parts = value.strip().split()
        if parts and parts[0].lower() == "stepback":
            if len(parts) != 2:
                print("[runner.py] Rejected stepback: use 'stepback' followed by the feedback episode number")
                return
            try:
                self._stepbacks.put(int(parts[1]))
            except ValueError:
                print("[runner.py] Rejected stepback: feedback episode must be an integer")
            return
        try:
            episode = int(value.strip())
        except ValueError:
            print(
                f"[runner.py] Rejected {value!r}: enter an integer episode "
                "or 'stepback' followed by the feedback episode number"
            )
            return
        self._requests.put(episode)

    def poll(self, current_episode: int) -> set[int]:
        """
        Validate queued requests against current training progress

        :param current_episode: Episode that just finished
        :return: Newly accepted future feedback episodes
        """
        accepted = set()
        while True:
            try:
                episode = self._requests.get_nowait()
            except Empty:
                break
            if episode <= current_episode:
                print(
                    f"[runner.py] Rejected episode {episode}: "
                    f"training is already at episode {current_episode}"
                )
            elif episode > self.total_episodes:
                print(
                    f"[runner.py] Rejected episode {episode}: "
                    f"training ends at episode {self.total_episodes}"
                )
            elif episode in self._scheduled:
                print(f"[runner.py] Ignored episode {episode}: feedback is already scheduled")
            else:
                self._scheduled.add(episode)
                accepted.add(episode)
                print(f"[runner.py] Added feedback at episode {episode}")
        return accepted

    def poll_stepbacks(self) -> list[int]:
        """
        Return queued feedback episodes requested as rollback targets

        :return: Requested feedback episodes
        """
        requested = []
        while True:
            try:
                requested.append(self._stepbacks.get_nowait())
            except Empty:
                return requested

    def _read(self) -> None:
        """
        Read standard input until it closes or runtime scheduling stops

        :return: None
        """
        while not self._stopped.is_set():
            try:
                value = input()
            except (EOFError, OSError):
                return
            if not self._stopped.is_set():
                self.submit(value)

@dataclass
class InteractiveFeedback:
    """
    Validate a checkpoint and collect a preference choice through its replay
    """

    config: AppConfig
    logger: RunLogger

    def __call__(self, checkpoint: TrainingCheckpoint) -> FeedbackChoice:
        """
        Evaluate the policy, show a representative replay, and return feedback

        :param checkpoint: Current training state
        :return: User preference choice
        """
        settings = self.config.evaluation
        seeds = range(settings.checkpoint_seed_start, settings.checkpoint_seed_start + settings.checkpoint_episodes)
        results, summary = evaluate(
            self.config.environment,
            checkpoint.model,
            checkpoint.weights,
            seeds,
            settings.max_steps,
        )
        self.logger.log(
            {
                "validation/return": summary["mean_return_scalar"],
                "validation/success_rate": summary["success_rate"],
                "validation/explosion_rate": summary["explosion_rate"],
                "validation/timeout_rate": summary["timeout_rate"],
            },
            step=checkpoint.episode,
        )
        episode = representative_episode(results, successful_only=True)
        if episode is None:
            episode = representative_episode(results, successful_only=False)
        replay = build_replay(
            self.config.environment,
            checkpoint.model,
            checkpoint.weights,
            episode,
            disable_explosions=(
                self.config.feedback.render_without_explosions and bool(episode.success)
            ),
        )
        viewer = ReplayViewer(
            title=f"Feedback checkpoint #{checkpoint.feedback_number}",
            replay=replay,
            summary=format_summary(summary, checkpoint.weights),
            current_weights=checkpoint.weights,
            action_names=self.config.environment.action_names,
            environment=self.config.environment,
            model=checkpoint.model,
            gamma_q=self.config.training.gamma_q,
            preview_samples=self.config.feedback.preview_samples,
            feedback_number=checkpoint.feedback_number,
            long_explanation=self.config.gui.long_explanation,
            trajectory_line=self.config.gui.trajectory_line,
        )
        return viewer.run()

def run_experiment(config: AppConfig, cli_feedback: bool = True):
    """
    Run the complete XMORL lifecycle and save its results

    :param config: Validated experiment configuration
    :param cli_feedback: Whether future feedback episodes can be entered at runtime
    :return: Completed training result
    """
    with RunLogger(config) as logger:
        feedback = InteractiveFeedback(config, logger) if config.feedback.enabled else None
        scheduler = None
        if cli_feedback and feedback is not None:
            scheduled = feedback_schedule(
                config.training.num_episodes,
                config.feedback.num_moments,
                config.feedback.episodes,
            )
            scheduler = CLIFeedbackScheduler(config.training.num_episodes, scheduled)
            scheduler.start()
        try:
            result = train(
                config,
                logger,
                feedback_provider=feedback,
                feedback_episode_provider=scheduler.poll if scheduler is not None else None,
                stepback_provider=scheduler.poll_stepbacks if scheduler is not None else None,
                segment_saver=logger.save_training_segment,
            )
        finally:
            if scheduler is not None:
                scheduler.stop()

        settings = config.evaluation
        seeds = range(settings.final_seed_start, settings.final_seed_start + settings.final_episodes)
        evaluation_results, summary = evaluate(
            config.environment,
            result.model,
            result.final_weights,
            seeds,
            settings.max_steps,
        )
        logger.log({
            "final/return": summary["mean_return_scalar"],
            "final/success_rate": summary["success_rate"],
            "final/explosion_rate": summary["explosion_rate"],
            "final/timeout_rate": summary["timeout_rate"],
        }, step=config.training.num_episodes)
        logger.save_results(result, summary)

        if config.plotting.enabled:
            from xmorl.plotting import plot_learning

            plot_learning(
                result,
                logger.run_dir / "learning_curves.png",
                window=config.plotting.moving_average_window,
                show=config.plotting.show,
            )

        selected = best_episode(evaluation_results, successful_only=True)
        if selected is None:
            selected = best_episode(evaluation_results, successful_only=False)
        replay = build_replay(
            config.environment,
            result.model,
            result.final_weights,
            selected,
            disable_explosions=bool(selected.success),
        )
        final_summary = format_summary(summary, result.final_weights) + (
            f"\nSelected seed: {selected.seed}"
            f"\nSelected return: {selected.scalar_return:.4f}"
            f"\nSelected length: {selected.length}"
        )
        ReplayViewer(
            title="Final replay viewer",
            replay=replay,
            summary=final_summary,
            current_weights=result.final_weights,
            action_names=config.environment.action_names,
            environment=config.environment,
            model=result.model,
            gamma_q=config.training.gamma_q,
            preview_samples=config.feedback.preview_samples,
            long_explanation=config.gui.long_explanation,
            trajectory_line=config.gui.trajectory_line,
        ).run()
        print(f"Run complete. Outputs: {logger.run_dir}")
        return result
