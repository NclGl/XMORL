from __future__ import annotations

from dataclasses import dataclass

from xmorl.config import AppConfig
from xmorl.evaluation import best_episode, build_replay, evaluate, format_summary, representative_episode
from xmorl.gui import ReplayViewer
from xmorl.logging import RunLogger
from xmorl.training import FeedbackChoice, TrainingCheckpoint, train

@dataclass
class InteractiveFeedback:
    config: AppConfig
    logger: RunLogger

    def __call__(self, checkpoint: TrainingCheckpoint) -> FeedbackChoice:
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

def run_experiment(config: AppConfig):
    with RunLogger(config) as logger:
        feedback = InteractiveFeedback(config, logger) if config.feedback.enabled else None
        result = train(config, logger, feedback_provider=feedback)

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