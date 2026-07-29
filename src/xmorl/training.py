from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

import numpy as np

from xmorl.config import AppConfig
from xmorl.envs.factory import make_env
from xmorl.success import success_value
from xmorl.utils import (
    apply_preference_update,
    feedback_schedule,
    linear_utility,
    normalize_weights,
    obs_to_state,
    reset_env_with_seed,
    set_global_seed,
)

State = tuple[int | float, ...]

@dataclass
class TabularModel:
    """
    (Q-learning with P values)
    Store utility, success, and visit estimates for each state-action pair
    """

    num_actions: int
    q: dict[State, np.ndarray] = field(default_factory=dict)
    p: dict[State, np.ndarray] = field(default_factory=dict)
    visits: dict[State, np.ndarray] = field(default_factory=dict)

    def values(self, state: State) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return a state's tables, creating entries when unseen

        :param state: Hashable tabular state
        :return: Utility, success, and visit arrays
        """
        if state not in self.q:
            self.q[state] = np.zeros(self.num_actions, dtype=np.float64)
            self.p[state] = np.zeros(self.num_actions, dtype=np.float64)
            self.visits[state] = np.zeros(self.num_actions, dtype=np.int64)
        return self.q[state], self.p[state], self.visits[state]

@dataclass
class FeedbackChoice:
    """
    Represent one human choice about the current objective weights
    """

    direction: str = "keep"
    objective: int = -1
    strength: str = "small"

@dataclass
class TrainingCheckpoint:
    """
    Expose the current policy and weights to a feedback provider
    """

    episode: int
    feedback_number: int
    model: TabularModel
    weights: tuple[float, ...]

@dataclass
class TrainingResult:
    """
    Collect the learned model and histories needed for analysis
    """

    model: TabularModel
    scalar_returns: np.ndarray
    vector_returns: np.ndarray
    final_weights: tuple[float, ...]
    weight_history: list[dict]
    correlation_history: list[dict]
    feedback_history: list[dict]

class MetricLogger(Protocol):
    def log(self, metrics: dict, step: int | None = None) -> None:
        """
        Log metrics for later analysis

        :param metrics: Metric names and values
        :param step: Optional training step
        :return: None
        """
        ...

FeedbackProvider = Callable[[TrainingCheckpoint], FeedbackChoice]
FeedbackEpisodeProvider = Callable[[int], Iterable[int]]

def _preference_metric_key(index: int) -> str:
    """
    Return the stable metric name for one objective weight

    :param index: Objective index
    :return: Preference metric name
    """
    return f"preferences/objective_{index + 1}"

def q_p_correlations(model: TabularModel, min_visits: int = 1) -> tuple[float, float]:
    """
    Compare learned utility and success estimates over visited actions
    (intuitively, this tells how useful the success signal is for learning the utility function)

    :param model: Learned tabular estimates
    :param min_visits: Minimum visits required for inclusion
    :return: Pearson and Spearman correlations
    """
    pairs = []
    for state in model.q:
        mask = model.visits[state] >= min_visits
        pairs.extend(zip(model.q[state][mask], model.p[state][mask]))
    if len(pairs) < 2:
        return float("nan"), float("nan")
    q_values, p_values = np.asarray(pairs, dtype=np.float64).T
    if np.std(q_values) == 0 or np.std(p_values) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(q_values, p_values)[0, 1])
    q_ranks = _average_ranks(q_values)
    p_ranks = _average_ranks(p_values)
    spearman = float(np.corrcoef(q_ranks, p_ranks)[0, 1])
    return pearson, spearman

def _average_ranks(values: np.ndarray) -> np.ndarray:
    """
    Assign average ranks while preserving tied values

    :param values: Values to rank
    :return: Average rank for each original position
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks

def train(
    config: AppConfig,
    logger: MetricLogger,
    feedback_provider: FeedbackProvider | None = None,
    feedback_episode_provider: FeedbackEpisodeProvider | None = None,
) -> TrainingResult:
    """
    Learn with epsilon-greedy Q-learning and apply feedback at checkpoints

    :param config: Validated experiment configuration
    :param logger: Training metric sink
    :param feedback_provider: Optional checkpoint feedback callback
    :param feedback_episode_provider: Optional source of future feedback episodes
    :return: Learned model and training histories
    """
    settings = config.training
    weights = normalize_weights(settings.utility_weights)
    set_global_seed(settings.seed)
    env = make_env(config.environment)
    if not hasattr(env.action_space, "n"):
        env.close()
        raise TypeError("Tabular Q-learning requires a discrete action space")
    model = TabularModel(int(env.action_space.n))
    schedule = feedback_schedule(
        settings.num_episodes,
        config.feedback.num_moments,
        config.feedback.episodes,
    ) if config.feedback.enabled else set()

    epsilon = settings.epsilon_start
    scalar_returns: list[float] = []
    vector_returns: list[np.ndarray] = []
    weight_history = [{"episode": 1, "weights": weights}]
    feedback_history: list[dict] = []
    correlation_history: list[dict] = []
    feedback_number = 0

    try:
        for episode in range(1, settings.num_episodes + 1):
            observation, _ = reset_env_with_seed(env, settings.seed + episode)
            state = obs_to_state(observation)
            done = False
            scalar_return = 0.0
            vector_return = np.zeros(len(weights), dtype=np.float64)

            while not done:
                q_values, p_values, visits = model.values(state)
                if np.random.random() < epsilon:
                    action = int(env.action_space.sample())
                else:
                    action = int(np.argmax(q_values))
                next_observation, reward, terminated, truncated, info = env.step(action)
                reward = np.asarray(reward, dtype=np.float64)
                if reward.size != len(weights):
                    raise ValueError(
                        f"Environment returned {reward.size} objectives, but {len(weights)} weights were configured"
                    )
                done = bool(terminated or truncated)
                utility = linear_utility(reward, weights)
                target_p = success_value(reward, terminated, truncated, info, config.environment)
                visits[action] += 1

                if done:
                    target_q = utility
                else:
                    next_state = obs_to_state(next_observation)
                    next_q, next_p, _ = model.values(next_state)
                    greedy_next = int(np.argmax(next_q))
                    target_q = utility + settings.gamma_q * next_q[greedy_next]
                    target_p += settings.gamma_p * next_p[greedy_next]

                q_values[action] += settings.alpha_q * (target_q - q_values[action])
                p_values[action] += settings.alpha_p * (target_p - p_values[action])
                p_values[action] = np.clip(p_values[action], 0.0, 1.0)
                scalar_return += utility
                vector_return += reward
                if not done:
                    state = next_state

            epsilon = max(settings.epsilon_min, epsilon * settings.epsilon_decay)
            scalar_returns.append(scalar_return)
            vector_returns.append(vector_return)

            if episode % config.logging.log_every == 0 or episode == settings.num_episodes:
                pearson, spearman = q_p_correlations(model)
                correlation = {"episode": episode, "pearson": pearson, "spearman": spearman}
                correlation_history.append(correlation)
                logger.log(
                    {
                        "train/return": scalar_return,
                        "train/epsilon": epsilon,
                        "train/q_p_pearson": pearson,
                        "train/q_p_spearman": spearman,
                        **{
                            _preference_metric_key(index): weight
                            for index, weight in enumerate(weights)
                        },
                    },
                    step=episode,
                )

            if feedback_episode_provider is not None:
                schedule.update(feedback_episode_provider(episode))

            if episode in schedule and feedback_provider is not None:
                feedback_number += 1
                choice = feedback_provider(
                    TrainingCheckpoint(episode, feedback_number, model, weights)
                )
                before = weights
                weights, explanation = apply_preference_update(
                    weights, choice.direction, choice.objective, choice.strength, feedback_number
                )
                if weights != before:
                    epsilon = max(epsilon, 0.1)
                    if episode < settings.num_episodes:
                        weight_history.append({"episode": episode + 1, "weights": weights})
                event = {
                    "episode": episode,
                    "feedback_number": feedback_number,
                    "direction": choice.direction,
                    "objective": choice.objective,
                    "strength": choice.strength,
                    "weights_before": before,
                    "weights_after": weights,
                    "explanation": explanation,
                }
                feedback_history.append(event)
                logger.log(
                    {
                        "feedback/index": feedback_number,
                        **{
                            _preference_metric_key(index): weight
                            for index, weight in enumerate(weights)
                        },
                    },
                    step=episode,
                )

            if settings.progress_every and (
                episode % settings.progress_every == 0 or episode == settings.num_episodes
            ):
                recent = scalar_returns[-min(settings.progress_every, len(scalar_returns)):]
                print(
                    f"Episode {episode}/{settings.num_episodes} | "
                    f"mean return {np.mean(recent):.3f} | epsilon {epsilon:.3f} | weights {weights}"
                )
    finally:
        env.close()

    return TrainingResult(
        model=model,
        scalar_returns=np.asarray(scalar_returns, dtype=np.float64),
        vector_returns=np.asarray(vector_returns, dtype=np.float64),
        final_weights=weights,
        weight_history=weight_history,
        correlation_history=correlation_history,
        feedback_history=feedback_history,
    )
