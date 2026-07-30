from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import random
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
    num_objectives: int = 1
    q: dict[State, np.ndarray] = field(default_factory=dict)
    objective_q: dict[State, np.ndarray] = field(default_factory=dict)
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
            self.objective_q[state] = np.zeros(
                (self.num_actions, self.num_objectives),
                dtype=np.float64,
            )
            self.p[state] = np.zeros(self.num_actions, dtype=np.float64)
            self.visits[state] = np.zeros(self.num_actions, dtype=np.int64)
        return self.q[state], self.p[state], self.visits[state]

    def objective_values(self, state: State) -> np.ndarray:
        """
        Return action values for a state of an objective

        :param state: Hashable tabular state
        :return: Action-by-objective value table
        """
        self.values(state)
        return self.objective_q[state]

    def scalar_values(
        self,
        state: State,
        weights: tuple[float, ...],
    ) -> np.ndarray:
        """
        Scalarize objective values for a state with current preferences

        :param state: Hashable tabular state
        :param weights: Current objective weights
        :return: Preference action values
        """
        if len(weights) != self.num_objectives:
            raise ValueError(
                f"Model has {self.num_objectives} objectives but received {len(weights)} weights"
            )
        q_values, _, _ = self.values(state)
        q_values[:] = self.objective_q[state] @ np.asarray(weights, dtype=np.float64)
        return q_values

    def scalarize(self, weights: tuple[float, ...]) -> None:
        """
        Refresh stored scalar values for all known states

        :param weights: Current objective weights
        :return: None
        """
        for state in tuple(self.q):
            self.scalar_values(state, weights)

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

@dataclass
class TrainingSnapshot:
    """
    Preserve training state immediately before feedback is collected
    """

    episode: int
    feedback_number: int
    model: TabularModel
    weights: tuple[float, ...]
    epsilon: float
    scalar_returns: list[float]
    vector_returns: list[np.ndarray]
    weight_history: list[dict]
    correlation_history: list[dict]
    feedback_history: list[dict]
    numpy_random_state: tuple
    python_random_state: object

@dataclass
class TrainingSegment:
    """
    Preserve one completed training interval between feedback moments
    """

    start_episode: int
    end_episode: int
    start_feedback_number: int
    weights: tuple[float, ...]
    feedback_event: dict
    episode_numbers: np.ndarray
    scalar_returns: np.ndarray
    vector_returns: np.ndarray
    correlation_history: list[dict]
    start_model: TabularModel
    end_model: TabularModel

class MetricLogger(Protocol):
    """
    Accept metrics emitted during training
    """

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
StepbackProvider = Callable[[], Iterable[int]]
SegmentSaver = Callable[[TrainingSegment], object]

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
    stepback_provider: StepbackProvider | None = None,
    segment_saver: SegmentSaver | None = None,
) -> TrainingResult:
    """
    Learn with epsilon-greedy Q-learning and apply feedback at checkpoints

    :param config: Validated experiment configuration
    :param logger: Training metric sink
    :param feedback_provider: Optional checkpoint feedback callback
    :param feedback_episode_provider: Optional source of future feedback episodes
    :param stepback_provider: Optional source of feedback episodes to restore
    :param segment_saver: Optional destination for completed feedback intervals
    :return: Learned model and training histories
    """
    settings = config.training
    weights = normalize_weights(settings.utility_weights)
    set_global_seed(settings.seed)
    env = make_env(config.environment)
    if not hasattr(env.action_space, "n"):
        env.close()
        raise TypeError("Tabular Q-learning requires a discrete action space")
    model = TabularModel(int(env.action_space.n), len(weights))
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
    snapshots: dict[int, TrainingSnapshot] = {}
    feedback_number = 0

    episode = 1
    try:
        while episode <= settings.num_episodes:
            observation, _ = reset_env_with_seed(env, settings.seed + episode)
            state = obs_to_state(observation)
            done = False
            scalar_return = 0.0
            vector_return = np.zeros(len(weights), dtype=np.float64)

            while not done:
                q_values = model.scalar_values(state, weights)
                objective_q_values = model.objective_values(state)
                _, p_values, visits = model.values(state)
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
                    target_objective_q = reward
                else:
                    next_state = obs_to_state(next_observation)
                    next_q = model.scalar_values(next_state, weights)
                    next_objective_q = model.objective_values(next_state)
                    _, next_p, _ = model.values(next_state)
                    greedy_next = int(np.argmax(next_q))
                    target_objective_q = (
                        reward
                        + settings.gamma_q * next_objective_q[greedy_next]
                    )
                    target_p += settings.gamma_p * next_p[greedy_next]

                objective_q_values[action] += settings.alpha_q * (
                    target_objective_q - objective_q_values[action]
                )
                q_values[action] = linear_utility(
                    objective_q_values[action],
                    weights,
                )
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

            stepback_episode = None
            restored_feedback = False
            if stepback_provider is not None:
                for requested_episode in stepback_provider():
                    if requested_episode >= episode:
                        print(
                            f"[training.py] Rejected stepback to episode {requested_episode}: "
                            f"training is at episode {episode}"
                        )
                    elif requested_episode not in snapshots:
                        available = ", ".join(str(value) for value in sorted(snapshots)) or "none"
                        print(
                            f"[training.py] Rejected stepback to episode {requested_episode}: "
                            f"available feedback episodes: {available}"
                        )
                    else:
                        stepback_episode = requested_episode

            if stepback_episode is not None:
                snapshot = snapshots[stepback_episode]
                model = deepcopy(snapshot.model)
                weights = snapshot.weights
                epsilon = snapshot.epsilon
                scalar_returns = snapshot.scalar_returns.copy()
                vector_returns = [value.copy() for value in snapshot.vector_returns]
                weight_history = deepcopy(snapshot.weight_history)
                correlation_history = deepcopy(snapshot.correlation_history)
                feedback_history = deepcopy(snapshot.feedback_history)
                feedback_number = snapshot.feedback_number
                np.random.set_state(deepcopy(snapshot.numpy_random_state))
                random.setstate(snapshot.python_random_state)
                for saved_episode in tuple(snapshots):
                    if saved_episode >= stepback_episode:
                        del snapshots[saved_episode]
                episode = stepback_episode
                restored_feedback = True
                print(f"[training.py] Stepped back to feedback episode {stepback_episode}")

            if episode in schedule and feedback_provider is not None:
                previous_feedback_episode = max(
                    (saved_episode for saved_episode in snapshots if saved_episode < episode),
                    default=None,
                )
                if (
                    segment_saver is not None
                    and previous_feedback_episode is not None
                    and not restored_feedback
                ):
                    start_snapshot = snapshots[previous_feedback_episode]
                    start_event = next(
                        event
                        for event in reversed(feedback_history)
                        if event["episode"] == previous_feedback_episode
                    )
                    segment_saver(
                        TrainingSegment(
                            start_episode=previous_feedback_episode,
                            end_episode=episode,
                            start_feedback_number=start_event["feedback_number"],
                            weights=start_event["weights_after"],
                            feedback_event=deepcopy(start_event),
                            episode_numbers=np.arange(
                                previous_feedback_episode + 1,
                                episode + 1,
                                dtype=np.int64,
                            ),
                            scalar_returns=np.asarray(
                                scalar_returns[previous_feedback_episode:episode],
                                dtype=np.float64,
                            ),
                            vector_returns=np.asarray(
                                vector_returns[previous_feedback_episode:episode],
                                dtype=np.float64,
                            ),
                            correlation_history=[
                                deepcopy(entry)
                                for entry in correlation_history
                                if previous_feedback_episode < entry["episode"] <= episode
                            ],
                            start_model=deepcopy(start_snapshot.model),
                            end_model=deepcopy(model),
                        )
                    )
                snapshots[episode] = TrainingSnapshot(
                    episode=episode,
                    feedback_number=feedback_number,
                    model=deepcopy(model),
                    weights=weights,
                    epsilon=epsilon,
                    scalar_returns=scalar_returns.copy(),
                    vector_returns=[value.copy() for value in vector_returns],
                    weight_history=deepcopy(weight_history),
                    correlation_history=deepcopy(correlation_history),
                    feedback_history=deepcopy(feedback_history),
                    numpy_random_state=deepcopy(np.random.get_state()),
                    python_random_state=random.getstate(),
                )
                feedback_number += 1
                choice = feedback_provider(
                    TrainingCheckpoint(episode, feedback_number, model, weights)
                )
                before = weights
                weights, explanation = apply_preference_update(
                    weights, choice.direction, choice.objective, choice.strength, feedback_number
                )
                model.scalarize(weights)
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
            episode += 1
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
