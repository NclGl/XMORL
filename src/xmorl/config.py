from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

@dataclass(frozen=True)
class EnvironmentConfig:
    id: str = "stochastic-deep-sea-treasure-v0"
    kwargs: dict[str, Any] = field(default_factory=dict)
    action_names: dict[int, str] = field(
        default_factory=lambda: {0: "up", 1: "down", 2: "left", 3: "right"}
    )
    objective_names: tuple[str, ...] = ("treasure", "time")

@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    num_episodes: int = 500_000
    utility_weights: tuple[float, ...] = (0.5, 0.5)
    alpha_q: float = 0.3
    gamma_q: float = 0.99
    alpha_p: float = 0.1
    gamma_p: float = 1.0
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.99999
    progress_every: int = 25_000

@dataclass(frozen=True)
class FeedbackConfig:
    enabled: bool = True
    num_moments: int = 4
    episodes: tuple[int, ...] | None = None
    preview_samples: int = 24
    render_without_explosions: bool = True

@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_episodes: int = 100
    checkpoint_seed_start: int = 50_000
    final_episodes: int = 50
    final_seed_start: int = 10_000
    max_steps: int = 200

@dataclass(frozen=True)
class LoggingConfig:
    output_dir: str = "runs"
    run_name: str | None = None
    legacy_files: bool = True
    wandb_enabled: bool = True
    wandb_mode: str = "offline"
    wandb_project: str = "xmorl"
    wandb_entity: str | None = None
    log_every: int = 1_000

@dataclass(frozen=True)
class PlottingConfig:
    enabled: bool = True
    show: bool = True
    moving_average_window: int = 50

@dataclass(frozen=True)
class AppConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return dict(value)

def _reject_unknown(section: str, values: Mapping[str, Any], cls: type) -> None:
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown option(s) in '{section}': {names}")

def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The YAML document must contain a mapping at its root")

    known_sections = {"environment", "training", "feedback", "evaluation", "logging", "plotting"}
    unknown_sections = set(raw) - known_sections
    if unknown_sections:
        raise ValueError(f"Unknown configuration section(s): {', '.join(sorted(unknown_sections))}")

    env = _section(raw, "environment")
    training = _section(raw, "training")
    feedback = _section(raw, "feedback")
    evaluation = _section(raw, "evaluation")
    logging = _section(raw, "logging")
    plotting = _section(raw, "plotting")
    for name, values, cls in (
        ("environment", env, EnvironmentConfig),
        ("training", training, TrainingConfig),
        ("feedback", feedback, FeedbackConfig),
        ("evaluation", evaluation, EvaluationConfig),
        ("logging", logging, LoggingConfig),
        ("plotting", plotting, PlottingConfig),
    ):
        _reject_unknown(name, values, cls)

    if "action_names" in env:
        env["action_names"] = {int(key): str(value) for key, value in env["action_names"].items()}
    if "objective_names" in env:
        env["objective_names"] = tuple(str(value) for value in env["objective_names"])
    if "utility_weights" in training:
        training["utility_weights"] = tuple(float(value) for value in training["utility_weights"])
    if "episodes" in feedback and feedback["episodes"] is not None:
        feedback["episodes"] = tuple(int(value) for value in feedback["episodes"])

    config = AppConfig(
        environment=EnvironmentConfig(**env),
        training=TrainingConfig(**training),
        feedback=FeedbackConfig(**feedback),
        evaluation=EvaluationConfig(**evaluation),
        logging=LoggingConfig(**logging),
        plotting=PlottingConfig(**plotting),
    )
    validate_config(config)
    return config

def validate_config(config: AppConfig) -> None:
    if config.training.num_episodes <= 0:
        raise ValueError("training.num_episodes must be positive")
    if not config.training.utility_weights:
        raise ValueError("training.utility_weights cannot be empty")
    if len(config.environment.objective_names) != len(config.training.utility_weights):
        raise ValueError("environment.objective_names must match the number of utility weights")
    if not 0 <= config.training.epsilon_min <= config.training.epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= epsilon_min <= epsilon_start <= 1")
    if not 0 < config.training.epsilon_decay <= 1:
        raise ValueError("training.epsilon_decay must be in (0, 1]")
    if config.feedback.episodes is not None:
        invalid = [ep for ep in config.feedback.episodes if ep < 1 or ep > config.training.num_episodes]
        if invalid:
            raise ValueError(f"feedback.episodes outside the training range: {invalid}")
    if config.logging.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("logging.wandb_mode must be online, offline, or disabled")
    if config.logging.log_every <= 0:
        raise ValueError("logging.log_every must be positive")
    if config.feedback.preview_samples <= 0:
        raise ValueError("feedback.preview_samples must be positive")
    if config.evaluation.checkpoint_episodes <= 0 or config.evaluation.final_episodes <= 0:
        raise ValueError("evaluation episode counts must be positive")
    if config.evaluation.max_steps <= 0:
        raise ValueError("evaluation.max_steps must be positive")
