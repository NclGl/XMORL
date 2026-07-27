from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

@dataclass
class SuccessConfig:
    """
    Describe how an environment outcome is classified as successful
    """

    type: str = "auto"
    key: str = "success"
    indices: tuple[int, ...] = ()
    condition: str = "any_positive"
    threshold: float = 0.0
    values: tuple[str, ...] = ()
    require_terminated: bool = True

@dataclass
class EnvironmentConfig:
    """
    Identify an environment and the names XMORL uses to explain it
    """

    id: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    action_names: dict[int, str] = field(default_factory=dict)
    objective_names: tuple[str, ...] = ("objective 1", "objective 2")
    adapter: str = "default"
    success: SuccessConfig = field(default_factory=SuccessConfig)

@dataclass
class TrainingConfig:
    """
    Configure tabular learning, exploration, and utility discounting
    """

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

@dataclass
class FeedbackConfig:
    """
    Configure when and how human preference feedback is collected
    """

    enabled: bool = True
    num_moments: int = 4
    episodes: tuple[int, ...] | None = None
    preview_samples: int = 24
    render_without_explosions: bool = True

@dataclass
class EvaluationConfig:
    """
    Configure the fixed-seed checkpoint and final evaluation batches
    """

    checkpoint_episodes: int = 100
    checkpoint_seed_start: int = 50_000
    final_episodes: int = 50
    final_seed_start: int = 10_000
    max_steps: int = 200

@dataclass
class LoggingConfig:
    """
    Configure local run artifacts and optional Weights & Biases logging
    """

    output_dir: str = "runs"
    run_name: str | None = None
    legacy_files: bool = True
    wandb_enabled: bool = True
    wandb_mode: str = "offline"
    wandb_project: str = "xmorl"
    wandb_entity: str | None = None
    log_every: int = 1_000

@dataclass
class PlottingConfig:
    """
    Configure learning-curve generation and display
    """

    enabled: bool = True
    show: bool = True
    moving_average_window: int = 50

@dataclass
class GUIConfig:
    """
    Configure gui details
    """

    enabled: bool = True
    long_explanation: bool = True
    trajectory_line: bool = True

@dataclass
class AppConfig:
    """
    Collect every experiment setting into one validated object
    """

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)

    def to_dict(self) -> dict[str, Any]:
        """
        Return a serialization-ready view of the full configuration
        """
        return asdict(self)

def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    """
    Read one YAML section
    """
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return dict(value)

def _reject_unknown(section: str, values: Mapping[str, Any], cls: type) -> None:
    """
    Reject configuration keys that are not fields of the target dataclass
    """
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown option(s) in '{section}': {names}")

def load_config(path: str | Path) -> AppConfig:
    """
    Load YAML, normalize collection types, and return a validated configuration
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The YAML document must contain a mapping at its root")

    known_sections = {"environment", "training", "feedback", "evaluation", "logging", "plotting", "gui"}
    unknown_sections = set(raw) - known_sections
    if unknown_sections:
        raise ValueError(f"Unknown configuration section(s): {', '.join(sorted(unknown_sections))}")

    env = _section(raw, "environment")
    training = _section(raw, "training")
    feedback = _section(raw, "feedback")
    evaluation = _section(raw, "evaluation")
    logging = _section(raw, "logging")
    plotting = _section(raw, "plotting")
    gui = _section(raw, "gui")
    long_explanation = _section(raw, "long_explanation")

    for name, values, cls in (
        ("environment", env, EnvironmentConfig),
        ("training", training, TrainingConfig),
        ("feedback", feedback, FeedbackConfig),
        ("evaluation", evaluation, EvaluationConfig),
        ("logging", logging, LoggingConfig),
        ("plotting", plotting, PlottingConfig),
        ("gui", gui, GUIConfig),
    ):
        _reject_unknown(name, values, cls)

    if "success" in env:
        success = env["success"]
        if not isinstance(success, Mapping):
            raise ValueError("Configuration section 'environment.success' must be a mapping")
        success = dict(success)
        _reject_unknown("environment.success", success, SuccessConfig)
        if "indices" in success:
            success["indices"] = tuple(int(value) for value in success["indices"])
        if "values" in success:
            success["values"] = tuple(str(value) for value in success["values"])
        if "threshold" in success:
            success["threshold"] = float(success["threshold"])
        env["success"] = SuccessConfig(**success)
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
        gui=GUIConfig(**gui),
    )
    validate_config(config)
    return config

def validate_config(config: AppConfig) -> None:
    """
    Reject settings that would make training or interpretation inconsistent
    """
    from xmorl.environment_adapters import environment_adapter_names

    if config.environment.adapter not in environment_adapter_names():
        available = ", ".join(sorted(environment_adapter_names()))
        raise ValueError(f"environment.adapter must be one of: {available}")
    success = config.environment.success
    if success.type not in {"auto", "info", "reward", "terminal_reason"}:
        raise ValueError("environment.success.type must be auto, info, reward, or terminal_reason")
    if success.condition not in {"any_positive", "all_positive", "sum_positive"}:
        raise ValueError("environment.success.condition must be any_positive, all_positive, or sum_positive")
    if success.type == "info" and not success.key:
        raise ValueError("environment.success.key cannot be empty for an info rule")
    if success.type == "reward":
        if not success.indices:
            raise ValueError("environment.success.indices cannot be empty for a reward rule")
        invalid_indices = [index for index in success.indices if index < 0 or index >= len(config.environment.objective_names)]
        if invalid_indices:
            raise ValueError(f"environment.success.indices outside the objective range: {invalid_indices}")
    if success.type == "terminal_reason" and not success.values:
        raise ValueError("environment.success.values cannot be empty for a terminal_reason rule")
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
