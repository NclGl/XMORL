from __future__ import annotations

import csv
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any
import numpy as np
import wandb

from xmorl.config import AppConfig

if TYPE_CHECKING:
    from xmorl.training import TrainingResult, TrainingSegment

class _Encoder(json.JSONEncoder):
    """
    Convert scientific values into representations compatible with JSON
    """

    def default(self, value: Any) -> Any:
        """
        Encode NumPy values and paths before using the standard fallback

        :param value: Value to encode
        :return: JSON-compatible value
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return super().default(value)

class RunLogger:
    """
    Own one run directory and its optional Weights & Biases session
    """

    def __init__(self, config: AppConfig):
        """
        Create storage and initialize configured logging backends

        :param config: Experiment and logging configuration
        :return: None
        """
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = config.logging.run_name or f"run-{timestamp}"
        self.run_dir = Path(config.logging.output_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        matplotlib_dir = self.run_dir.parent / ".matplotlib"
        matplotlib_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir.resolve()))
        self._wandb_run = None
        self._feedback_rows: list[dict] = []

        if config.logging.legacy_files:
            self.write_json("run_config.json", config.to_dict())

        if config.logging.wandb_enabled and config.logging.wandb_mode != "disabled":
            wandb_state_dir = self.run_dir.parent / ".wandb-state"
            wandb_cache_dir = wandb_state_dir / "cache"
            wandb_config_dir = wandb_state_dir / "config"
            wandb_data_dir = wandb_state_dir / "data"
            for directory in (wandb_cache_dir, wandb_config_dir, wandb_data_dir):
                directory.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_cache_dir.resolve()))
            os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_config_dir.resolve()))
            os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data_dir.resolve()))
            
            self._wandb_run = wandb.init(
                project=config.logging.wandb_project,
                entity=config.logging.wandb_entity,
                name=self.run_name,
                mode=config.logging.wandb_mode,
                config=config.to_dict(),
                dir=str(self.run_dir),
            )

    def log(self, metrics: dict, step: int | None = None) -> None:
        """
        Send metrics to the active Weights & Biases run

        :param metrics: Metric names and values
        :param step: Optional training step
        :return: None
        """
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def log_feedback(self, event: dict) -> None:
        """
        Buffer a feedback event for callers that use incremental collection

        :param event: Feedback event values
        :return: None
        """
        self._feedback_rows.append(event)

    def write_json(self, filename: str, payload: Any) -> Path:
        """
        Write a JSON artifact inside the current run directory

        :param filename: Artifact filename
        :param payload: JSON-serializable content
        :return: Written artifact path
        """
        path = self.run_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, cls=_Encoder, indent=2)
        return path

    def write_csv(self, filename: str, rows: list[dict]) -> Path | None:
        """
        Write tabular rows when at least one row is available

        :param filename: Artifact filename
        :param rows: Records to write
        :return: Written path or None when rows are empty
        """
        if not rows:
            return None
        path = self.run_dir / filename
        fieldnames = list(rows[0])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def save_results(self, result: TrainingResult, evaluation_summary: dict) -> None:
        """
        Save legacy feedback, evaluation, and learning artifacts when enabled

        :param result: Completed training result
        :param evaluation_summary: Final evaluation metrics
        :return: None
        """
        if self.config.logging.legacy_files:
            self.write_json("feedback_history.json", result.feedback_history)
            self.write_csv("feedback_log.csv", result.feedback_history)
            self.write_json("evaluation_summary.json", evaluation_summary)
            learning_rows = [
                {
                    "episode": index + 1,
                    "return_scalar": scalar,
                    **{f"return_objective_{j}": value for j, value in enumerate(vector)},
                }
                for index, (scalar, vector) in enumerate(zip(result.scalar_returns, result.vector_returns))
            ]
            self.write_csv("learning_returns.csv", learning_rows)

    def save_training_segment(self, segment: TrainingSegment) -> Path:
        """
        Save a versioned completed interval for comparison after retraining

        :param segment: Completed feedback-to-feedback training interval
        :return: Directory containing the saved interval
        """
        segment_root = (
            self.run_dir
            / "training_segments"
            / f"episodes_{segment.start_episode}_{segment.end_episode}"
        )
        segment_root.mkdir(parents=True, exist_ok=True)
        versions = [
            int(path.name.removeprefix("version_"))
            for path in segment_root.glob("version_*")
            if path.is_dir() and path.name.removeprefix("version_").isdigit()
        ]
        version = max(versions, default=0) + 1
        version_dir = segment_root / f"version_{version:03d}"
        version_dir.mkdir()

        metadata = {
            "version": version,
            "start_episode": segment.start_episode,
            "end_episode": segment.end_episode,
            "start_feedback_number": segment.start_feedback_number,
            "weights": segment.weights,
            "feedback_event": segment.feedback_event,
            "correlation_history": segment.correlation_history,
        }
        with (version_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, cls=_Encoder, indent=2)
        np.savez_compressed(
            version_dir / "returns.npz",
            episodes=segment.episode_numbers,
            scalar_returns=segment.scalar_returns,
            vector_returns=segment.vector_returns,
        )
        with (version_dir / "start_model.pkl").open("wb") as handle:
            pickle.dump(segment.start_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (version_dir / "end_model.pkl").open("wb") as handle:
            pickle.dump(segment.end_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return version_dir

    def finish(self) -> None:
        """
        Finish the active external logging session

        :return: None
        """
        if self._wandb_run is not None:
            self._wandb_run.finish()

    def __enter__(self) -> RunLogger:
        """
        Return this logger as a managed experiment resource

        :return: Active logger
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Close logging when the experiment context exits

        :param exc_type: Raised exception type
        :param exc_value: Raised exception instance
        :param traceback: Raised exception traceback
        :return: None
        """
        self.finish()
