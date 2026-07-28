from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    Smooth a sequence with a valid window arithmetic mean

    :param values: Sequence to smooth
    :param window: Moving-average window length
    :return: Smoothed sequence
    """
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) < window:
        return values.copy()
    return np.convolve(values, np.ones(window) / window, mode="valid")

def plot_learning(result, path: str | Path, window: int = 50, show: bool = True) -> Path:
    """
    Save learning curves and mark episodes where preferences changed

    :param result: Completed training result
    :param path: Output image path
    :param window: Moving-average window length
    :param show: Whether to display the figure interactively
    :return: Saved image path
    """
    path = Path(path)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    episodes = np.arange(1, len(result.scalar_returns) + 1)
    axes[0].plot(episodes, result.scalar_returns, color="#7194B7", alpha=0.25, label="episode")
    smoothed = moving_average(result.scalar_returns, window)
    smooth_x = np.arange(len(smoothed)) + (window if len(result.scalar_returns) >= window else 1)
    axes[0].plot(smooth_x, smoothed, color="#1F4E79", label=f"moving average ({window})")
    axes[0].set_ylabel("Utility return")
    axes[0].legend()

    colors = ("#2E8B57", "#B22222", "#6A5ACD", "#C0504D")
    for objective in range(result.vector_returns.shape[1]):
        values = result.vector_returns[:, objective]
        smoothed = moving_average(values, window)
        smooth_x = np.arange(len(smoothed)) + (window if len(values) >= window else 1)
        axes[1].plot(smooth_x, smoothed, color=colors[objective % len(colors)], label=f"objective {objective}")
    for item in result.weight_history[1:]:
        for axis in axes:
            axis.axvline(item["episode"], color="#666666", alpha=0.4, linestyle="--")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Vector return")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return path
