from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from xmorl.config import EnvironmentConfig
from xmorl.environment_adapters import environment_adapter
from xmorl.envs.factory import make_env
from xmorl.training import FeedbackChoice, TabularModel
from xmorl.utils import (
    apply_preference_update,
    linear_utility,
    obs_to_state,
    reset_env_with_seed,
    serialize_weights,
    stage_ratio_map,
)

def _format_probability(value: float) -> str:
    return f"{100 * float(value):.1f}%"

def _relative_preview_score(q_values: np.ndarray, stochasticity: float) -> np.ndarray:
    if q_values.size == 0:
        return q_values.copy()
    q_min = float(np.min(q_values))
    q_max = float(np.max(q_values))
    span = max(q_max - q_min, 1e-8)
    scores = (q_values - q_min) / span
    if np.isclose(q_max, q_min):
        scores = np.full_like(q_values, 0.5, dtype=np.float64)
    uniform = np.full_like(q_values, 1.0 / len(q_values), dtype=np.float64)
    return np.clip((1 - stochasticity) * scores + stochasticity * uniform, 0, 1)

# toggle short and longer explanation
def build_decision_explanation(
    entry: dict,
    success_probabilities: np.ndarray,
    weights: tuple[float, ...],
    action_names: dict[int, str],
    objective_names: tuple[str, ...],
    long_explanation: bool = True,
) -> str:
    action = entry["action"]
    if action is None:
        reason = entry.get("terminal_reason")
        ending = f" The episode ended because of {reason}." if reason else ""
        return "I made this decision because...\n\nThis is the final state of the replay." + ending

    action_name = action_names.get(action, f"action {action}")
    q_values = np.asarray(entry["q_values"], dtype=np.float64)
    chosen_q = float(q_values[action])
    best_q = float(np.max(q_values))
    if np.isclose(chosen_q, best_q):
        policy_reason = f"Going {action_name} had the highest learned long-term utility (Q = {chosen_q:.2f})"
    else:
        policy_reason = f"Going {action_name} had learned long-term utility Q = {chosen_q:.2f}"
    policy_reason += f" and an estimated {success_probabilities[action]:.1%} probability of success in this state."

    lines = ["I made this decision because...", "", policy_reason]

    if long_explanation:
        lines += ["", "Per-objective effect of this step:"]

        reward = entry.get("reward_vec")
        if reward is None:
            lines.append("- No immediate objective reward was produced in this state.")
        else:
            reward = np.asarray(reward, dtype=np.float64).reshape(-1)
            for index, value in enumerate(reward):
                name = objective_names[index] if index < len(objective_names) else f"objective {index + 1}"
                weight = weights[index]
                contribution = weight * float(value)
                if value > 0:
                    effect = "improves"
                elif value < 0:
                    effect = "reduces"
                else:
                    effect = "does not (immediately) change"
                lines.append(
                    f"- {name.capitalize()}: this action {effect} the objective by {abs(float(value)):.2f}, and with weight {weight:.3g}, its utility contribution is {contribution:+.3f}."
                )
    return "\n".join(lines)

def estimate_state_preview(
    environment: EnvironmentConfig,
    model: TabularModel,
    state: tuple,
    weights: tuple[float, ...],
    gamma_q: float,
    seed: int,
    samples: int,
    fallback: np.ndarray,
) -> np.ndarray:
    adapter = environment_adapter(environment.adapter)
    if not adapter.supports_state_preview:
        return fallback.copy()
    env = make_env(environment)
    q_values = np.zeros(model.num_actions, dtype=np.float64)
    try:
        for action in range(model.num_actions):
            estimates = []
            for sample in range(samples):
                reset_env_with_seed(env, seed + action * 1_000 + sample)
                adapter.set_preview_state(env, state)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                estimate = linear_utility(reward, weights)
                if not (terminated or truncated):
                    next_q, _, _ = model.values(obs_to_state(next_observation))
                    estimate += gamma_q * float(np.max(next_q))
                estimates.append(estimate)
            q_values[action] = np.mean(estimates)
        return _relative_preview_score(q_values, adapter.preview_stochasticity(env))
    finally:
        env.close()

class ReplayViewer:
    def __init__(
        self,
        title: str,
        replay: dict,
        summary: str,
        current_weights: tuple[float, ...],
        action_names: dict[int, str],
        environment: EnvironmentConfig,
        model: TabularModel,
        gamma_q: float,
        preview_samples: int,
        feedback_number: int | None = None,
        long_explanation: bool = True,
    ):
        self.entries = replay.get("entries", []) if replay else []
        self.replay_seed = replay.get("seed", 0) if replay else 0
        self.current_weights = current_weights
        self.action_names = action_names
        self.environment = environment
        self.model = model
        self.gamma_q = gamma_q
        self.preview_samples = preview_samples
        self.feedback_number = feedback_number
        self.long_explanation = long_explanation
        self.allow_feedback = feedback_number is not None
        self.index = 0
        self.result = FeedbackChoice()
        self._photo = None
        self._preview_cache: dict[tuple, np.ndarray] = {}

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1320x900")
        self.root.minsize(1000, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind_all("<Left>", self._on_left_key)
        self.root.bind_all("<Right>", self._on_right_key)
        self._build(summary)
        self._update_view()

    def _build(self, summary: str) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.image_label = ttk.Label(left, anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        right_container = ttk.Frame(main)
        right_container.grid(row=0, column=1, sticky="nsew")
        right_container.rowconfigure(0, weight=1)
        right_container.columnconfigure(0, weight=1)

        self.right_canvas = tk.Canvas(right_container, highlightthickness=0, borderwidth=0)
        self.right_canvas.grid(row=0, column=0, sticky="nsew")
        right_scrollbar = ttk.Scrollbar(
            right_container,
            orient="vertical",
            command=self.right_canvas.yview,
        )
        right_scrollbar.grid(row=0, column=1, sticky="ns")
        self.right_canvas.configure(yscrollcommand=right_scrollbar.set)

        right = ttk.Frame(self.right_canvas)
        self._right_window = self.right_canvas.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>", self._update_right_scroll_region)
        self.right_canvas.bind("<Configure>", self._resize_right_panel)
        self.root.bind_all("<MouseWheel>", self._scroll_right_panel)
        right.columnconfigure(0, weight=1)

        explanation_frame = tk.Frame(
            right,
            background="white",
            highlightbackground="black",
            highlightcolor="black",
            highlightthickness=3,
            padx=16,
            pady=14,
        )
        explanation_frame.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        explanation_frame.columnconfigure(0, weight=1)
        explanation_font = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self.explanation = tk.Label(
            explanation_frame,
            text="I made this decision because...",
            background="white",
            foreground="black",
            font=explanation_font,
            justify="left",
            anchor="nw",
            wraplength=500,
        )
        self.explanation.grid(row=0, column=0, sticky="ew")

        ttk.Label(right, text=summary, justify="left", wraplength=500).grid(row=1, column=0, sticky="w", pady=(0, 8))
        self.next_weights = ttk.Label(right, text="New weights if applied: unchanged")
        self.next_weights.grid(row=2, column=0, sticky="w", pady=(0, 8))

        navigation = ttk.LabelFrame(right, text="Replay navigation", padding=8)
        navigation.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        navigation.columnconfigure(1, weight=1)
        self.back = ttk.Button(navigation, text="← Back", command=self._back)
        self.back.grid(row=0, column=0)
        self.step_label = ttk.Label(navigation)
        self.step_label.grid(row=0, column=1)
        self.forward = ttk.Button(navigation, text="Forward →", command=self._forward)
        self.forward.grid(row=0, column=2)
        table = ttk.LabelFrame(right, text="Success preview at current state", padding=8)
        table.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        table.columnconfigure(0, weight=1)
        self.details = tk.BooleanVar(value=False)
        ttk.Checkbutton(table, text="Show value details", variable=self.details, command=self._update_columns).grid(row=0, column=0, sticky="w")
        self.tree = ttk.Treeview(table, columns=("action", "q", "now", "after", "delta"), show="headings", height=6)
        for key, label, width in (("action", "Action", 115), ("q", "Q", 80), ("now", "Now", 100), ("after", "After", 105), ("delta", "Δ", 90)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor="center")
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.tag_configure("chosen", background="#E8F2FF")

        if self.allow_feedback:
            preference = ttk.LabelFrame(right, text="Preference change", padding=8)
            preference.grid(row=5, column=0, sticky="ew")
            ratios = stage_ratio_map(self.feedback_number)
            ttk.Label(preference, text=f"Impact: small ×{ratios['small']:g}, big ×{ratios['big']:g}, huge ×{ratios['huge']:g}").grid(row=0, column=0, columnspan=3, sticky="w")
            self.direction = tk.StringVar(value="keep")
            self.objective = tk.IntVar(value=-1)
            ttk.Radiobutton(
                preference,
                text="Keep current weights",
                variable=self.direction,
                value="keep",
                command=self._preference_changed,
            ).grid(row=1, column=0, columnspan=3, sticky="w")
            objective_names = self.environment.objective_names
            for row, label in enumerate(objective_names, start=2):
                ttk.Radiobutton(
                    preference,
                    text=f"Prioritize {label}",
                    variable=self.direction,
                    value=label,
                    command=lambda selected_label=label, selected_index=row - 2: self._objective_changed(selected_label, selected_index),
                ).grid(row=row, column=0, columnspan=3, sticky="w")
            self.strength_label = ttk.Label(preference, text="Strength:")
            self.strength_label.grid(row=len(objective_names) + 2, column=0, sticky="w")
            self.strength = tk.StringVar(value="small")
            self.strength_box = ttk.Combobox(preference, textvariable=self.strength, values=("small", "big", "huge"), state="readonly", width=10)
            self.strength_box.grid(row=len(objective_names) + 2, column=1, sticky="w")
            self.strength_box.bind("<<ComboboxSelected>>", lambda _: self._preference_changed())
            ttk.Button(preference, text="Apply choice and continue training", command=self._confirm).grid(row=len(objective_names) + 3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
            self._preference_changed()
        else:
            ttk.Button(right, text="Close", command=self._close).grid(row=5, column=0, sticky="ew")
        self._update_columns()

    def _update_right_scroll_region(self, _event=None) -> None:
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    def _resize_right_panel(self, event) -> None:
        self.right_canvas.itemconfigure(self._right_window, width=event.width)

    def _scroll_right_panel(self, event) -> None:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget == self.right_canvas:
                if event.delta:
                    direction = -1 if event.delta > 0 else 1
                    self.right_canvas.yview_scroll(direction, "units")
                return
            widget = getattr(widget, "master", None)

    def _update_columns(self) -> None:
        columns = ["action"]
        if self.details.get():
            columns.append("q")
        columns.append("now")
        if self.allow_feedback:
            columns.extend(("after", "delta"))
        self.tree.configure(displaycolumns=columns)
        if self.entries:
            self._update_view()

    def _preference_changed(self) -> None:
        if self.direction.get() == "keep":
            self.strength_label.grid_remove()
            self.strength_box.grid_remove()
            self.next_weights.configure(text="New weights if applied: unchanged")
        else:
            self.strength_label.grid()
            self.strength_box.grid()
            weights, _ = apply_preference_update(self.current_weights, self.direction.get(), self.objective.get(), self.strength.get(), self.feedback_number)
            self.next_weights.configure(text=f"New weights if applied: {serialize_weights(weights)}")
        if self.entries:
            self._update_view()

    def _objective_changed(self, label: str, index: int) -> None:
        self.direction.set(label)
        self.objective.set(index)
        self._preference_changed()

    def _preview(self, entry: dict, weights: tuple[float, ...]) -> np.ndarray:
        key = (self.index, weights)
        if key not in self._preview_cache:
            self._preview_cache[key] = estimate_state_preview(
                self.environment, self.model, entry["state"], weights, self.gamma_q,
                self.replay_seed * 1_000 + self.index * 100 + 77,
                self.preview_samples, entry["p_values"],
            )
        return self._preview_cache[key]

    def _update_view(self) -> None:
        if not self.entries:
            self.image_label.configure(text="No successful replay is available.")
            self.back.state(["disabled"])
            self.forward.state(["disabled"])
            return
        entry = self.entries[self.index]
        image = Image.fromarray(entry["frame"])
        scale = min(1.0, 760 / image.width, 700 / image.height)
        if scale < 1:
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.NEAREST)
        self._photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self._photo, text="")
        self.step_label.configure(text=f"Step {self.index + 1} / {len(self.entries)}")
        self.back.state(["disabled"] if self.index == 0 else ["!disabled"])
        self.forward.state(["disabled"] if self.index == len(self.entries) - 1 else ["!disabled"])
        action = entry["action"]
        current = self._preview(entry, self.current_weights)
        after = current
        if self.allow_feedback and self.direction.get() != "keep":
            new_weights, _ = apply_preference_update(self.current_weights, self.direction.get(), self.objective.get(), self.strength.get(), self.feedback_number)
            after = self._preview(entry, new_weights)
        self.explanation.configure(
            text=build_decision_explanation(
                entry,
                current,
                self.current_weights,
                self.action_names,
                self.environment.objective_names,
                self.long_explanation,
            )
        )
        for item in self.tree.get_children():
            self.tree.delete(item)
        for index, q_value in enumerate(entry["q_values"]):
            label = self.action_names.get(index, str(index))
            tag = ()
            if action == index:
                label = f"➜ {label}"
                tag = ("chosen",)
            delta = 100 * (after[index] - current[index])
            self.tree.insert("", "end", values=(label, f"{q_value:.4f}", _format_probability(current[index]), _format_probability(after[index]), f"{delta:+.1f} pp"), tags=tag)

    def _back(self) -> None:
        self.index = max(0, self.index - 1)
        self._update_view()

    def _forward(self) -> None:
        self.index = min(len(self.entries) - 1, self.index + 1)
        self._update_view()

    def _on_left_key(self, _event) -> str:
        self._back()
        return "break"

    def _on_right_key(self, _event) -> str:
        self._forward()
        return "break"

    def _confirm(self) -> None:
        self.result = FeedbackChoice(self.direction.get(), self.objective.get(), self.strength.get())
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Left>")
        self.root.unbind_all("<Right>")
        self.root.destroy()

    def _close(self) -> None:
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Left>")
        self.root.unbind_all("<Right>")
        self.root.destroy()

    def run(self) -> FeedbackChoice:
        self.root.mainloop()
        return self.result