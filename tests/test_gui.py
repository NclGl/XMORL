import numpy as np

from xmorl.gui import build_decision_explanation

def test_decision_explanation_contains_policy_and_objective_attribution():
    entry = {
        "action": 3,
        "q_values": np.array([1.0, 2.0, 0.5, 4.0]),
        "reward_vec": np.array([0.0, -1.0]),
        "terminal_reason": None,
    }

    explanation = build_decision_explanation(
        entry,
        success_probabilities=np.array([0.1, 0.2, 0.05, 0.8]),
        weights=(1.0, 0.5),
        action_names={0: "up", 1: "down", 2: "left", 3: "right"},
        objective_names=("treasure", "time"),
        long_explanation=True
    )

    assert explanation.startswith("I made this decision because...")
    assert "Going right had the highest learned long-term utility" in explanation
    assert "80.0% probability of success" in explanation
    assert "Treasure:" in explanation
    assert "Time:" in explanation
    assert "-0.500" in explanation

def test_final_state_explanation_mentions_terminal_reason():
    explanation = build_decision_explanation(
        {"action": None, "terminal_reason": "treasure"},
        np.zeros(4),
        (1.0, 1.0),
        {},
        ("treasure", "time"),
    )
    assert "final state" in explanation
    assert "treasure" in explanation
