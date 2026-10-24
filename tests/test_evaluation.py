import numpy as np

from moba_rl.evaluation.evaluate import evaluate


class FixedActionModel:
    def __init__(self, action):
        self.action = action

    def predict(self, observation, action_masks, deterministic):
        assert action_masks[self.action]
        return np.asarray(self.action), None


def test_evaluation_reports_advanced_action_and_match_metrics():
    result = evaluate(
        model=FixedActionModel(27),
        opponent_name="balanced",
        episodes=1,
        seed=7,
        hierarchical_actions=True,
        advanced_decisions=True,
        fog_of_war=True,
        ticks_per_decision=1,
        max_steps=2,
    )

    assert result["episodes"] == 1
    assert result["blue_action_diversity"] == 1
    assert result["blue_advanced_action_rate"] == 1.0
    assert result["blue_action_counts"] == '{"27":2}'
    assert result["mean_blue_wave_orders"] == 1.0
    assert result["mean_blue_objectives"] >= 0.0
