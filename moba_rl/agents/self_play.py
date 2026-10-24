from __future__ import annotations

import numpy as np

from moba_rl.simulation.lini_tities import Team


class SelfPlayOpponent:
    """Use the current MaskablePPO policy as the RED-side opponent."""

    def __init__(self, model, deterministic: bool = False):
        self.model = model
        self.deterministic = deterministic

    def act_for_env(self, env) -> None:
        observation = env._get_observation(Team.RED)
        action_mask = env.action_masks(Team.RED)
        action, _ = self.model.predict(
            observation,
            action_masks=action_mask,
            deterministic=self.deterministic,
        )
        action = int(np.asarray(action).item())

        # Policy observations use a mirrored, forward-facing coordinate frame.
        # Reverse horizontal movement when applying RED's canonical action.
        if action == 1:
            action = 2
        elif action == 2:
            action = 1

        env._execute_opponent_action(action)
        return int(np.asarray(action).item())
