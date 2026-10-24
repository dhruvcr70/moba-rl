import numpy as np

from moba_rl.environment import MOBAEnv
from moba_rl.simulation.lini_tities import Team, Position, Minion


def test_environment_reset():
    env = MOBAEnv(seed=42)

    observation, info = env.reset()

    assert observation.shape == (33,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["steps"] == 0
    assert info["time"] == 0.0


def test_environment_step():
    env = MOBAEnv(seed=42)

    observation, _ = env.reset()

    next_observation, reward, terminated, truncated, info = (
        env.step(0)
    )

    assert next_observation.shape == (33,)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert "steps" in info
    assert "time" in info


def test_all_actions_are_valid():
    env = MOBAEnv(seed=42)

    env.reset()

    for action in range(env.action_space.n):
        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert env.observation_space.contains(observation)

        if terminated or truncated:
            env.reset()


def test_action_mask_has_correct_shape():
    env = MOBAEnv(seed=42)

    env.reset()

    mask = env.action_mask()

    assert mask.shape == (env.action_space.n,)

    assert np.all(
        (mask == 0) | (mask == 1)
    )


def test_environment_can_run_multiple_steps():
    env = MOBAEnv(
        seed=42,
        max_steps=100,
    )

    observation, _ = env.reset()

    for _ in range(100):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert env.observation_space.contains(observation)

        if terminated or truncated:
            break


def test_environment_resets_after_episode():
    env = MOBAEnv(
        seed=42,
        max_steps=5,
    )

    env.reset()

    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(0)

        if terminated or truncated:
            break

    observation, info = env.reset()

    assert observation.shape == (33,)
    assert env.steps == 0
    assert info["steps"] == 0
    assert info["time"] == 0.0


def test_agent_can_attack_enemy_base_after_tower_is_destroyed():
    env = MOBAEnv(seed=42)

    observation, _ = env.reset()

    # The enemy base cannot be attacked while its tower is alive.
    assert env.action_masks()[11] is np.False_

    # Destroy the enemy tower manually.
    env.game.towers[Team.RED].hp = 0.0

    # Place the BLUE champion directly on the RED base.
    env.game.champions[Team.BLUE].position = Position(
        env.game.bases[Team.RED].position.x,
        env.game.bases[Team.RED].position.y,
    )

    env.game.champions[Team.BLUE].attack_timer = 0.0

    mask = env.action_masks()

    # Action 11 = attack enemy base.
    assert mask[11] is np.True_

    old_hp = env.game.bases[Team.RED].hp

    observation, reward, terminated, truncated, info = (
        env.step(11)
    )

    assert env.game.bases[Team.RED].hp < old_hp
    assert observation.shape == (33,)
    assert env.observation_space.contains(observation)

def test_environment_can_complete_blue_base_destruction():
    env = MOBAEnv(seed=42, max_steps=500)

    env.reset()

    # Remove the RED tower so the BLUE champion can damage the base.
    env.game.towers[Team.RED].hp = 0.0

    # Put BLUE directly on the RED base.
    env.game.champions[Team.BLUE].position = Position(
        env.game.bases[Team.RED].position.x,
        env.game.bases[Team.RED].position.y,
    )

    # Make attacks happen every step for a deterministic test.
    env.game.champions[Team.BLUE].attack_cooldown = 0.0
    env.game.champions[Team.BLUE].attack_timer = 0.0

    terminated = False

    for _ in range(200):
        observation, reward, terminated, truncated, info = env.step(11)

        assert env.observation_space.contains(observation)

        if terminated or truncated:
            break

    assert terminated is True
    assert truncated is False
    assert env.game.game_over is True
    assert env.game.winner == Team.BLUE
    assert env.game.bases[Team.RED].hp <= 0.0
    assert info["winner"] == "blue"

def test_scripted_opponents_can_step_environment():
    from moba_rl.agents.scripted import (
        AggressiveOpponent,
        DefensiveOpponent,
        ObjectiveOpponent,
        BalancedOpponent,
    )

    opponents = [
        AggressiveOpponent(),
        DefensiveOpponent(),
        ObjectiveOpponent(),
        BalancedOpponent(),
    ]

    for opponent in opponents:
        env = MOBAEnv(
            opponent=opponent,
            seed=42,
            max_steps=100,
        )

        try:
            observation, _ = env.reset()

            for _ in range(100):
                observation, reward, terminated, truncated, info = (
                    env.step(0)
                )

                assert env.observation_space.contains(observation)

                if terminated or truncated:
                    break
        finally:
            env.close()


def test_progress_reward_rewards_movement_toward_enemy_tower():
    env = MOBAEnv(seed=42, progress_reward_scale=5.0)
    env.reset()

    _, reward, _, _, _ = env.step(2)  # Move right, toward RED tower.

    map_diagonal = np.hypot(
        env.game.map.config.width,
        env.game.map.config.height,
    )
    expected_progress_reward = 5.0 * 10.0 / map_diagonal

    # Includes the existing 0.001 survival reward.
    assert np.isclose(reward, 0.001 + expected_progress_reward)


def test_progress_reward_targets_enemy_base_after_tower_falls():
    env = MOBAEnv(seed=42, progress_reward_scale=5.0)
    env.reset()
    env.game.towers[Team.RED].hp = 0.0
    env.game.champions[Team.BLUE].position = Position(1600.0, 300.0)

    _, reward, _, _, _ = env.step(2)  # Move toward RED base at x=1900.

    map_diagonal = np.hypot(
        env.game.map.config.width,
        env.game.map.config.height,
    )
    expected_progress_reward = 5.0 * 10.0 / map_diagonal

    assert np.isclose(reward, 0.001 + expected_progress_reward)


def test_objective_then_lane_progress_target_changes_after_capture():
    env = MOBAEnv(
        seed=42,
        progress_reward_scale=5.0,
        progress_target_mode="objective_then_lane",
    )
    env.reset()
    env.game.champions[Team.BLUE].position = Position(1400.0, 300.0)

    # While the objective is alive, reward follows actual distance reduction.
    agent = env.game.champions[Team.BLUE]
    target = env._get_progress_target()
    distance_before = agent.position.distance_to(target)
    _, reward, _, _, _ = env.step(1)
    map_diagonal = np.hypot(
        env.game.map.config.width,
        env.game.map.config.height,
    )
    expected_progress_reward = 5.0 * (
        distance_before - agent.position.distance_to(target)
    ) / map_diagonal
    assert np.isclose(reward, 0.001 + expected_progress_reward)

    # After the objective is taken, right moves toward the enemy tower.
    env.game.objective.hp = 0.0
    env.game.champions[Team.BLUE].position = Position(1400.0, 300.0)
    target = env._get_progress_target()
    distance_before = agent.position.distance_to(target)
    _, reward, _, _, _ = env.step(2)
    expected_progress_reward = 5.0 * (
        distance_before - agent.position.distance_to(target)
    ) / map_diagonal
    assert np.isclose(reward, 0.001 + expected_progress_reward)


def test_damage_reward_applies_only_to_direct_damage_dealt():
    scale = 0.002
    env = MOBAEnv(seed=42, damage_reward_scale=scale)
    env.reset()
    blue = env.game.champions[Team.BLUE]
    red = env.game.champions[Team.RED]
    blue.position = Position(red.position.x, red.position.y)
    blue.attack_timer = 0.0

    _, reward, _, _, _ = env.step(5)  # Attack the RED champion.

    assert np.isclose(reward, 0.001 + scale * blue.attack_damage)


def test_damage_reward_is_zero_when_agent_deals_no_damage():
    env = MOBAEnv(seed=42, damage_reward_scale=0.002)
    env.reset()

    _, reward, _, _, _ = env.step(0)

    # The RED champion moves toward BLUE, but the damage bonus only tracks
    # health lost immediately from the BLUE agent's selected action.
    assert np.isclose(reward, 0.001)


def test_structure_loss_penalty_applies_to_blue_tower_damage():
    scale = 0.001
    env = MOBAEnv(seed=42, structure_loss_penalty_scale=scale)
    env.reset()
    blue_tower = env.game.towers[Team.BLUE]

    def damage_tower(_dt):
        blue_tower.take_damage(100.0)

    env.game.step = damage_tower

    _, reward, _, _, _ = env.step(0)

    assert np.isclose(reward, 0.001 - scale * 100.0)


def test_unseeded_resets_advance_reproducible_episode_seeds():
    first_env = MOBAEnv(seed=42)
    second_env = MOBAEnv(seed=42)

    first_env.reset()
    first_seed_1 = first_env.game.seed
    first_env.reset()
    first_seed_2 = first_env.game.seed

    second_env.reset()
    second_seed_1 = second_env.game.seed
    second_env.reset()
    second_seed_2 = second_env.game.seed

    assert first_seed_1 != first_seed_2
    assert (first_seed_1, first_seed_2) == (second_seed_1, second_seed_2)


def test_explicit_reset_seed_reproduces_episode_seed():
    env = MOBAEnv(seed=42)

    env.reset(seed=1234)
    first_game_seed = env.game.seed
    env.reset(seed=1234)

    assert env.game.seed == first_game_seed


def test_fog_of_war_hides_unseen_enemy_and_keeps_last_known_state():
    env = MOBAEnv(seed=42, fog_of_war=True)
    observation, _ = env.reset()

    assert observation.shape == (37,)
    assert env.observation_space.contains(observation)
    assert np.all(observation[8:13] == 0.0)
    assert observation[32] == 0.0
    assert np.all(observation[33:37] == 0.0)

    enemy = env.game.champions[Team.RED]
    enemy.position = Position(200.0, 300.0)
    visible_observation = env._get_observation()
    assert visible_observation[33] == 1.0
    assert visible_observation[8] > 0.0
    last_seen_x = visible_observation[8]

    enemy.position = Position(1800.0, 300.0)
    hidden_observation = env._get_observation()
    assert hidden_observation[33] == 0.0
    assert hidden_observation[8] == last_seen_x


def test_fog_of_war_action_mask_restricts_unseen_targets():
    env = MOBAEnv(seed=42, fog_of_war=True)
    env.reset()

    mask = env.action_masks()

    assert mask.shape == (12,)
    assert mask[0]
    assert mask[1:5].all()
    assert not mask[5]
    assert not mask[6]
    assert not mask[10]


def test_red_policy_observation_uses_mirrored_perspective():
    env = MOBAEnv(seed=42)
    blue_observation, _ = env.reset()
    red_observation = env._get_observation(Team.RED)

    assert red_observation.shape == blue_observation.shape
    assert np.allclose(red_observation, blue_observation)
    assert env.action_masks(Team.RED).shape == (env.action_space.n,)


def test_self_play_opponent_applies_mirrored_forward_action():
    from moba_rl.agents.self_play import SelfPlayOpponent

    class FixedPolicy:
        def predict(self, observation, action_masks, deterministic):
            assert action_masks[2]
            return np.asarray(2), None  # Canonical move toward the enemy.

    env = MOBAEnv(seed=42, opponent=SelfPlayOpponent(FixedPolicy()))
    env.reset()
    red_start_x = env.game.champions[Team.RED].position.x

    env.step(0)

    assert env.game.champions[Team.RED].position.x < red_start_x


def test_curriculum_stages_expand_opponent_pool():
    from moba_rl.training.train_ppo import (
        OPPONENT_NAMES,
        RotatingOpponentWrapper,
    )

    env = RotatingOpponentWrapper(
        MOBAEnv(seed=42),
        [*OPPONENT_NAMES, "self_play"],
        curriculum=True,
    )

    env.set_training_progress(0.0)
    assert env.eligible_opponents() == ["aggressive"]

    env.set_training_progress(0.25)
    assert env.eligible_opponents() == ["aggressive", "defensive"]

    env.set_training_progress(0.50)
    assert env.eligible_opponents() == [
        "aggressive", "defensive", "objective"
    ]

    env.set_training_progress(0.70)
    assert env.eligible_opponents() == OPPONENT_NAMES

    env.set_training_progress(0.90)
    assert env.eligible_opponents() == [*OPPONENT_NAMES, "self_play"]


def test_strategic_actions_expose_farming_and_shop_controls():
    env = MOBAEnv(seed=42, strategic_actions=True)
    observation, _ = env.reset()

    assert observation.shape == (39,)
    assert env.observation_space.contains(observation)
    assert env.action_space.n == 17
    assert env.action_masks().shape == (17,)
    assert env.action_masks()[13]  # Buy a blade at base with starting gold.

    blue = env.game.champions[Team.BLUE]
    starting_gold = blue.gold
    _, _, _, _, _ = env.step(13)
    assert blue.gold == starting_gold - 200
    assert blue.attack_upgrade_level == 1


def test_strategic_farming_action_can_last_hit_minion():
    env = MOBAEnv(seed=42, strategic_actions=True)
    env.reset()
    blue = env.game.champions[Team.BLUE]
    target = Minion(
        id=9999,
        team=Team.RED,
        position=Position(blue.position.x + 50.0, blue.position.y),
        max_hp=300.0,
        hp=25.0,
    )
    env.game.minions.append(target)
    blue.attack_timer = 0.0

    assert env.action_masks()[12]
    gold_before = blue.gold
    env.step(12)

    assert not target.alive
    assert blue.gold == gold_before + env.game.minion_kill_gold


def test_strategic_and_fog_modes_have_consistent_observation_shape():
    env = MOBAEnv(seed=42, strategic_actions=True, fog_of_war=True)
    observation, _ = env.reset()

    assert observation.shape == (43,)
    assert env.observation_space.contains(observation)
    assert env.action_space.n == 17
    assert env.action_masks().shape == (17,)


def test_pygame_rgb_render_returns_frame(monkeypatch):
    import importlib.util
    import pytest

    if importlib.util.find_spec("pygame") is None:
        pytest.skip("Pygame visualization dependency is not installed")

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    env = MOBAEnv(seed=42, render_mode="rgb_array")
    env.reset()
    frame = env.render()

    assert frame.shape == (700, 1000, 3)
    assert frame.dtype == np.uint8
    env.close()


def test_recorded_trajectory_is_json_serializable_and_replayable():
    import json
    from moba_rl.visualization.replay import _game_from_state

    env = MOBAEnv(seed=42, record_trajectory=True)
    env.reset()
    env.step(0)

    assert len(env.trajectory) == 2
    json.dumps(env.trajectory)
    replay_game = _game_from_state(env.trajectory[-1]["state"])
    assert replay_game.champions[Team.BLUE].hp == env.game.champions[Team.BLUE].hp
    assert replay_game.map.config.width == env.game.map.config.width
    assert len(replay_game.minions) == len(env.game.minions)


def test_advanced_profile_exposes_all_macro_orders_and_observation_features():
    env = MOBAEnv(
        seed=42, hierarchical_actions=True, advanced_decisions=True,
        fog_of_war=True, ticks_per_decision=1,
    )
    try:
        observation, _ = env.reset()
        assert observation.shape == (57,)
        assert env.action_space.n == 36
        assert env.action_masks().shape == (36,)
        assert env.action_masks()[24:36].all()

        for action in range(24, 36):
            env.reset(seed=100 + action)
            result, _, terminated, truncated, _ = env.step(action)
            assert result.shape == (57,)
            assert env.observation_space.contains(result)
            assert not terminated and not truncated
    finally:
        env.close()


def test_wards_grant_vision_and_sweeping_removes_only_enemy_wards():
    env = MOBAEnv(seed=42, fog_of_war=True, advanced_decisions=True)
    try:
        env.reset()
        game = env.game
        blue = game.champions[Team.BLUE]
        remote = Position(game.map.config.width * 0.5, blue.position.y)
        assert not env._position_visible(remote, Team.BLUE)
        assert game.place_ward(Team.BLUE, remote)
        assert env._position_visible(remote, Team.BLUE)
        near_blue = Position(blue.position.x + 20, blue.position.y)
        assert game.place_ward(Team.RED, near_blue)

        removed = game.sweep_wards(Team.BLUE, radius=500)
        assert removed == 1
        assert len(game.wards) == 1
        assert game.wards[0].team == Team.BLUE
    finally:
        env.close()


def test_wave_orders_change_mode_and_expire():
    from moba_rl.simulation.game import Game

    game = Game(seed=42)
    assert game.set_wave_order(Team.BLUE, "top", "fast_push")
    assert game.wave_order_for(Team.BLUE, "top") == "fast_push"
    game._update_wave_orders(20.0)
    assert game.wave_order_for(Team.BLUE, "top") == "none"


def test_adaptive_opponent_is_available_for_held_out_evaluation():
    from moba_rl.agents.scripted import create_opponent

    opponent = create_opponent("adaptive")
    env = MOBAEnv(seed=42, opponent=opponent)
    env.reset()
    action = opponent.act(env.game)
    assert isinstance(action, int)
    assert 0 <= action < env.action_space.n
