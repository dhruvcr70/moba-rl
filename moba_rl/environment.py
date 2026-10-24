from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from moba_rl.simulation.game import Game
from moba_rl.simulation.lini_tities import Team, Position
from moba_rl.actions import ADVANCED_ACTIONS, HIERARCHICAL_ACTIONS, action_name


class MOBAEnv(gym.Env):
    """
    Gymnasium environment for the custom 1v1 MOBA simulation.

    The agent controls BLUE; the opponent controls RED.
    Observation: 33 base features, with optional strategic and visibility features.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        opponent=None,
        max_steps: int = 5000,
        simulation_dt: float = 0.1,
        ticks_per_decision: int = 1,
        seed: int | None = None,
        progress_reward_scale: float = 0.0,
        progress_target_mode: str = "lane",
        damage_reward_scale: float = 0.0,
        objective_reward_scale: float = 0.0,
        structure_loss_penalty_scale: float = 0.0,
        objective_capture_reward: float = 2.0,
        procedural_map: bool = False,
        fog_of_war: bool = False,
        strategic_actions: bool = False,
        hierarchical_actions: bool = False,
        advanced_decisions: bool = False,
        render_mode: str | None = None,
        record_trajectory: bool = False,
    ):
        super().__init__()
        self.opponent = opponent
        self.max_steps = max_steps
        self.simulation_dt = float(simulation_dt)
        self.ticks_per_decision = int(ticks_per_decision)
        if self.simulation_dt <= 0.0 or self.ticks_per_decision < 1:
            raise ValueError("simulation_dt must be positive and ticks_per_decision at least 1")
        self.initial_seed = seed
        self.procedural_map = bool(procedural_map)
        self.fog_of_war = bool(fog_of_war)
        self.advanced_decisions = bool(advanced_decisions)
        self.hierarchical_actions = bool(hierarchical_actions or advanced_decisions)
        self.strategic_actions = bool(strategic_actions or self.hierarchical_actions)
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.render_mode = render_mode
        self.record_trajectory = bool(record_trajectory)
        self.trajectory: list[dict[str, Any]] = []
        self.last_opponent_action: int | None = None
        self._last_event_count = 0
        self._pygame = None
        self._screen = None
        self._render_clock = None
        self.vision_radius = 420.0
        self.minion_vision_radius = 220.0
        self.progress_reward_scale = float(progress_reward_scale)
        if self.progress_reward_scale < 0.0:
            raise ValueError("progress_reward_scale must be non-negative")
        if progress_target_mode not in {"lane", "objective_then_lane"}:
            raise ValueError(
                "progress_target_mode must be 'lane' or 'objective_then_lane'"
            )
        self.progress_target_mode = progress_target_mode
        self.damage_reward_scale = float(damage_reward_scale)
        if self.damage_reward_scale < 0.0:
            raise ValueError("damage_reward_scale must be non-negative")
        self.objective_reward_scale = float(objective_reward_scale)
        if self.objective_reward_scale < 0.0:
            raise ValueError("objective_reward_scale must be non-negative")
        self.structure_loss_penalty_scale = float(
            structure_loss_penalty_scale
        )
        if self.structure_loss_penalty_scale < 0.0:
            raise ValueError(
                "structure_loss_penalty_scale must be non-negative"
            )
        self.objective_capture_reward = float(objective_capture_reward)
        if self.objective_capture_reward < 0.0:
            raise ValueError("objective_capture_reward must be non-negative")
        self.game: Game | None = None
        self.steps = 0
        self._has_reset = False
        self._episode_action_counts: dict[int, int] = {}

        self.action_space = spaces.Discrete(
            36 if self.advanced_decisions
            else 24 if self.hierarchical_actions
            else 17 if self.strategic_actions
            else 12
        )
        observation_size = (
            33
            + (6 if self.strategic_actions else 0)
            + (4 if self.fog_of_war else 0)
            + (14 if self.advanced_decisions else 0)
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )

    # Gymnasium API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        reset_seed = seed
        if reset_seed is None and not self._has_reset:
            reset_seed = self.initial_seed
        super().reset(seed=reset_seed)

        # Advance Gymnasium's RNG so unseeded episode resets do not recreate
        # the same world. An explicit reset seed still reproduces that episode.
        game_seed = int(
            self.np_random.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
        )
        self._has_reset = True
        self.game = Game(
            seed=game_seed,
            procedural_map=self.procedural_map,
            objective_capture_reward=self.objective_capture_reward,
        )
        self._fog_memory = {}
        for team in Team:
            opposing_team = self.game.enemy_team(team)
            self._fog_memory[team] = {
                "enemy": None,
                "tower_hp": {
                    lane: self.game.tower_for_lane(opposing_team, lane).hp
                    for lane in self.game.map.lanes
                },
                "base_hp": self.game.bases[opposing_team].hp,
                "objective_hp": self.game.objective.hp,
                "objective_alive": float(self.game.objective.alive),
            }
        self.steps = 0
        self._episode_action_counts = {}
        self.last_opponent_action = None
        self._last_event_count = len(self.game.events)
        self.trajectory = []
        observation = self._get_observation()
        if self.record_trajectory:
            self.trajectory.append({
                "step": 0,
                "time": self.game.time,
                "state": self.game.get_state(),
                "observation": observation.tolist(),
                "events": [],
            })

        return observation, {
            "steps": self.steps,
            "time": self.game.time,
            "game_seed": game_seed,
        }

    def step(self, action: int):
        if self.game is None:
            raise RuntimeError(
                "Environment must be reset before calling step()."
            )

        self.steps += 1
        chosen_action = int(action)
        self._episode_action_counts[chosen_action] = (
            self._episode_action_counts.get(chosen_action, 0) + 1
        )

        agent = self.game.champions[Team.BLUE]
        target = self._get_progress_target()
        if self.hierarchical_actions and int(action) >= 17:
            target = self._get_hierarchical_target(int(action), Team.BLUE)
        target = Position(target.x, target.y)
        distance_before_action = agent.position.distance_to(target)
        target_health_before = self._tracked_target_health()
        own_structure_health_before = self._own_structure_health()
        blue_objectives_before = self.game.champions[Team.BLUE].objectives_taken
        red_objectives_before = self.game.champions[Team.RED].objectives_taken

        event_start = len(self.game.events)
        reward = 0.0
        simulation_ticks = 0
        # A scripted opponent commits to an order for the whole decision
        # interval, matching the Blue agent's macro action. Keep the original
        # per-tick fallback behavior when no opponent policy was supplied.
        if self.opponent is not None:
            self._run_opponent()
        for _ in range(self.ticks_per_decision):
            self._execute_agent_action(int(action))
            if self.opponent is None:
                self._run_opponent()
            elif simulation_ticks > 0 and self.last_opponent_action is not None:
                self._execute_opponent_action(self.last_opponent_action)
            self.game.step(self.simulation_dt)
            simulation_ticks += 1
            reward += self.game.get_reward(Team.BLUE)
            if self.game.game_over:
                break

        distance_after_action = agent.position.distance_to(target)
        target_health_after = self._tracked_target_health()
        direct_damage = sum(
            max(0.0, before - after)
            for before, after in zip(
                target_health_before,
                target_health_after,
            )
        )
        own_structure_health_after = self._own_structure_health()
        structure_damage_taken = sum(
            max(0.0, before - after)
            for before, after in zip(
                own_structure_health_before,
                own_structure_health_after,
            )
        )

        if self.progress_reward_scale:
            map_width = self.game.map.config.width
            map_height = self.game.map.config.height
            map_diagonal = float(np.hypot(map_width, map_height))
            if map_diagonal > 0.0:
                progress = (
                    distance_before_action - distance_after_action
                ) / map_diagonal
                reward += self.progress_reward_scale * progress

        if self.damage_reward_scale:
            reward += self.damage_reward_scale * direct_damage

        if self.objective_reward_scale:
            blue_captures = (
                self.game.champions[Team.BLUE].objectives_taken
                - blue_objectives_before
            )
            red_captures = (
                self.game.champions[Team.RED].objectives_taken
                - red_objectives_before
            )
            reward += self.objective_reward_scale * (
                blue_captures - red_captures
            )

        if self.structure_loss_penalty_scale:
            reward -= (
                self.structure_loss_penalty_scale
                * structure_damage_taken
            )

        # Small survival reward retained from the existing environment.
        if self.game.champions[Team.BLUE].alive:
            reward += 0.001 * simulation_ticks

        terminated = self.game.game_over
        truncated = self.steps >= self.max_steps and not terminated

        observation = self._get_observation()

        info: dict[str, Any] = {
            "steps": self.steps,
            "time": self.game.time,
            "simulation_ticks": simulation_ticks,
            "decision_interval": self.simulation_dt * simulation_ticks,
            "blue_objectives_taken": self.game.champions[Team.BLUE].objectives_taken,
            "red_objectives_taken": self.game.champions[Team.RED].objectives_taken,
        }

        if terminated or truncated:
            total_actions = max(1, sum(self._episode_action_counts.values()))
            blue = self.game.champions[Team.BLUE]
            red = self.game.champions[Team.RED]
            info.update({
                "blue_action_counts": {
                    action_name(action_id): count
                    for action_id, count in sorted(self._episode_action_counts.items())
                },
                "blue_action_diversity": len(self._episode_action_counts),
                "blue_advanced_action_rate": sum(
                    count for action_id, count in self._episode_action_counts.items()
                    if action_id >= 24
                ) / total_actions,
                "blue_objectives_taken": blue.objectives_taken,
                "red_objectives_taken": red.objectives_taken,
                "blue_towers_destroyed": sum(
                    not tower.alive for tower in self.game.lane_towers[Team.RED].values()
                ),
                "red_towers_destroyed": sum(
                    not tower.alive for tower in self.game.lane_towers[Team.BLUE].values()
                ),
                "blue_ward_placements": sum(
                    "BLUE placed a vision ward" in event for event in self.game.events
                ),
                "red_ward_placements": sum(
                    "RED placed a vision ward" in event for event in self.game.events
                ),
                "blue_wave_orders": sum("BLUE called a " in event for event in self.game.events),
                "red_wave_orders": sum("RED called a " in event for event in self.game.events),
            })

        if self.game.game_over:
            info["winner"] = (
                self.game.winner.value
                if self.game.winner is not None
                else None
            )

        if self.record_trajectory:
            self.trajectory.append({
                "step": self.steps,
                "time": self.game.time,
                "blue_action": int(action),
                "red_action": self.last_opponent_action,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "winner": info.get("winner"),
                "state": self.game.get_state(),
                "observation": observation.tolist(),
                "events": self.game.events[event_start:],
            })
        self._last_event_count = len(self.game.events)

        return observation, reward, terminated, truncated, info

    def _get_progress_target(self) -> Position:
        assert self.game is not None

        objective = self.game.objective
        if (
            self.progress_target_mode == "objective_then_lane"
            and objective is not None
            and objective.alive
        ):
            return objective.position

        lane = self.game.lane_for_team(Team.BLUE)
        enemy_tower = self.game.tower_for_lane(Team.RED, lane)
        if enemy_tower.alive:
            return enemy_tower.position
        return self.game.bases[Team.RED].position

    def _get_hierarchical_target(
        self,
        action: int,
        team: Team,
    ) -> Position:
        """Return the destination associated with a high-level action."""
        assert self.game is not None
        enemy_team = self.game.enemy_team(team)

        if action == HIERARCHICAL_ACTIONS["engage_champion"]:
            enemy_position = self._known_enemy_position(team)
            return (
                enemy_position
                if enemy_position is not None
                else self._get_progress_target_for_team(team)
            )
        if action == HIERARCHICAL_ACTIONS["push_lane"]:
            return self._lane_target(team)
        if action == HIERARCHICAL_ACTIONS["contest_objective"]:
            objective = self.game.objective
            return objective.position if objective.alive else self._lane_target(team)
        if action == HIERARCHICAL_ACTIONS["farm_lane"]:
            minion = self._nearest_enemy_minion(team)
            return minion.position if minion is not None else self._lane_target(team)
        if action == HIERARCHICAL_ACTIONS["return_to_shop"]:
            return self.game.bases[team].position
        if action == HIERARCHICAL_ACTIONS["defend_lane"]:
            own_tower = self.game.tower_for_lane(
                team, self.game.lane_for_team(team)
            )
            return (
                own_tower.position
                if own_tower.alive
                else self.game.bases[team].position
            )
        if action == HIERARCHICAL_ACTIONS["retreat_to_base"]:
            return self.game.bases[team].position
        if action in (24, 25, 26):
            lane = {24: "top", 25: "mid", 26: "bottom"}[action]
            return Position(self.game.map.config.width * 0.5, self.game.map.lane_y(lane))
        if action in (27, 28, 29):
            return self._lane_target(team)
        if action in (30, 31, 32):
            return self.game.objective.position
        if action in (33, 34, 35):
            return self._known_enemy_position(team) or self._lane_target(team)
        return self._get_progress_target_for_team(team)

    def decision_target(
        self,
        action: int | None,
        team: Team,
    ) -> dict[str, float | str] | None:
        """Describe the target implied by an action for the arena UI."""
        assert self.game is not None
        if action is None or action == 0:
            return None

        champion = self.game.champions[team]
        enemy_team = self.game.enemy_team(team)
        target: Position | None = None
        label = ""

        if action in (1, 2, 3, 4):
            distance = 200.0
            targets = {
                1: Position(champion.position.x - distance, champion.position.y),
                2: Position(champion.position.x + distance, champion.position.y),
                3: Position(champion.position.x, champion.position.y - distance),
                4: Position(champion.position.x, champion.position.y + distance),
            }
            target = targets[action]
            target.x = float(np.clip(target.x, 0.0, self.game.map.config.width))
            target.y = float(np.clip(target.y, 0.0, self.game.map.config.height))
            label = f"move {('left', 'right', 'up', 'down')[action - 1]}"
        elif action in (5, 7, 8, 9):
            target = self._known_enemy_position(team)
            label = f"{enemy_team.value.upper()} champion"
        elif action == 6:
            lane = self.game.lane_for_team(team)
            target = self.game.tower_for_lane(enemy_team, lane).position
            label = f"{enemy_team.value.upper()} {lane} tower"
        elif action == 10:
            target = self.game.objective.position
            label = "neutral objective"
        elif action == 11:
            target = self.game.bases[enemy_team].position
            label = f"{enemy_team.value.upper()} base"
        elif action == 12 or action == HIERARCHICAL_ACTIONS["farm_lane"]:
            minion = self._nearest_enemy_minion(team)
            if minion is not None:
                target = minion.position
                label = f"{enemy_team.value.upper()} lane minion"
            else:
                target = self._lane_target(team)
                label = f"{enemy_team.value.upper()} lane tower"
        elif action in (13, 14, 15, 16, HIERARCHICAL_ACTIONS["return_to_shop"]):
            target = self.game.bases[team].position
            label = f"{team.value.upper()} shop"
        elif self.hierarchical_actions and 17 <= action < 24:
            target = self._get_hierarchical_target(action, team)
            labels = {
                17: f"{enemy_team.value.upper()} champion / lane",
                18: f"{enemy_team.value.upper()} lane tower / base",
                19: "neutral objective / lane",
                20: f"{enemy_team.value.upper()} lane minion",
                21: f"{team.value.upper()} shop",
                22: f"{team.value.upper()} tower / base",
                23: f"{team.value.upper()} base",
            }
            label = labels[action]
        elif self.advanced_decisions and 24 <= action < 36:
            target = self._get_hierarchical_target(action, team)
            label = {
                24: "rotate top", 25: "rotate mid", 26: "rotate bottom",
                27: "freeze wave", 28: "slow push", 29: "fast push",
                30: "ward objective", 31: "sweep vision", 32: "zone objective",
                33: "all in", 34: "poke", 35: "save cooldowns",
            }[action]

        if target is None:
            return {"label": label or "no visible target"}
        return {"label": label, "x": float(target.x), "y": float(target.y)}

    def _get_progress_target_for_team(self, team: Team) -> Position:
        """Return a team's lane target, respecting its tower state."""
        assert self.game is not None
        enemy_team = self.game.enemy_team(team)
        lane = self.game.lane_for_team(team)
        tower = self.game.tower_for_lane(enemy_team, lane)
        return tower.position if tower.alive else self.game.bases[enemy_team].position

    def _lane_target(self, team: Team) -> Position:
        return self._get_progress_target_for_team(team)

    def _nearest_enemy_minion(self, team: Team):
        assert self.game is not None
        enemy_team = self.game.enemy_team(team)
        champion = self.game.champions[team]
        targets = [
            minion for minion in self.game.minions
            if minion.alive
            and minion.team == enemy_team
            and minion.lane == self.game.lane_for_team(team)
            and self._position_visible(minion.position, team)
        ]
        return min(
            targets,
            key=lambda minion: champion.position.distance_to(minion.position),
            default=None,
        )

    def _known_enemy_position(self, team: Team) -> Position | None:
        """Return current or remembered enemy location without fog leaks."""
        assert self.game is not None
        enemy_team = self.game.enemy_team(team)
        enemy = self.game.champions[enemy_team]
        if not self.fog_of_war or self._position_visible(enemy.position, team):
            return enemy.position if enemy.alive else None
        remembered = self._fog_memory[team]["enemy"]
        if remembered is None:
            return None
        return Position(remembered[0], remembered[1])

    def _move_team_towards(self, team: Team, target: Position) -> None:
        assert self.game is not None
        champion = self.game.champions[team]
        champion.move_towards(
            target,
            dt=0.1,
            map_width=self.game.map.config.width,
            map_height=self.game.map.config.height,
        )

    def _execute_hierarchical_action(self, action: int, team: Team) -> bool:
        """Execute a macro action, returning whether it handled the action."""
        if (
            action < 17
            or (action < 24 and not self.hierarchical_actions)
            or (action >= 24 and (not self.advanced_decisions or action >= 36))
        ):
            return False

        assert self.game is not None
        champion = self.game.champions[team]
        if not champion.alive:
            return True

        enemy_team = self.game.enemy_team(team)
        target = self._get_hierarchical_target(action, team)
        distance = champion.position.distance_to(target)
        can_attack = champion.attack_timer <= 0.0

        if action == HIERARCHICAL_ACTIONS["engage_champion"]:
            enemy = self.game.champions[enemy_team]
            enemy_position = self._known_enemy_position(team)
            if not enemy.alive or enemy_position is None:
                self._move_team_towards(team, self._lane_target(team))
            elif (
                self._position_visible(enemy.position, team)
                and distance <= champion.attack_range * 1.5
            ):
                for ability, timer in (
                    (3, champion.ability_3_timer),
                    (2, champion.ability_2_timer),
                    (1, champion.ability_1_timer),
                ):
                    if timer <= 0.0:
                        self.game.use_ability(team, ability, enemy_team)
                        return True
                if can_attack and distance <= champion.attack_range:
                    self.game.attack_champion(team, enemy_team)
                else:
                    self._move_team_towards(team, enemy_position)
            else:
                self._move_team_towards(team, enemy_position)
            return True

        if action == HIERARCHICAL_ACTIONS["push_lane"]:
            lane = self.game.lane_for_team(team)
            tower = self.game.tower_for_lane(enemy_team, lane)
            if tower.alive:
                if can_attack and distance <= champion.attack_range:
                    self.game.attack_tower(team, enemy_team)
                else:
                    self._move_team_towards(team, tower.position)
            else:
                base = self.game.bases[enemy_team]
                if can_attack and distance <= self.game.base_attack_range:
                    self.game.attack_base(team, enemy_team)
                else:
                    self._move_team_towards(team, base.position)
            return True

        if action == HIERARCHICAL_ACTIONS["contest_objective"]:
            objective = self.game.objective
            if objective.alive:
                enemy = self.game.champions[enemy_team]
                enemy_visible = self._position_visible(enemy.position, team)
                enemy_distance = champion.position.distance_to(enemy.position)
                enemy_at_objective = (
                    enemy.alive
                    and enemy_visible
                    and enemy.position.distance_to(objective.position)
                    <= objective.attack_range + champion.attack_range
                )
                if (
                    enemy_at_objective
                    and enemy_distance <= champion.attack_range * 1.5
                ):
                    for ability, timer in (
                        (3, champion.ability_3_timer),
                        (2, champion.ability_2_timer),
                        (1, champion.ability_1_timer),
                    ):
                        if timer <= 0.0:
                            self.game.use_ability(team, ability, enemy_team)
                            return True
                    if can_attack and enemy_distance <= champion.attack_range:
                        self.game.attack_champion(team, enemy_team)
                        return True
                if can_attack and distance <= champion.attack_range:
                    self.game.attack_objective(team)
                else:
                    self._move_team_towards(team, objective.position)
            else:
                self._move_team_towards(team, self._lane_target(team))
            return True

        if action == HIERARCHICAL_ACTIONS["farm_lane"]:
            minion = self._nearest_enemy_minion(team)
            if minion is None:
                self._move_team_towards(team, self._lane_target(team))
            elif can_attack and distance <= champion.attack_range:
                self.game.attack_minion(team)
            else:
                self._move_team_towards(team, minion.position)
            return True

        if action == HIERARCHICAL_ACTIONS["return_to_shop"]:
            base = self.game.bases[team]
            if distance > self.game.shop_range:
                self._move_team_towards(team, base.position)
                return True
            if self.advanced_decisions:
                return True
            upgrades = (
                ("blade", 200, "attack_upgrade_level"),
                ("vitality", 200, "vitality_upgrade_level"),
                ("armor", 150, "armor_upgrade_level"),
                ("boots", 150, "speed_upgrade_level"),
            )
            for upgrade, cost, rank_attribute in upgrades:
                if (
                    champion.gold >= cost
                    and getattr(champion, rank_attribute) < 5
                ):
                    self.game.purchase_upgrade(team, upgrade)
                    break
            return True

        if action == HIERARCHICAL_ACTIONS["defend_lane"]:
            own_tower = self.game.tower_for_lane(
                team, self.game.lane_for_team(team)
            )
            defense_position = (
                own_tower.position
                if own_tower.alive
                else self.game.bases[team].position
            )
            defense_radius = own_tower.attack_range if own_tower.alive else 400.0
            enemy = self.game.champions[enemy_team]
            enemy_visible = (
                enemy.alive and self._position_visible(enemy.position, team)
            )
            if (
                enemy_visible
                and enemy.position.distance_to(defense_position)
                <= defense_radius + champion.attack_range
            ):
                enemy_distance = champion.position.distance_to(enemy.position)
                if enemy_distance <= champion.attack_range * 1.5:
                    for ability, timer in (
                        (3, champion.ability_3_timer),
                        (2, champion.ability_2_timer),
                        (1, champion.ability_1_timer),
                    ):
                        if timer <= 0.0:
                            self.game.use_ability(team, ability, enemy_team)
                            return True
                if can_attack and enemy_distance <= champion.attack_range:
                    self.game.attack_champion(team, enemy_team)
                    return True
            threats = [
                minion for minion in self.game.minions
                if minion.alive and minion.team == enemy_team
                and minion.lane == self.game.lane_for_team(team)
                and minion.position.distance_to(defense_position)
                <= defense_radius * 1.5
                and self._position_visible(minion.position, team)
            ]
            minion = min(
                threats,
                key=lambda item: champion.position.distance_to(item.position),
                default=None,
            )
            if minion is not None:
                minion_distance = champion.position.distance_to(minion.position)
                if can_attack and minion_distance <= champion.attack_range:
                    self.game.attack_minion(team)
                else:
                    self._move_team_towards(team, minion.position)
            elif champion.position.distance_to(defense_position) > champion.attack_range:
                self._move_team_towards(team, defense_position)
            return True

        if action == HIERARCHICAL_ACTIONS["retreat_to_base"]:
            base = self.game.bases[team]
            if champion.position.distance_to(base.position) > self.game.shop_range:
                self._move_team_towards(team, base.position)
            return True

        if self.advanced_decisions and action >= 24:
            lane = self.game.lane_for_team(team)
            if action in (24, 25, 26):
                lane_name = {24: "top", 25: "mid", 26: "bottom"}[action]
                destination = Position(self.game.map.config.width * 0.5, self.game.map.lane_y(lane_name))
                self._move_team_towards(team, destination)
            elif action in (27, 28, 29):
                mode = {27: "freeze", 28: "slow_push", 29: "fast_push"}[action]
                self.game.set_wave_order(team, lane, mode)
                wave = [m for m in self.game.minions if m.alive and m.team == team and m.lane == lane]
                front = (max(wave, key=lambda m: m.position.x, default=None) if team == Team.BLUE else min(wave, key=lambda m: m.position.x, default=None))
                if front:
                    if champion.position.distance_to(front.position) > champion.attack_range:
                        self._move_team_towards(team, front.position)
                    elif action == 29 and can_attack:
                        self.game.attack_minion(team)
            elif action == 30:
                objective = self.game.objective
                if champion.position.distance_to(objective.position) > 140:
                    self._move_team_towards(team, objective.position)
                elif not any(w.team == team and w.position.distance_to(objective.position) < 100 for w in self.game.wards):
                    self.game.place_ward(team, objective.position)
            elif action == 31:
                if champion.position.distance_to(self.game.objective.position) > 280:
                    self._move_team_towards(team, self.game.objective.position)
                else:
                    self.game.sweep_wards(team)
            elif action == 32:
                objective = self.game.objective
                enemy = self.game.champions[enemy_team]
                if enemy.alive and self._position_visible(enemy.position, team) and enemy.position.distance_to(objective.position) < 450:
                    d = champion.position.distance_to(enemy.position)
                    if d <= champion.attack_range * 1.5:
                        for ability in (3, 2, 1):
                            if getattr(champion, f"ability_{ability}_timer") <= 0:
                                self.game.use_ability(team, ability, enemy_team)
                                break
                    elif can_attack and d <= champion.attack_range:
                        self.game.attack_champion(team, enemy_team)
                    else:
                        self._move_team_towards(team, Position(objective.position.x + (220 if team == Team.BLUE else -220), objective.position.y))
                else:
                    self._move_team_towards(team, Position(objective.position.x + (220 if team == Team.BLUE else -220), objective.position.y))
            elif action == 33:
                enemy = self.game.champions[enemy_team]
                known = self._known_enemy_position(team)
                if enemy.alive and known and self._position_visible(enemy.position, team):
                    d = champion.position.distance_to(enemy.position)
                    if d <= champion.attack_range * 1.5:
                        for ability in (3, 2, 1):
                            if getattr(champion, f"ability_{ability}_timer") <= 0:
                                self.game.use_ability(team, ability, enemy_team)
                                break
                        else:
                            if can_attack and d <= champion.attack_range:
                                self.game.attack_champion(team, enemy_team)
                    else:
                        self._move_team_towards(team, known)
                else:
                    self._move_team_towards(team, self._lane_target(team))
            elif action == 34:
                enemy = self.game.champions[enemy_team]
                known = self._known_enemy_position(team)
                if enemy.alive and known and self._position_visible(enemy.position, team):
                    d = champion.position.distance_to(enemy.position)
                    if d <= champion.attack_range * 1.5 and champion.ability_1_timer <= 0:
                        self.game.use_ability(team, 1, enemy_team)
                    elif d <= champion.attack_range * 0.75:
                        self._move_team_towards(team, Position(champion.position.x + (-180 if team == Team.BLUE else 180), champion.position.y))
                    else:
                        self._move_team_towards(team, known)
                else:
                    self._move_team_towards(team, self._lane_target(team))
            elif action == 35:
                enemy = self.game.champions[enemy_team]
                if enemy.alive and self._position_visible(enemy.position, team) and champion.position.distance_to(enemy.position) < 400:
                    own_tower = self.game.tower_for_lane(team, lane)
                    self._move_team_towards(team, own_tower.position if own_tower.alive else self.game.bases[team].position)
                else:
                    minion = self._nearest_enemy_minion(team)
                    if minion and can_attack and champion.position.distance_to(minion.position) <= champion.attack_range:
                        self.game.attack_minion(team)
            return True

        return False

    def _tracked_target_health(self) -> tuple[float, ...]:
        assert self.game is not None

        return (
            self.game.champions[Team.RED].hp,
            self.game.tower_for_lane(
                Team.RED, self.game.lane_for_team(Team.BLUE)
            ).hp,
            self.game.bases[Team.RED].hp,
            self.game.objective.hp,
        )

    def _own_structure_health(self) -> tuple[float, float]:
        assert self.game is not None

        return (
            self.game.tower_for_lane(
                Team.BLUE, self.game.lane_for_team(Team.BLUE)
            ).hp,
            self.game.bases[Team.BLUE].hp,
        )

    # Agent actions

    def _execute_agent_action(self, action: int) -> None:
        assert self.game is not None

        agent = self.game.champions[Team.BLUE]

        if not agent.alive:
            return

        if self._execute_hierarchical_action(action, Team.BLUE):
            return

        if action == 0:
            return

        elif action in (1, 2, 3, 4):
            targets = {
                1: Position(agent.position.x - 200.0, agent.position.y),
                2: Position(agent.position.x + 200.0, agent.position.y),
                3: Position(agent.position.x, agent.position.y - 200.0),
                4: Position(agent.position.x, agent.position.y + 200.0),
            }

            agent.move_towards(
                targets[action],
                dt=0.1,
                map_width=self.game.map.config.width,
                map_height=self.game.map.config.height,
            )

        elif action == 5:
            self.game.attack_champion(Team.BLUE, Team.RED)

        elif action == 6:
            self.game.attack_tower(Team.BLUE, Team.RED)

        elif action == 7:
            self.game.use_ability(Team.BLUE, 1, Team.RED)

        elif action == 8:
            self.game.use_ability(Team.BLUE, 2, Team.RED)

        elif action == 9:
            self.game.use_ability(Team.BLUE, 3, Team.RED)

        elif action == 10:
            self.game.attack_objective(Team.BLUE)

        elif action == 11:
            self.game.attack_base(Team.BLUE, Team.RED)

        elif self.strategic_actions and action == 12:
            self.game.attack_minion(Team.BLUE)

        elif self.strategic_actions and action == 13:
            self.game.purchase_upgrade(Team.BLUE, "blade")

        elif self.strategic_actions and action == 14:
            self.game.purchase_upgrade(Team.BLUE, "armor")

        elif self.strategic_actions and action == 15:
            self.game.purchase_upgrade(Team.BLUE, "boots")

        elif self.strategic_actions and action == 16:
            self.game.purchase_upgrade(Team.BLUE, "vitality")

        else:
            raise ValueError(f"Invalid action: {action}")

    # Opponent

    def _run_opponent(self) -> None:
        assert self.game is not None

        self.last_opponent_action = None
        if self.opponent is not None:
            if hasattr(self.opponent, "act_for_env"):
                self.last_opponent_action = self.opponent.act_for_env(self)
                return

            action = self.opponent.act(self.game)
            if action is not None:
                self.last_opponent_action = int(action)
                self._execute_opponent_action(int(action))
            return

        # Fallback aggressive behavior.
        enemy = self.game.champions[Team.RED]
        agent = self.game.champions[Team.BLUE]

        if not enemy.alive:
            return

        distance = enemy.position.distance_to(agent.position)

        if distance <= enemy.attack_range:
            self.game.attack_champion(Team.RED, Team.BLUE)
        else:
            enemy.move_towards(
                agent.position,
                dt=0.1,
                map_width=self.game.map.config.width,
                map_height=self.game.map.config.height,
            )

    def _execute_opponent_action(self, action: int) -> None:
        """Translate scripted-opponent actions into simulation operations."""
        assert self.game is not None

        enemy = self.game.champions[Team.RED]

        if not enemy.alive:
            return

        if self._execute_hierarchical_action(action, Team.RED):
            return

        if action == 0:
            return

        elif action in (1, 2, 3, 4):
            targets = {
                1: Position(enemy.position.x - 200.0, enemy.position.y),
                2: Position(enemy.position.x + 200.0, enemy.position.y),
                3: Position(enemy.position.x, enemy.position.y - 200.0),
                4: Position(enemy.position.x, enemy.position.y + 200.0),
            }

            enemy.move_towards(
                targets[action],
                dt=0.1,
                map_width=self.game.map.config.width,
                map_height=self.game.map.config.height,
            )

        elif action == 5:
            self.game.attack_champion(Team.RED, Team.BLUE)

        elif action == 6:
            self.game.attack_tower(Team.RED, Team.BLUE)

        elif action == 7:
            self.game.use_ability(Team.RED, 1, Team.BLUE)

        elif action == 8:
            self.game.use_ability(Team.RED, 2, Team.BLUE)

        elif action == 9:
            self.game.use_ability(Team.RED, 3, Team.BLUE)

        elif action == 10:
            self.game.attack_objective(Team.RED)

        elif action == 11:
            self.game.attack_base(Team.RED, Team.BLUE)

        elif self.strategic_actions and action == 12:
            self.game.attack_minion(Team.RED)

        elif self.strategic_actions and action == 13:
            self.game.purchase_upgrade(Team.RED, "blade")

        elif self.strategic_actions and action == 14:
            self.game.purchase_upgrade(Team.RED, "armor")

        elif self.strategic_actions and action == 15:
            self.game.purchase_upgrade(Team.RED, "boots")

        elif self.strategic_actions and action == 16:
            self.game.purchase_upgrade(Team.RED, "vitality")

        else:
            raise ValueError(f"Invalid opponent action: {action}")

    # Observations

    def _position_visible(
        self,
        position: Position,
        team: Team = Team.BLUE,
    ) -> bool:
        """Check radial vision from a team's champion, minions, and tower."""
        assert self.game is not None
        if not self.fog_of_war:
            return True

        agent = self.game.champions[team]
        if (
            agent.alive
            and agent.position.distance_to(position) <= self.vision_radius
        ):
            return True

        for minion in self.game.minions:
            if (
                minion.alive
                and minion.team == team
                and minion.position.distance_to(position)
                <= self.minion_vision_radius
            ):
                return True

        return any(
            tower.alive and tower.position.distance_to(position) <= 300.0
            for tower in self.game.lane_towers[team].values()
        ) or any(
            ward.team == team and ward.position.distance_to(position) <= 380.0
            for ward in self.game.wards
        )

    def _get_observation(self, team: Team = Team.BLUE) -> np.ndarray:
        """Build one side's observation, mirrored into a forward-facing view."""
        assert self.game is not None
        if team not in Team:
            raise ValueError(f"Unknown team: {team!r}")

        enemy_team = self.game.enemy_team(team)
        agent = self.game.champions[team]
        enemy = self.game.champions[enemy_team]
        lane = self.game.lane_for_team(team)
        own_tower = self.game.tower_for_lane(team, lane)
        enemy_tower = self.game.tower_for_lane(enemy_team, lane)
        own_base = self.game.bases[team]
        enemy_base = self.game.bases[enemy_team]
        objective = self.game.objective
        width = self.game.map.config.width
        height = self.game.map.config.height
        memory = self._fog_memory[team]

        def oriented_x(position: Position) -> float:
            return position.x if team == Team.BLUE else width - position.x

        def hp_ratio(hp: float, max_hp: float) -> float:
            if max_hp <= 0:
                return 0.0
            return self._clip01(hp / max_hp)

        def normalize_distance(distance: float) -> float:
            diagonal = float(np.hypot(width, height))
            return self._clip01(distance / diagonal) if diagonal > 0 else 0.0

        enemy_visible = self._position_visible(enemy.position, team)
        tower_visible = self._position_visible(enemy_tower.position, team)
        base_visible = self._position_visible(enemy_base.position, team)
        objective_visible = self._position_visible(objective.position, team)

        if enemy_visible:
            memory["enemy"] = (
                enemy.position.x,
                enemy.position.y,
                enemy.hp,
                enemy.level,
                enemy.gold,
                float(enemy.alive),
            )
        if tower_visible:
            memory["tower_hp"][lane] = enemy_tower.hp
        if base_visible:
            memory["base_hp"] = enemy_base.hp
        if objective_visible:
            memory["objective_hp"] = objective.hp
            memory["objective_alive"] = float(objective.alive)

        if memory["enemy"] is None:
            enemy_x, enemy_y, enemy_hp, enemy_level, enemy_gold, enemy_alive = (
                0.0, 0.0, 0.0, 0, 0, 0.0
            )
            enemy_distance = float(np.hypot(width, height))
        elif enemy_visible or not self.fog_of_war:
            enemy_x, enemy_y, enemy_hp, enemy_level, enemy_gold, enemy_alive = (
                enemy.position.x,
                enemy.position.y,
                enemy.hp,
                enemy.level,
                enemy.gold,
                float(enemy.alive),
            )
            enemy_distance = agent.position.distance_to(enemy.position)
        else:
            enemy_x, enemy_y, enemy_hp, enemy_level, enemy_gold, enemy_alive = (
                memory["enemy"]
            )
            enemy_distance = agent.position.distance_to(
                Position(enemy_x, enemy_y)
            )

        own_minions = sum(
            1 for minion in self.game.minions
            if minion.alive and minion.team == team
        )
        enemy_minions = sum(
            1 for minion in self.game.minions
            if minion.alive
            and minion.team == enemy_team
            and self._position_visible(minion.position, team)
        )

        enemy_tower_hp = (
            enemy_tower.hp if tower_visible or not self.fog_of_war
            else memory["tower_hp"][lane]
        )
        enemy_base_hp = (
            enemy_base.hp if base_visible or not self.fog_of_war
            else memory["base_hp"]
        )
        objective_hp = (
            objective.hp if objective_visible or not self.fog_of_war
            else memory["objective_hp"]
        )
        objective_alive = (
            float(objective.alive)
            if objective_visible or not self.fog_of_war
            else memory["objective_alive"]
        )

        features = [
            # Agent state: 8 features.
            self._clip01(oriented_x(agent.position) / width),
            self._clip01(agent.position.y / height),
            hp_ratio(agent.hp, agent.max_hp),
            self._clip01(agent.level / 10.0),
            self._clip01(agent.xp / max(agent.xp_to_next_level(), 1.0)),
            self._clip01(agent.gold / 5000.0),
            self._clip01(agent.kills / 10.0),
            self._clip01(agent.deaths / 10.0),

            # Opposing champion: current sight or remembered information.
            self._clip01((enemy_x if team == Team.BLUE else width - enemy_x) / width),
            self._clip01(enemy_y / height),
            hp_ratio(enemy_hp, enemy.max_hp),
            self._clip01(enemy_level / 10.0),
            self._clip01(enemy_gold / 5000.0),

            # Structures.
            hp_ratio(own_tower.hp, own_tower.max_hp),
            hp_ratio(enemy_tower_hp, enemy_tower.max_hp),
            hp_ratio(own_base.hp, own_base.max_hp),
            hp_ratio(enemy_base_hp, enemy_base.max_hp),

            # Objective and normalized distances.
            hp_ratio(objective_hp, objective.max_hp),
            self._clip01(oriented_x(objective.position) / width),
            self._clip01(objective.position.y / height),
            normalize_distance(agent.position.distance_to(objective.position)),
            normalize_distance(enemy_distance),
            normalize_distance(agent.position.distance_to(enemy_tower.position)),
            normalize_distance(agent.position.distance_to(enemy_base.position)),

            # Agent cooldowns.
            self._clip01(agent.ability_1_timer / agent.ability_1_cooldown if agent.ability_1_cooldown > 0 else 0.0),
            self._clip01(agent.ability_2_timer / agent.ability_2_cooldown if agent.ability_2_cooldown > 0 else 0.0),
            self._clip01(agent.ability_3_timer / agent.ability_3_cooldown if agent.ability_3_cooldown > 0 else 0.0),
            self._clip01(agent.attack_timer / agent.attack_cooldown if agent.attack_cooldown > 0 else 0.0),

            # Visible minions and global clock.
            self._clip01(own_minions / 20.0),
            self._clip01(enemy_minions / 20.0),
            self._clip01(self.game.time / 300.0),
            objective_alive,
            enemy_alive,
        ]

        if self.strategic_actions:
            visible_enemy_minions = [
                minion for minion in self.game.minions
                if minion.alive
                and minion.team == enemy_team
                and minion.lane == lane
                and self._position_visible(minion.position, team)
            ]
            nearest_minion = min(
                visible_enemy_minions,
                key=lambda minion: agent.position.distance_to(minion.position),
                default=None,
            )
            if nearest_minion is None:
                minion_distance = float(np.hypot(width, height))
                minion_hp = 0.0
            else:
                minion_distance = agent.position.distance_to(nearest_minion.position)
                minion_hp = hp_ratio(nearest_minion.hp, nearest_minion.max_hp)

            own_base_distance = agent.position.distance_to(own_base.position)
            features.extend([
                self._clip01(agent.attack_damage / 250.0),
                self._clip01(agent.armor / 25.0),
                self._clip01(agent.move_speed / 250.0),
                normalize_distance(minion_distance),
                minion_hp,
                normalize_distance(own_base_distance),
            ])

        if self.fog_of_war:
            features.extend([
                float(enemy_visible),
                float(tower_visible),
                float(base_visible),
                float(objective_visible),
            ])

        if self.advanced_decisions:
            current_lane = self.game.lane_for_team(team)
            wave_mode = self.game.wave_order_for(team, current_lane)
            own_wave = sum(1 for m in self.game.minions if m.alive and m.team == team and m.lane == current_lane)
            visible_enemy_wave = sum(1 for m in self.game.minions if m.alive and m.team == enemy_team and m.lane == current_lane and self._position_visible(m.position, team))
            own_wards = sum(1 for w in self.game.wards if w.team == team)
            visible_enemy_wards = sum(1 for w in self.game.wards if w.team == enemy_team and self._position_visible(w.position, team))
            features.extend([
                float(current_lane == "top"), float(current_lane == "mid"), float(current_lane == "bottom"),
                float(wave_mode == "none"), float(wave_mode == "freeze"), float(wave_mode == "slow_push"), float(wave_mode == "fast_push"),
                self._clip01(own_wave / 9.0), self._clip01(visible_enemy_wave / 9.0),
                self._clip01(own_wards / 2.0), self._clip01(visible_enemy_wards / 2.0),
                self._clip01(agent.ward_timer / max(agent.ward_cooldown, 1.0)),
                self._clip01(agent.sweep_timer / max(agent.sweep_cooldown, 1.0)),
                self._clip01((self.game.objective_respawn_timer or 0.0) / 90.0),
            ])

        observation = np.asarray(features, dtype=np.float32)
        return np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)

    # Action masking

    def action_masks(self, team: Team = Team.BLUE) -> np.ndarray:
        """
        Return a boolean action mask for MaskablePPO.
        True means legal; False means unavailable.
        """
        assert self.game is not None

        enemy_team = self.game.enemy_team(team)
        agent = self.game.champions[team]
        enemy = self.game.champions[enemy_team]
        lane = self.game.lane_for_team(team)
        enemy_tower = self.game.tower_for_lane(enemy_team, lane)
        enemy_base = self.game.bases[enemy_team]
        objective = self.game.objective

        mask = np.zeros(self.action_space.n, dtype=bool)

        # MaskablePPO requires at least one valid action.
        if not agent.alive or self.game.game_over:
            mask[0] = True
            return mask

        # Movement and no-op are always available while alive.
        mask[0:5] = True

        enemy_distance = agent.position.distance_to(enemy.position)
        tower_distance = agent.position.distance_to(enemy_tower.position)
        objective_distance = agent.position.distance_to(objective.position)
        base_distance = agent.position.distance_to(enemy_base.position)

        # Basic champion attack: attack cooldown and standard range.
        mask[5] = (
            (not self.fog_of_war or self._position_visible(enemy.position, team))
            and enemy.alive
            and enemy_distance <= agent.attack_range
            and agent.attack_timer <= 0.0
        )

        # Tower attack: standard range and attack cooldown.
        mask[6] = (
            (not self.fog_of_war or self._position_visible(enemy_tower.position, team))
            and enemy_tower.alive
            and tower_distance <= agent.attack_range
            and agent.attack_timer <= 0.0
        )

        # Abilities: game.use_ability() permits 1.5x attack range.
        ability_range = agent.attack_range * 1.5

        mask[7] = (
            (not self.fog_of_war or self._position_visible(enemy.position, team))
            and enemy.alive
            and enemy_distance <= ability_range
            and agent.ability_1_timer <= 0.0
        )
        mask[8] = (
            (not self.fog_of_war or self._position_visible(enemy.position, team))
            and enemy.alive
            and enemy_distance <= ability_range
            and agent.ability_2_timer <= 0.0
        )
        mask[9] = (
            (not self.fog_of_war or self._position_visible(enemy.position, team))
            and enemy.alive
            and enemy_distance <= ability_range
            and agent.ability_3_timer <= 0.0
        )

        # Objective attack: standard range and attack cooldown.
        mask[10] = (
            (not self.fog_of_war or self._position_visible(objective.position, team))
            and objective.alive
            and objective_distance <= agent.attack_range
            and agent.attack_timer <= 0.0
        )

        # Base attack uses the game's dedicated base attack range.
        mask[11] = (
            (not self.fog_of_war or self._position_visible(enemy_base.position, team))
            and not enemy_tower.alive
            and enemy_base.alive
            and base_distance <= self.game.base_attack_range
            and agent.attack_timer <= 0.0
        )

        if self.strategic_actions:
            minion_targets = [
                minion for minion in self.game.minions
                if minion.alive
                and minion.team == enemy_team
                and minion.lane == lane
                and self._position_visible(minion.position, team)
                and agent.position.distance_to(minion.position)
                <= agent.attack_range
            ]
            mask[12] = (
                bool(minion_targets)
                and agent.attack_timer <= 0.0
            )

            near_shop = (
                agent.position.distance_to(self.game.bases[team].position)
                <= self.game.shop_range
            )
            shop_upgrades = (
                ("blade", 200, "attack_upgrade_level"),
                ("armor", 150, "armor_upgrade_level"),
                ("boots", 150, "speed_upgrade_level"),
                ("vitality", 200, "vitality_upgrade_level"),
            )
            for action, (upgrade, cost, rank_attribute) in enumerate(
                shop_upgrades,
                start=13,
            ):
                del upgrade
                mask[action] = (
                    near_shop
                    and agent.gold >= cost
                    and getattr(agent, rank_attribute) < 5
                )

        if self.hierarchical_actions:
            # Macro actions are legal at all times. They either act on the
            # target now or move toward it, so no target-range masking is
            # needed.
            mask[17:24] = True

        if self.advanced_decisions:
            mask[24:36] = True

        # Safety fallback.
        if not mask.any():
            mask[0] = True

        return mask

    # Compatibility alias for existing code/tests.
    def action_mask(self) -> np.ndarray:
        return self.action_masks()

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def render(self):
        """Render the current match using Pygame or return an RGB frame."""
        if self.game is None or self.render_mode is None:
            return None

        import pygame
        self._pygame = pygame

        if not pygame.get_init():
            pygame.init()
        if self._render_clock is None:
            self._render_clock = pygame.time.Clock()

        frame_width, frame_height = 1000, 700
        surface = pygame.Surface((frame_width, frame_height))
        surface.fill((38, 92, 62))
        game = self.game
        config = game.map.config
        left, top, field_width, field_height = 36, 112, 928, 548

        def screen_position(position):
            return (
                int(left + position.x / config.width * field_width),
                int(top + (1.0 - position.y / config.height) * field_height),
            )

        def draw_bar(x, y, width, height, hp, max_hp, color):
            pygame.draw.rect(surface, (28, 28, 28), (x, y, width, height))
            fraction = 0.0 if max_hp <= 0 else max(0.0, min(1.0, hp / max_hp))
            pygame.draw.rect(
                surface,
                color,
                (x, y, int(width * fraction), height),
            )

        lane_colors = {
            "top": (123, 108, 80),
            "mid": (140, 124, 91),
            "bottom": (123, 108, 80),
        }
        for lane in game.map.lanes:
            lane_y = screen_position(Position(0.0, game.map.lane_y(lane)))[1]
            pygame.draw.line(surface, lane_colors[lane], (left, lane_y),
                             (left + field_width, lane_y), 42)
            pygame.draw.line(surface, (183, 164, 124), (left, lane_y),
                             (left + field_width, lane_y), 2)
            surface.blit(pygame.font.Font(None, 18).render(lane.upper(), True, (220, 210, 185)),
                         (left + 4, lane_y - 25))
            active_orders = [
                f"{team.value.upper()} {game.wave_order_for(team, lane).replace('_', ' ')}"
                for team in Team if game.wave_order_for(team, lane) != "none"
            ]
            if active_orders:
                surface.blit(pygame.font.Font(None, 18).render(" · ".join(active_orders), True, (255, 221, 143)),
                             (left + 72, lane_y - 25))

        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        blue = (66, 154, 255)
        red = (239, 79, 79)
        gold = (245, 194, 66)
        for upper, lower in (("top", "mid"), ("mid", "bottom")):
            jungle_y = (game.map.lane_y(upper) + game.map.lane_y(lower)) * 0.5
            x, y = screen_position(Position(config.width * 0.5, jungle_y))
            surface.blit(small_font.render("JUNGLE", True, (151, 171, 143)),
                         (x - 24, y - 8))

        # Header and match state.
        header = font.render(
            f"MOBA-RL  |  {game.time:6.1f}s  |  "
            f"Blue {game.champions[Team.BLUE].gold}g  "
            f"Lv {game.champions[Team.BLUE].level}  |  "
            f"Red {game.champions[Team.RED].gold}g  "
            f"Lv {game.champions[Team.RED].level}",
            True,
            (244, 244, 244),
        )
        surface.blit(header, (28, 18))

        # Structures.
        for team, color in ((Team.BLUE, blue), (Team.RED, red)):
            structures = [game.bases[team], *game.lane_towers[team].values()]
            for structure in structures:
                x, y = screen_position(structure.position)
                size = 30 if structure in game.bases.values() else 22
                rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
                pygame.draw.rect(surface, color, rect, border_radius=4)
                draw_bar(
                    x - 30,
                    y - size // 2 - 10,
                    60,
                    6,
                    structure.hp,
                    structure.max_hp,
                    (105, 220, 120),
                )

        # Objective and units.
        objective = game.objective
        if objective is not None and objective.alive:
            x, y = screen_position(objective.position)
            pygame.draw.circle(surface, gold, (x, y), 16)
            pygame.draw.circle(surface, (255, 236, 165), (x, y), 16, 2)
            draw_bar(x - 28, y - 26, 56, 6, objective.hp, objective.max_hp, gold)
            surface.blit(small_font.render("OBJ", True, (25, 25, 25)), (x - 12, y - 7))
        elif game.objective_respawn_timer is not None:
            text = small_font.render(
                f"Objective returns in {game.objective_respawn_timer:.0f}s",
                True,
                gold,
            )
            surface.blit(text, (left + field_width // 2 - 72, 82))

        for ward in game.wards:
            x, y = screen_position(ward.position)
            ward_color = blue if ward.team == Team.BLUE else red
            pygame.draw.circle(surface, (20, 25, 30), (x, y), 8)
            pygame.draw.circle(surface, ward_color, (x, y), 7, 2)
            pygame.draw.circle(surface, ward_color, (x, y), 17, 1)

        for minion in game.minions:
            x, y = screen_position(minion.position)
            pygame.draw.circle(surface, blue if minion.team == Team.BLUE else red, (x, y), 5)

        for team, color in ((Team.BLUE, blue), (Team.RED, red)):
            champion = game.champions[team]
            x, y = screen_position(champion.position)
            if champion.alive:
                pygame.draw.circle(surface, (245, 245, 245), (x, y), 15)
                pygame.draw.circle(surface, color, (x, y), 12)
                draw_bar(x - 24, y - 24, 48, 6, champion.hp, champion.max_hp, (90, 230, 130))
                surface.blit(
                    small_font.render(f"{team.value.upper()} L{champion.level}", True, (255, 255, 255)),
                    (x - 25, y + 17),
                )
            else:
                surface.blit(
                    small_font.render(f"{team.value.upper()} respawn {champion.respawn_timer:.1f}s", True, color),
                    (x - 35, y + 18),
                )

        if game.game_over:
            winner = game.winner.value.upper() if game.winner else "DRAW"
            banner = font.render(f"{winner} WINS", True, gold)
            surface.blit(banner, (frame_width // 2 - banner.get_width() // 2, 76))

        if self.render_mode == "human":
            if self._screen is None:
                self._screen = pygame.display.set_mode((frame_width, frame_height))
                pygame.display.set_caption("MOBA-RL Match Viewer")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            self._screen.blit(surface, (0, 0))
            pygame.display.flip()
            self._render_clock.tick(self.metadata["render_fps"])
            return None

        return pygame.surfarray.array3d(surface).swapaxes(0, 1)

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
        elif self._screen is not None:
            import pygame
            pygame.display.quit()
        self._screen = None
        self._render_clock = None
