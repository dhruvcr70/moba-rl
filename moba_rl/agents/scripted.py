from __future__ import annotations

from abc import ABC, abstractmethod

from moba_rl.simulation.lini_tities import Team, Position


class ScriptedOpponent(ABC):
    """
    Base class for deterministic scripted opponents.

    The opponent always controls RED.
    """

    @abstractmethod
    def act(self, game) -> int:
        """
        Return an action using the same action IDs as MOBAEnv.

        0  = no-op
        1  = move left
        2  = move right
        3  = move up
        4  = move down
        5  = attack champion
        6  = attack tower
        7  = ability 1
        8  = ability 2
        9  = ability 3
        10 = attack objective
        11 = attack enemy base
        """
        raise NotImplementedError

    def act_for_env(self, env) -> int:
        """Apply one readable, state-aware RED order through the shared executor."""
        game = env.game
        team = Team.RED
        champion = game.champions[team]
        if not champion.alive:
            return 0

        action = self._advanced_order(env) if env.advanced_decisions else self.act(game)
        env._execute_opponent_action(action)
        return action

    def _advanced_order(self, env) -> int:
        """Choose an advanced macro from visible threats and current map state."""
        game = env.game
        team = Team.RED
        enemy_team = Team.BLUE
        champion = game.champions[team]
        enemy = game.champions[enemy_team]
        lane = game.lane_for_team(team)
        own_tower = game.tower_for_lane(team, lane)
        objective = game.objective
        health = champion.hp / max(1.0, champion.max_hp)
        enemy_visible = enemy.alive and env._position_visible(enemy.position, team)
        enemy_distance = champion.position.distance_to(enemy.position) if enemy_visible else float("inf")

        # Spend accumulated gold on a deliberate stat choice while in shop range.
        if champion.position.distance_to(game.bases[team].position) <= game.shop_range:
            affordable = [
                (13, "attack_upgrade_level", 200),
                (14, "armor_upgrade_level", 150),
                (15, "speed_upgrade_level", 150),
                (16, "vitality_upgrade_level", 200),
            ]
            priorities = (14, 16, 13, 15) if health < 0.55 else (13, 15, 16, 14)
            for action in priorities:
                _, attribute, cost = next(item for item in affordable if item[0] == action)
                if champion.gold >= cost + 125 and getattr(champion, attribute) < 5:
                    return action

        if health < 0.28:
            return 23

        # Protect a threatened home lane before pursuing optional objectives.
        enemy_minions_near_tower = any(
            m.alive and m.team == enemy_team and m.lane == lane
            and m.position.distance_to(own_tower.position) <= own_tower.attack_range * 1.5
            for m in game.minions
        ) if own_tower.alive else False
        if isinstance(self, DefensiveOpponent) and (enemy_minions_near_tower or (enemy_visible and enemy.position.distance_to(own_tower.position) < 500)):
            return 22

        # These choices are intentionally style-specific so the baselines form
        # meaningful opponents instead of all repeating the same macro.
        if isinstance(self, ObjectiveOpponent):
            owned_wards = sum(w.team == team for w in game.wards)
            if objective.alive and owned_wards == 0 and champion.ward_timer <= 0 and champion.gold >= 75:
                return 30
            if objective.alive:
                return 32 if enemy_visible and enemy.position.distance_to(objective.position) < 500 else 19

        if isinstance(self, AggressiveOpponent) and enemy_visible and enemy_distance < 750:
            return 33

        if isinstance(self, BalancedOpponent) and enemy_visible and enemy_distance < champion.attack_range * 1.7:
            if enemy.hp / max(1.0, enemy.max_hp) < 0.38 and health > 0.38:
                return 33
            return 34

        enemy_wards_near = any(
            w.team == enemy_team and champion.position.distance_to(w.position) <= 360
            for w in game.wards
        )
        if enemy_wards_near and champion.sweep_timer <= 0 and champion.gold >= 50:
            return 31

        if objective.alive and objective.position.distance_to(champion.position) < 650:
            if champion.ward_timer <= 0 and champion.gold >= 75 and not any(
                w.team == team and w.position.distance_to(objective.position) < 160
                for w in game.wards
            ):
                return 30
            if isinstance(self, (ObjectiveOpponent, AdaptiveOpponent)):
                return 32 if enemy_visible and enemy.position.distance_to(objective.position) < 500 else 19

        # Push faster with a healthy allied wave; hold a risky wave near home.
        allied_wave = sum(m.alive and m.team == team and m.lane == lane for m in game.minions)
        if health < 0.45 and allied_wave > 0 and own_tower.alive:
            return 27
        if allied_wave >= 5:
            return 29

        # Rotate toward the weakest enemy tower when the current lane is safe.
        enemy_towers = game.lane_towers[enemy_team]
        safest_pressure = min(
            game.map.lanes,
            key=lambda name: enemy_towers[name].hp / max(1.0, enemy_towers[name].max_hp)
            if enemy_towers[name].alive else -1.0,
        )
        if safest_pressure != lane and own_tower.hp / max(1.0, own_tower.max_hp) > 0.4:
            return {"top": 24, "mid": 25, "bottom": 26}[safest_pressure]

        return 18


class AggressiveOpponent(ScriptedOpponent):
    """
    Prioritizes fighting the enemy champion.

    Once the enemy tower is destroyed, it pushes directly
    toward and attacks the enemy base.
    """

    def act(self, game) -> int:
        enemy = game.champions[Team.RED]
        target = game.champions[Team.BLUE]

        if not enemy.alive:
            return 0

        enemy_tower = game.towers[Team.BLUE]
        enemy_base = game.bases[Team.BLUE]

        # ----------------------------------------------------------
        # Finish the game if the tower is already down.
        # ----------------------------------------------------------

        if not enemy_tower.alive:
            distance = enemy.position.distance_to(
                enemy_base.position
            )

            if (
                enemy_base.alive
                and distance <= enemy.attack_range
                and enemy.attack_timer <= 0.0
            ):
                return 11

            return self._move_towards_enemy_base(game)

        # ----------------------------------------------------------
        # Fight the enemy champion when nearby.
        # ----------------------------------------------------------

        distance = enemy.position.distance_to(
            target.position
        )

        if (
            target.alive
            and distance <= enemy.attack_range
            and enemy.attack_timer <= 0.0
        ):
            return 5

        # ----------------------------------------------------------
        # Otherwise move toward the enemy.
        # ----------------------------------------------------------

        return self._move_towards(
            enemy,
            target.position,
            game,
        )

    @staticmethod
    def _move_towards(enemy, target_position, game) -> int:
        dx = target_position.x - enemy.position.x
        dy = target_position.y - enemy.position.y

        if abs(dx) >= abs(dy):
            return 2 if dx > 0 else 1

        return 4 if dy > 0 else 3

    @staticmethod
    def _move_towards_enemy_base(game) -> int:
        enemy = game.champions[Team.RED]
        base = game.bases[Team.BLUE]

        dx = base.position.x - enemy.position.x
        dy = base.position.y - enemy.position.y

        if abs(dx) >= abs(dy):
            return 2 if dx > 0 else 1

        return 4 if dy > 0 else 3


class DefensiveOpponent(ScriptedOpponent):
    """
    Defensive policy.

    Protects its own tower, but once the enemy tower is
    destroyed it transitions into a simple pushing policy.
    """

    def act(self, game) -> int:
        enemy = game.champions[Team.RED]
        target = game.champions[Team.BLUE]

        if not enemy.alive:
            return 0

        own_tower = game.towers[Team.RED]
        enemy_tower = game.towers[Team.BLUE]
        enemy_base = game.bases[Team.BLUE]

        # ----------------------------------------------------------
        # If the enemy tower is down, finish the game.
        # ----------------------------------------------------------

        if not enemy_tower.alive:
            distance = enemy.position.distance_to(
                enemy_base.position
            )

            if (
                enemy_base.alive
                and distance <= enemy.attack_range
                and enemy.attack_timer <= 0.0
            ):
                return 11

            return self._move_towards(
                enemy,
                enemy_base.position,
            )

        # ----------------------------------------------------------
        # Defend own tower.
        # ----------------------------------------------------------

        tower_distance = enemy.position.distance_to(
            own_tower.position
        )

        champion_distance = enemy.position.distance_to(
            target.position
        )

        if (
            target.alive
            and champion_distance <= enemy.attack_range
            and enemy.attack_timer <= 0.0
        ):
            return 5

        # Stay near own tower.
        if tower_distance > 300.0:
            return self._move_towards(
                enemy,
                own_tower.position,
            )

        # Attack enemy tower if it is within range.
        if (
            enemy_tower.alive
            and enemy.position.distance_to(
                enemy_tower.position
            ) <= enemy.attack_range
            and enemy.attack_timer <= 0.0
        ):
            return 6

        return 0

    @staticmethod
    def _move_towards(enemy, target_position) -> int:
        dx = target_position.x - enemy.position.x
        dy = target_position.y - enemy.position.y

        if abs(dx) >= abs(dy):
            return 2 if dx > 0 else 1

        return 4 if dy > 0 else 3


class ObjectiveOpponent(ScriptedOpponent):
    """
    Prioritizes the neutral objective.

    After securing enough map control, it pushes the enemy
    tower and eventually the enemy base.
    """

    def act(self, game) -> int:
        enemy = game.champions[Team.RED]
        target = game.champions[Team.BLUE]
        objective = game.objective

        if not enemy.alive:
            return 0

        enemy_tower = game.towers[Team.BLUE]
        enemy_base = game.bases[Team.BLUE]

        # ----------------------------------------------------------
        # Finish the enemy base if the tower is gone.
        # ----------------------------------------------------------

        if not enemy_tower.alive:
            distance = enemy.position.distance_to(
                enemy_base.position
            )

            if (
                enemy_base.alive
                and distance <= enemy.attack_range
                and enemy.attack_timer <= 0.0
            ):
                return 11

            return self._move_towards(
                enemy,
                enemy_base.position,
            )

        # ----------------------------------------------------------
        # Prioritize objective when alive and nearby.
        # ----------------------------------------------------------

        objective_distance = enemy.position.distance_to(
            objective.position
        )

        if (
            objective.alive
            and objective_distance <= enemy.attack_range
            and enemy.attack_timer <= 0.0
        ):
            return 10

        # Move toward objective if it is alive.
        if objective.alive:
            return self._move_towards(
                enemy,
                objective.position,
            )

        # ----------------------------------------------------------
        # Fight enemy if nearby.
        # ----------------------------------------------------------

        champion_distance = enemy.position.distance_to(
            target.position
        )

        if (
            target.alive
            and champion_distance <= enemy.attack_range
            and enemy.attack_timer <= 0.0
        ):
            return 5

        # ----------------------------------------------------------
        # Push tower.
        # ----------------------------------------------------------

        tower_distance = enemy.position.distance_to(
            enemy_tower.position
        )

        if (
            enemy_tower.alive
            and tower_distance <= enemy.attack_range
            and enemy.attack_timer <= 0.0
        ):
            return 6

        return self._move_towards(
            enemy,
            enemy_tower.position,
        )

    @staticmethod
    def _move_towards(enemy, target_position) -> int:
        dx = target_position.x - enemy.position.x
        dy = target_position.y - enemy.position.y

        if abs(dx) >= abs(dy):
            return 2 if dx > 0 else 1

        return 4 if dy > 0 else 3


class BalancedOpponent(ScriptedOpponent):
    """
    General-purpose baseline.

    Priority:
        1. Finish enemy base.
        2. Fight nearby enemy champion.
        3. Attack enemy tower.
        4. Move toward enemy tower.
    """

    def act(self, game) -> int:
        enemy = game.champions[Team.RED]
        target = game.champions[Team.BLUE]

        if not enemy.alive:
            return 0

        enemy_tower = game.towers[Team.BLUE]
        enemy_base = game.bases[Team.BLUE]

        # ----------------------------------------------------------
        # 1. Finish base.
        # ----------------------------------------------------------

        if not enemy_tower.alive:
            distance = enemy.position.distance_to(
                enemy_base.position
            )

            if (
                enemy_base.alive
                and distance <= enemy.attack_range
                and enemy.attack_timer <= 0.0
            ):
                return 11

            return self._move_towards(
                enemy,
                enemy_base.position,
            )

        # ----------------------------------------------------------
        # 2. Fight nearby champion.
        # ----------------------------------------------------------

        champion_distance = enemy.position.distance_to(
            target.position
        )

        if (
            target.alive
            and champion_distance <= enemy.attack_range
            and enemy.attack_timer <= 0.0
        ):
            return 5

        # ----------------------------------------------------------
        # 3. Attack enemy tower.
        # ----------------------------------------------------------

        tower_distance = enemy.position.distance_to(
            enemy_tower.position
        )

        if (
            enemy_tower.alive
            and tower_distance <= enemy.attack_range
            and enemy.attack_timer <= 0.0
        ):
            return 6

        # ----------------------------------------------------------
        # 4. Push toward enemy tower.
        # ----------------------------------------------------------

        return self._move_towards(
            enemy,
            enemy_tower.position,
        )

    @staticmethod
    def _move_towards(enemy, target_position) -> int:
        dx = target_position.x - enemy.position.x
        dy = target_position.y - enemy.position.y

        if abs(dx) >= abs(dy):
            return 2 if dx > 0 else 1

        return 4 if dy > 0 else 3


class AdaptiveOpponent(ScriptedOpponent):
    """Held-out opponent that switches between skirmishing and objectives."""

    def act(self, game) -> int:
        enemy = game.champions[Team.RED]
        target = game.champions[Team.BLUE]
        if not enemy.alive:
            return 0

        enemy_tower = game.towers[Team.BLUE]
        enemy_base = game.bases[Team.BLUE]
        objective = game.objective

        if not enemy_tower.alive:
            if (
                enemy.position.distance_to(enemy_base.position)
                <= enemy.attack_range
                and enemy.attack_timer <= 0.0
            ):
                return 11
            return BalancedOpponent._move_towards(enemy, enemy_base.position)

        champion_distance = enemy.position.distance_to(target.position)
        if (
            target.alive
            and target.hp <= 0.45 * target.max_hp
            and champion_distance <= enemy.attack_range
            and enemy.attack_timer <= 0.0
        ):
            return 5

        # Contest the objective when it is nearby and the Blue tower is not
        # already at risk of falling; otherwise keep pressure on the lane.
        objective_distance = enemy.position.distance_to(objective.position)
        blue_tower_urgent = enemy_tower.hp <= 0.35 * enemy_tower.max_hp
        if objective.alive and not blue_tower_urgent:
            if objective_distance <= enemy.attack_range and enemy.attack_timer <= 0.0:
                return 10
            return BalancedOpponent._move_towards(enemy, objective.position)

        tower_distance = enemy.position.distance_to(enemy_tower.position)
        if tower_distance <= enemy.attack_range and enemy.attack_timer <= 0.0:
            return 6
        return BalancedOpponent._move_towards(enemy, enemy_tower.position)


OPPONENTS = {
    "aggressive": AggressiveOpponent,
    "defensive": DefensiveOpponent,
    "objective": ObjectiveOpponent,
    "balanced": BalancedOpponent,
    "adaptive": AdaptiveOpponent,
}


def create_opponent(name: str) -> ScriptedOpponent:
    """
    Create a scripted opponent by name.
    """

    name = name.lower().strip()

    if name not in OPPONENTS:
        available = ", ".join(sorted(OPPONENTS))
        raise ValueError(
            f"Unknown opponent '{name}'. "
            f"Available opponents: {available}"
        )

    return OPPONENTS[name]()
