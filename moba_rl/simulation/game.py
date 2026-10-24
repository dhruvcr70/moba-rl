import random
from dataclasses import asdict
from typing import Dict, List

from .lini_tities import (
    Base,
    Champion,
    Entity,
    Minion,
    Objective,
    Position,
    Team,
    Tower,
    VisionWard,
)
from .map import MOBAmap, MapConfig


class Game:
    def __init__(self, seed: int = 42, procedural_map: bool = False,
                 objective_capture_reward: float = 2.0):
        self.seed = seed
        self.rng = random.Random(seed)

        map_config = (
            MapConfig.procedural(self.rng)
            if procedural_map
            else None
        )
        self.map = MOBAmap(map_config)
        self.procedural_map = bool(procedural_map)

        self.time = 0.0
        self.tick_count = 0
        self.next_entity_id = 1

        self.champions: Dict[Team, Champion] = {}
        # ``towers`` is kept as a middle-lane compatibility view for older
        # callers. New simulation systems use all three lane towers.
        self.towers: Dict[Team, Tower] = {}
        self.lane_towers: Dict[Team, Dict[str, Tower]] = {}
        self.bases: Dict[Team, Base] = {}
        self.minions: List[Minion] = []
        self.wards: list[VisionWard] = []
        self.wave_orders = {
            team: {lane: {"mode": "none", "timer": 0.0}
                   for lane in self.map.lanes}
            for team in Team
        }

        self.objective: Objective | None = None

        # ------------------------------------------------------------
        # GAME CONFIGURATION
        # ------------------------------------------------------------

        self.minion_spawn_interval = 10.0
        self.minion_spawn_timer = self.minion_spawn_interval

        self.minion_kill_gold = 25
        self.minion_kill_xp = 30.0

        self.champion_kill_gold = 300
        self.champion_kill_xp = 200.0

        self.tower_destroy_gold = 150
        self.tower_destroy_xp = 100.0

        # Base combat
        self.base_champion_damage = 50.0
        self.base_attack_range = 120.0
        self.shop_range = 160.0
        self.objective_respawn_interval = 60.0
        self.objective_respawn_timer: float | None = None
        self.objective_capture_reward = float(objective_capture_reward)
        if self.objective_capture_reward < 0.0:
            raise ValueError("objective_capture_reward must be non-negative")

        # ------------------------------------------------------------
        # EVENTS / REWARDS
        # ------------------------------------------------------------

        self.events: List[str] = []

        self.reward_events: Dict[Team, float] = {
            Team.BLUE: 0.0,
            Team.RED: 0.0,
        }

        self.winner: Team | None = None
        self.game_over = False

        self._create_world()

    # ================================================================
    # WORLD
    # ================================================================

    def _get_id(self) -> int:
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        return entity_id

    def _create_world(self) -> None:
        for team in (Team.BLUE, Team.RED):
            spawn = self.map.spawn_position(team)

            self.champions[team] = Champion(
                id=self._get_id(),
                team=team,
                position=Position(
                    spawn.x,
                    spawn.y,
                ),
                max_hp=1000.0,
                hp=1000.0,
            )

            self.lane_towers[team] = {}
            for lane in self.map.lanes:
                tower_position = self.map.tower_position(team, lane)
                self.lane_towers[team][lane] = Tower(
                    id=self._get_id(),
                    team=team,
                    position=Position(tower_position.x, tower_position.y),
                    max_hp=1500.0,
                    hp=1500.0,
                    lane=lane,
                )
            self.towers[team] = self.lane_towers[team]["mid"]

            base_position = self.map.base_position(team)

            self.bases[team] = Base(
                id=self._get_id(),
                team=team,
                position=Position(
                    base_position.x,
                    base_position.y,
                ),
                max_hp=5000.0,
                hp=5000.0,
            )

        objective_position = self.map.objective_position()

        self.objective = Objective(
            id=self._get_id(),
            team=Team.BLUE,
            position=Position(
                objective_position.x,
                objective_position.y,
            ),
            max_hp=2000.0,
            hp=2000.0,
        )

    # ================================================================
    # MAIN LOOP
    # ================================================================

    def step(self, dt: float = 0.1) -> None:
        if self.game_over:
            return

        self.time += dt
        self.tick_count += 1

        self._update_entities(dt)
        self._update_wave_orders(dt)
        self._update_wards(dt)
        self._update_objective_respawn(dt)

        self._update_minion_spawning(dt)
        self._update_minions(dt)

        # Minions first fight each other/champions.
        self._update_minion_combat()

        # Then minions attack structures.
        self._update_minion_tower_attacks()
        self._update_minion_base_attacks()

        # Towers defend their side.
        self._update_tower_combat()

        self._update_respawns()

        # Check whether a base has fallen.
        self._update_game_rules()

        self._cleanup_dead_minions()

    # ================================================================
    # CHAMPION MOVEMENT
    # ================================================================

    def move_champion(
        self,
        team: Team,
        target: Position,
        dt: float = 0.1,
    ) -> None:
        champion = self.champions[team]

        champion.move_towards(
            target=target,
            dt=dt,
            map_width=self.map.config.width,
            map_height=self.map.config.height,
        )

    def lane_for_position(self, position: Position) -> str:
        return min(
            self.map.lanes,
            key=lambda lane: abs(position.y - self.map.lane_y(lane)),
        )

    def lane_for_team(self, team: Team) -> str:
        return self.lane_for_position(self.champions[team].position)

    def tower_for_lane(self, team: Team, lane: str = "mid") -> Tower:
        """Get a team's tower on a lane, falling back to mid for old configs."""
        return self.lane_towers.get(team, {}).get(lane, self.towers[team])

    def set_wave_order(self, team: Team, lane: str, mode: str) -> bool:
        if lane not in self.map.lanes or mode not in {"freeze", "slow_push", "fast_push"}:
            return False
        current = self.wave_orders[team][lane]
        if current["mode"] == mode and current["timer"] > 0.0:
            return True
        previous = current["mode"]
        current.update(mode=mode, timer=20.0)
        if previous != mode:
            self.events.append(
                f"{team.value.upper()} called a {mode.replace('_', ' ')} in {lane} lane."
            )
        return True

    def wave_order_for(self, team: Team, lane: str) -> str:
        return self.wave_orders[team][lane]["mode"]

    def _update_wave_orders(self, dt: float) -> None:
        for team, lanes in self.wave_orders.items():
            for lane, order in lanes.items():
                if order["timer"] <= 0.0:
                    continue
                order["timer"] = max(0.0, order["timer"] - dt)
                if order["timer"] == 0.0:
                    previous = order["mode"]
                    order["mode"] = "none"
                    if previous != "none":
                        self.events.append(
                            f"{team.value.upper()} {lane} lane wave returned to normal."
                        )

    def place_ward(self, team: Team, position: Position) -> bool:
        champion = self.champions[team]
        if not champion.alive or champion.ward_timer > 0.0 or champion.gold < 75:
            return False
        champion.gold -= 75
        champion.ward_timer = champion.ward_cooldown
        owned = [ward for ward in self.wards if ward.team == team]
        if len(owned) >= 2:
            self.wards.remove(min(owned, key=lambda ward: ward.duration))
        self.wards.append(VisionWard(
            id=self._get_id(), team=team,
            position=Position(position.x, position.y),
        ))
        self.events.append(f"{team.value.upper()} placed a vision ward near the objective.")
        return True

    def sweep_wards(self, team: Team, radius: float = 360.0) -> int | None:
        champion = self.champions[team]
        if not champion.alive or champion.sweep_timer > 0.0 or champion.gold < 50:
            return None
        champion.gold -= 50
        champion.sweep_timer = champion.sweep_cooldown
        enemy_team = self.enemy_team(team)
        before = len(self.wards)
        self.wards = [
            ward for ward in self.wards
            if ward.team != enemy_team
            or champion.position.distance_to(ward.position) > radius
        ]
        removed = before - len(self.wards)
        self.events.append(
            f"{team.value.upper()} swept enemy vision ({removed} ward{'s' if removed != 1 else ''} removed)."
        )
        return removed

    # ================================================================
    # BASIC ATTACK
    # ================================================================

    def attack_champion(
        self,
        attacker_team: Team,
        target_team: Team,
    ) -> bool:
        attacker = self.champions[attacker_team]
        target = self.champions[target_team]

        if not attacker.alive:
            return False

        if not target.alive:
            return False

        if (
            attacker.position.distance_to(target.position)
            > attacker.attack_range
        ):
            return False

        if attacker.attack_timer > 0.0:
            return False

        target.take_damage(attacker.attack_damage)

        attacker.attack_timer = attacker.attack_cooldown

        self.events.append(
            f"{attacker_team.value.upper()} "
            f"attacked "
            f"{target_team.value.upper()} "
            f"for {attacker.attack_damage:.1f} damage."
        )

        if not target.alive:
            self._champion_killed(
                killer=attacker,
                victim=target,
            )

        return True

    # ================================================================
    # ABILITIES
    # ================================================================

    def use_ability(
        self,
        attacker_team: Team,
        ability: int,
        target_team: Team,
    ) -> bool:
        attacker = self.champions[attacker_team]
        target = self.champions[target_team]

        if not attacker.alive:
            return False

        if not target.alive:
            return False

        if ability == 1:
            damage = 100.0
            cooldown = attacker.ability_1_cooldown

            if attacker.ability_1_timer > 0.0:
                return False

        elif ability == 2:
            damage = 175.0
            cooldown = attacker.ability_2_cooldown

            if attacker.ability_2_timer > 0.0:
                return False

        elif ability == 3:
            damage = 300.0
            cooldown = attacker.ability_3_cooldown

            if attacker.ability_3_timer > 0.0:
                return False

        else:
            raise ValueError("ability must be 1, 2, or 3")

        if (
            attacker.position.distance_to(target.position)
            > attacker.attack_range * 1.5
        ):
            return False

        target.take_damage(damage)

        if ability == 1:
            attacker.ability_1_timer = cooldown
        elif ability == 2:
            attacker.ability_2_timer = cooldown
        else:
            attacker.ability_3_timer = cooldown

        self.events.append(
            f"{attacker_team.value.upper()} "
            f"used ability {ability} "
            f"on {target_team.value.upper()} "
            f"for {damage:.1f} damage."
        )

        if not target.alive:
            self._champion_killed(
                killer=attacker,
                victim=target,
            )

        return True

    # ================================================================
    # CHAMPION FARMING AND SHOP
    # ================================================================

    def attack_minion(self, attacker_team: Team) -> bool:
        attacker = self.champions[attacker_team]
        if not attacker.alive or attacker.attack_timer > 0.0:
            return False

        candidates = [
            minion for minion in self.minions
            if minion.alive
            and minion.team != attacker_team
            and minion.lane == self.lane_for_team(attacker_team)
            and attacker.position.distance_to(minion.position)
            <= attacker.attack_range
        ]
        if not candidates:
            return False

        target = min(
            candidates,
            key=lambda minion: attacker.position.distance_to(minion.position),
        )
        target.take_damage(attacker.attack_damage)
        attacker.attack_timer = attacker.attack_cooldown
        self.events.append(
            f"{attacker_team.value.upper()} champion attacked a "
            f"{target.team.value.upper()} minion for "
            f"{attacker.attack_damage:.1f} damage."
        )
        if not target.alive:
            self._minion_killed(target, killer_team=attacker_team)
        return True

    def purchase_upgrade(self, team: Team, upgrade: str) -> bool:
        champion = self.champions[team]
        base = self.bases[team]
        if (
            not champion.alive
            or champion.position.distance_to(base.position) > self.shop_range
        ):
            return False

        upgrades = {
            "blade": (200, "attack_upgrade_level", 5),
            "armor": (150, "armor_upgrade_level", 5),
            "boots": (150, "speed_upgrade_level", 5),
            "vitality": (200, "vitality_upgrade_level", 5),
        }
        if upgrade not in upgrades:
            raise ValueError(f"Unknown upgrade: {upgrade}")

        cost, rank_attribute, max_rank = upgrades[upgrade]
        rank = getattr(champion, rank_attribute)
        if champion.gold < cost or rank >= max_rank:
            return False

        champion.gold -= cost
        setattr(champion, rank_attribute, rank + 1)
        if upgrade == "blade":
            champion.attack_damage += 10.0
        elif upgrade == "armor":
            champion.armor += 5.0
        elif upgrade == "boots":
            champion.move_speed += 10.0
        else:
            champion.max_hp += 150.0
            champion.hp = min(champion.max_hp, champion.hp + 150.0)

        self.events.append(
            f"{team.value.upper()} bought {upgrade} upgrade (rank {rank + 1})."
        )
        return True

    # ================================================================
    # CHAMPION VS TOWER
    # ================================================================

    def attack_tower(
        self,
        attacker_team: Team,
        target_team: Team,
        lane: str | None = None,
    ) -> bool:
        attacker = self.champions[attacker_team]
        lane = lane or self.lane_for_position(attacker.position)
        tower = self.tower_for_lane(target_team, lane)

        if not attacker.alive:
            return False

        if not tower.alive:
            return False

        if (
            attacker.position.distance_to(tower.position)
            > attacker.attack_range
        ):
            return False

        if attacker.attack_timer > 0.0:
            return False

        tower.take_damage(attacker.attack_damage)

        attacker.attack_timer = attacker.attack_cooldown

        self.events.append(
            f"{attacker_team.value.upper()} "
            f"champion attacked "
            f"{target_team.value.upper()} {lane} lane tower "
            f"for {attacker.attack_damage:.1f} damage."
        )

        if not tower.alive:
            self._tower_destroyed(
                attacker_team,
                target_team,
                lane,
            )

        return True

    # ================================================================
    # CHAMPION VS BASE
    # ================================================================

    def attack_base(
        self,
        attacker_team: Team,
        target_team: Team,
    ) -> bool:
        """
        Allow a champion to directly attack the enemy base.

        The enemy tower must already be destroyed before the base
        becomes vulnerable.
        """
        attacker = self.champions[attacker_team]
        base = self.bases[target_team]
        lane = self.lane_for_position(attacker.position)
        enemy_tower = self.tower_for_lane(target_team, lane)

        if self.game_over:
            return False

        if not attacker.alive:
            return False

        if not enemy_tower.alive:
            pass
        else:
            # Cannot attack the base while its tower is alive.
            return False

        if not base.alive:
            return False

        if (
            attacker.position.distance_to(base.position)
            > self.base_attack_range
        ):
            return False

        if attacker.attack_timer > 0.0:
            return False

        base.take_damage(self.base_champion_damage)

        attacker.attack_timer = attacker.attack_cooldown

        self.events.append(
            f"{attacker_team.value.upper()} "
            f"champion attacked "
            f"{target_team.value.upper()} base "
            f"for {self.base_champion_damage:.1f} damage."
        )

        if not base.alive:
            self._base_destroyed(
                attacker_team,
                target_team,
            )

        return True

    # ================================================================
    # TOWER DESTRUCTION
    # ================================================================

    def _tower_destroyed(
        self,
        attacker_team: Team,
        target_team: Team,
        lane: str = "mid",
    ) -> None:
        attacker = self.champions[attacker_team]

        attacker.gold += self.tower_destroy_gold

        leveled_up = attacker.gain_xp(
            self.tower_destroy_xp
        )

        self.reward_events[attacker_team] += 1.0

        self.events.append(
            f"{target_team.value.upper()} {lane} lane tower "
            f"was destroyed by "
            f"{attacker_team.value.upper()}."
        )

        if leveled_up:
            self.events.append(
                f"{attacker_team.value.upper()} "
                f"leveled up to "
                f"{attacker.level}."
            )

    # ================================================================
    # BASE DESTRUCTION
    # ================================================================

    def _base_destroyed(
        self,
        attacker_team: Team,
        target_team: Team,
    ) -> None:
        """
        Resolve a base destruction immediately.

        Terminal rewards are applied here rather than waiting for a
        later tick, while _update_game_rules remains a safety check.
        """
        if self.game_over:
            return

        self.winner = attacker_team
        self.game_over = True

        # Large terminal reward.
        self.reward_events[attacker_team] += 10.0
        self.reward_events[target_team] -= 10.0

        self.events.append(
            f"{target_team.value.upper()} base "
            f"was destroyed by "
            f"{attacker_team.value.upper()}."
        )

        self.events.append(
            f"{attacker_team.value.upper()} wins!"
        )

    # ================================================================
    # MINION SPAWNING
    # ================================================================

    def _spawn_minion_wave(
        self,
        team: Team,
    ) -> None:
        for lane in self.map.lanes:
            spawn = self.map.spawn_position(team, lane)
            for _ in range(3):
                offset = self.rng.uniform(-8.0, 8.0)
                self.minions.append(
                    Minion(
                        id=self._get_id(),
                        team=team,
                        position=Position(spawn.x, spawn.y + offset),
                        max_hp=300.0,
                        hp=300.0,
                        lane=lane,
                    )
                )

        self.events.append(
            f"{team.value.upper()} minion wave spawned."
        )

    def _update_minion_spawning(
        self,
        dt: float,
    ) -> None:
        self.minion_spawn_timer -= dt

        if self.minion_spawn_timer > 0:
            return

        self.minion_spawn_timer += (
            self.minion_spawn_interval
        )

        self._spawn_minion_wave(Team.BLUE)
        self._spawn_minion_wave(Team.RED)

    # ================================================================
    # MINION MOVEMENT
    # ================================================================

    def _update_minions(
        self,
        dt: float,
    ) -> None:
        for minion in self.minions:
            if not minion.alive:
                continue

            enemy_team = self.enemy_team(
                minion.team
            )

            enemy_tower = self.tower_for_lane(enemy_team, minion.lane)
            enemy_base = self.bases[enemy_team]

            # Push tower first.
            if enemy_tower.alive:
                target = enemy_tower.position

            # Once tower is destroyed, push the base.
            elif enemy_base.alive:
                target = enemy_base.position

            else:
                continue

            dx = target.x - minion.position.x
            dy = target.y - minion.position.y

            distance = (
                dx * dx + dy * dy
            ) ** 0.5

            if distance == 0:
                continue

            mode = self.wave_order_for(minion.team, minion.lane)
            speed_multiplier = {
                "none": 1.0,
                "freeze": 0.0,
                "slow_push": 0.65,
                "fast_push": 1.75,
            }.get(mode, 1.0)
            movement = minion.move_speed * speed_multiplier * dt

            if distance <= movement:
                minion.position.x = target.x
                minion.position.y = target.y
                continue

            minion.position.x += (
                dx / distance
            ) * movement

            minion.position.y += (
                dy / distance
            ) * movement

    # ================================================================
    # MINION COMBAT
    # ================================================================

    def _update_minion_combat(self) -> None:
        for minion in self.minions:
            if not minion.alive:
                continue

            enemy_minion = (
                self._find_nearest_enemy_minion(
                    minion
                )
            )

            if enemy_minion is not None:
                distance = (
                    minion.position.distance_to(
                        enemy_minion.position
                    )
                )

                if (
                    distance <= minion.attack_range
                    and minion.attack_timer <= 0
                ):
                    self._minion_attack_minion(
                        minion,
                        enemy_minion,
                    )

                    continue

            enemy_champion = (
                self._find_enemy_champion(
                    minion
                )
            )

            if enemy_champion is not None:
                distance = (
                    minion.position.distance_to(
                        enemy_champion.position
                    )
                )

                if (
                    distance <= minion.attack_range
                    and minion.attack_timer <= 0
                ):
                    self._minion_attack_champion(
                        minion,
                        enemy_champion,
                    )

    def _find_enemy_champion(
        self,
        attacker: Minion,
    ) -> Champion | None:
        enemy = self.champions[
            self.enemy_team(attacker.team)
        ]

        if not enemy.alive:
            return None

        distance = (
            attacker.position.distance_to(
                enemy.position
            )
        )

        if (
            distance <= 150.0
            and self.lane_for_position(enemy.position) == attacker.lane
        ):
            return enemy

        return None

    def _find_nearest_enemy_minion(
        self,
        attacker: Minion,
    ) -> Minion | None:
        nearest = None
        nearest_distance = float("inf")

        for candidate in self.minions:
            if not candidate.alive:
                continue

            if candidate.team == attacker.team:
                continue

            if candidate.lane != attacker.lane:
                continue

            distance = (
                attacker.position.distance_to(
                    candidate.position
                )
            )

            if distance < nearest_distance:
                nearest = candidate
                nearest_distance = distance

        return nearest

    def _minion_attack_minion(
        self,
        attacker: Minion,
        target: Minion,
    ) -> None:
        damage_multiplier = 1.2 if self.wave_order_for(attacker.team, attacker.lane) == "fast_push" else 1.0
        target.take_damage(attacker.attack_damage * damage_multiplier)

        attacker.attack_timer = (
            attacker.attack_cooldown
        )

        if not target.alive:
            self._minion_killed(
                target,
                killer_team=attacker.team,
            )

    def _minion_attack_champion(
        self,
        attacker: Minion,
        target: Champion,
    ) -> None:
        target.take_damage(
            attacker.attack_damage
        )

        attacker.attack_timer = (
            attacker.attack_cooldown
        )

        self.events.append(
            f"{attacker.team.value.upper()} minion attacked "
            f"{target.team.value.upper()} champion for "
            f"{attacker.attack_damage:.1f} damage."
        )

        if not target.alive:
            target.deaths += 1
            target.respawn_timer = 5.0

            self.events.append(
                f"{target.team.value.upper()} "
                f"champion was killed by minions."
            )

    def _minion_killed(
        self,
        minion: Minion,
        killer_team: Team | None = None,
    ) -> None:
        self.events.append(
            f"{minion.team.value.upper()} "
            f"minion died."
        )

        # If the killer is not specified, preserve the old fallback
        # behavior for compatibility with existing tests/callers.
        if killer_team is None:
            killer_team = self.enemy_team(
                minion.team
            )

        killer = self.champions[killer_team]

        killer.gold += self.minion_kill_gold
        killer.minion_kills += 1

        leveled_up = killer.gain_xp(
            self.minion_kill_xp
        )

        self.reward_events[killer_team] += 0.1

        if leveled_up:
            self.events.append(
                f"{killer_team.value.upper()} "
                f"leveled up to "
                f"{killer.level}."
            )

    # ================================================================
    # MINION VS TOWER
    # ================================================================

    def _update_minion_tower_attacks(
        self,
    ) -> None:
        for minion in self.minions:
            if not minion.alive:
                continue

            enemy_team = self.enemy_team(
                minion.team
            )

            tower = self.tower_for_lane(enemy_team, minion.lane)

            if not tower.alive:
                continue

            distance = (
                minion.position.distance_to(
                    tower.position
                )
            )

            if distance > minion.attack_range:
                continue

            if minion.attack_timer > 0:
                continue

            damage_multiplier = 1.2 if self.wave_order_for(minion.team, minion.lane) == "fast_push" else 1.0
            tower.take_damage(minion.attack_damage * damage_multiplier)

            minion.attack_timer = (
                minion.attack_cooldown
            )

            self.events.append(
                f"{minion.team.value.upper()} minion attacked "
                f"{enemy_team.value.upper()} {minion.lane} lane tower "
                f"for {minion.attack_damage:.1f} damage."
            )

            if not tower.alive:
                self._tower_destroyed(minion.team, enemy_team, tower.lane)

    # ================================================================
    # MINION VS BASE
    # ================================================================

    def _update_minion_base_attacks(
        self,
    ) -> None:
        """
        Minions attack the enemy base after the enemy tower has
        been destroyed.
        """
        for minion in self.minions:
            if not minion.alive:
                continue

            enemy_team = self.enemy_team(
                minion.team
            )

            tower = self.tower_for_lane(enemy_team, minion.lane)
            base = self.bases[enemy_team]

            # The tower must fall before the base can be damaged.
            if tower.alive:
                continue

            if not base.alive:
                continue

            distance = (
                minion.position.distance_to(
                    base.position
                )
            )

            if distance > minion.attack_range:
                continue

            if minion.attack_timer > 0:
                continue

            damage_multiplier = 1.2 if self.wave_order_for(minion.team, minion.lane) == "fast_push" else 1.0
            base.take_damage(minion.attack_damage * damage_multiplier)

            minion.attack_timer = (
                minion.attack_cooldown
            )

            self.events.append(
                f"{minion.team.value.upper()} "
                f"minion attacked "
                f"{enemy_team.value.upper()} base "
                f"for {minion.attack_damage:.1f} damage."
            )

            if not base.alive:
                self._base_destroyed(
                    minion.team,
                    enemy_team,
                )

                return

    # ================================================================
    # TOWER DEFENSE
    # ================================================================

    def _update_tower_combat(self) -> None:
        for team_towers in self.lane_towers.values():
            for tower in team_towers.values():
                if not tower.alive:
                    continue

                if tower.attack_timer > 0:
                    continue

                target = self._find_tower_target(tower)

                if target is None:
                    continue

                target.take_damage(tower.attack_damage)

                tower.attack_timer = tower.attack_cooldown

                self.events.append(
                    f"{tower.team.value.upper()} {tower.lane} lane tower "
                    f"attacked {target.team.value.upper()} "
                    f"{type(target).__name__.lower()} for "
                    f"{tower.attack_damage:.1f} damage."
                )

                if not target.alive:
                    if isinstance(target, Champion):
                        target.deaths += 1
                        target.respawn_timer = 5.0

                        self.events.append(
                            f"{target.team.value.upper()} champion was killed by "
                            f"{tower.team.value.upper()} {tower.lane} lane tower."
                        )

                    elif isinstance(target, Minion):
                        self._minion_killed(target, killer_team=tower.team)

    def _find_tower_target(
        self,
        tower: Tower,
    ) -> Entity | None:
        enemy_team = self.enemy_team(
            tower.team
        )

        champion = self.champions[
            enemy_team
        ]

        # Champions have priority.
        if (
            champion.alive
            and self.lane_for_position(champion.position) == tower.lane
            and tower.position.distance_to(
                champion.position
            )
            <= tower.attack_range
        ):
            return champion

        nearest = None
        nearest_distance = float("inf")

        for minion in self.minions:
            if not minion.alive:
                continue

            if minion.team != enemy_team:
                continue

            if minion.lane != tower.lane:
                continue

            distance = (
                tower.position.distance_to(
                    minion.position
                )
            )

            if (
                distance <= tower.attack_range
                and distance < nearest_distance
            ):
                nearest = minion
                nearest_distance = distance

        return nearest

    # ================================================================
    # OBJECTIVE
    # ================================================================

    def attack_objective(
        self,
        attacker_team: Team,
    ) -> bool:
        if self.objective is None:
            return False

        objective = self.objective

        champion = self.champions[
            attacker_team
        ]

        if not champion.alive:
            return False

        if not objective.alive:
            return False

        if (
            champion.position.distance_to(
                objective.position
            )
            > champion.attack_range
        ):
            return False

        if champion.attack_timer > 0:
            return False

        objective.take_damage(
            champion.attack_damage
        )

        champion.attack_timer = (
            champion.attack_cooldown
        )

        self.events.append(
            f"{attacker_team.value.upper()} attacked the neutral objective "
            f"for {champion.attack_damage:.1f} damage."
        )

        if not objective.alive:
            champion.gold += (
                objective.reward_gold
            )

            champion.gain_xp(
                objective.reward_xp
            )

            champion.objectives_taken += 1
            self.objective_respawn_timer = self.objective_respawn_interval

            self.reward_events[
                attacker_team
            ] += self.objective_capture_reward

            self.events.append(
                f"{attacker_team.value.upper()} "
                f"captured the neutral objective."
            )

        return True

    def _update_objective_respawn(self, dt: float) -> None:
        if self.game_over or self.objective is None or self.objective.alive:
            return
        if self.objective_respawn_timer is None:
            return

        self.objective_respawn_timer = max(
            0.0,
            self.objective_respawn_timer - dt,
        )
        if self.objective_respawn_timer > 0.0:
            return

        current = self.objective.position
        candidates = [
            position
            for position in self.map.objective_spawn_positions()
            if position.distance_to(current) > 50.0
        ]
        if not candidates:
            candidates = self.map.objective_spawn_positions()
        self.objective.position = self.rng.choice(candidates)
        self.objective.hp = self.objective.max_hp
        self.objective.attack_timer = 0.0
        self.objective_respawn_timer = None
        self.events.append("The neutral objective respawned at a new site.")

    # ================================================================
    # KILLS
    # ================================================================

    def _champion_killed(
        self,
        killer: Champion,
        victim: Champion,
    ) -> None:
        killer.kills += 1
        victim.deaths += 1

        victim.respawn_timer = 5.0

        killer.gold += (
            self.champion_kill_gold
        )

        leveled_up = killer.gain_xp(
            self.champion_kill_xp
        )

        self.reward_events[
            killer.team
        ] += 1.0

        self.reward_events[
            victim.team
        ] -= 1.0

        self.events.append(
            f"{killer.team.value.upper()} "
            f"killed "
            f"{victim.team.value.upper()}."
        )

        if leveled_up:
            self.events.append(
                f"{killer.team.value.upper()} "
                f"leveled up to "
                f"{killer.level}."
            )

    # ================================================================
    # RESPAWN
    # ================================================================

    def _update_respawns(self) -> None:
        for team, champion in self.champions.items():
            if champion.alive:
                continue

            if champion.respawn_timer > 0:
                continue

            # Never respawn after the game has ended.
            if self.game_over:
                continue

            spawn = self.map.spawn_position(
                team
            )

            champion.hp = champion.max_hp

            champion.position = Position(
                spawn.x,
                spawn.y,
            )

            self.events.append(
                f"{team.value.upper()} "
                f"champion respawned."
            )

    # ================================================================
    # GAME RULES
    # ================================================================

    def _update_game_rules(self) -> None:
        if self.game_over:
            return

        for team, base in self.bases.items():
            if not base.alive:
                self._base_destroyed(
                    self.enemy_team(team),
                    team,
                )
                return

    # ================================================================
    # STATE / REWARD
    # ================================================================

    def get_reward(
        self,
        team: Team,
    ) -> float:
        reward = self.reward_events[team]

        self.reward_events[team] = 0.0

        return reward

    def get_state(self) -> dict:
        state = {
            "time": self.time,
            "tick": self.tick_count,
            "game_over": self.game_over,
            "winner": (
                self.winner.value
                if self.winner
                else None
            ),
            "champions": {},
            "towers": {},
            "lane_towers": {},
            "bases": {},
            "minions": len(self.minions),
            "minion_entities": [
                {
                    "team": minion.team.value,
                    "x": minion.position.x,
                    "y": minion.position.y,
                    "hp": minion.hp,
                    "max_hp": minion.max_hp,
                    "lane": minion.lane,
                }
                for minion in self.minions
                if minion.alive
            ],
            "objective": None,
            "objective_respawn_timer": self.objective_respawn_timer,
            "procedural_map": self.procedural_map,
            "map": asdict(self.map.config),
            "wards": [
                {"team": ward.team.value, "x": ward.position.x,
                 "y": ward.position.y, "duration": ward.duration,
                 "max_duration": ward.max_duration}
                for ward in self.wards
            ],
            "wave_orders": {
                team.value: {lane: dict(order) for lane, order in lanes.items()}
                for team, lanes in self.wave_orders.items()
            },
        }

        for team in Team:
            champion = self.champions[team]
            tower = self.towers[team]
            base = self.bases[team]

            state["champions"][
                team.value
            ] = {
                "hp": champion.hp,
                "max_hp": champion.max_hp,
                "x": champion.position.x,
                "y": champion.position.y,
                "level": champion.level,
                "xp": champion.xp,
                "gold": champion.gold,
                "kills": champion.kills,
                "deaths": champion.deaths,
                "alive": champion.alive,
                "respawn_timer": champion.respawn_timer,
                "attack_damage": champion.attack_damage,
                "armor": champion.armor,
                "move_speed": champion.move_speed,
                "lane": self.lane_for_team(team),
            }

            state["towers"][
                team.value
            ] = {
                "hp": tower.hp,
                "max_hp": tower.max_hp,
                "x": tower.position.x,
                "y": tower.position.y,
            }
            state["lane_towers"][team.value] = {
                lane: {
                    "hp": lane_tower.hp,
                    "max_hp": lane_tower.max_hp,
                    "x": lane_tower.position.x,
                    "y": lane_tower.position.y,
                    "alive": lane_tower.alive,
                }
                for lane, lane_tower in self.lane_towers[team].items()
            }

            state["bases"][
                team.value
            ] = {
                "hp": base.hp,
                "max_hp": base.max_hp,
                "x": base.position.x,
                "y": base.position.y,
            }

        if self.objective is not None:
            state["objective"] = {
                "hp": self.objective.hp,
                "max_hp": self.objective.max_hp,
                "x": self.objective.position.x,
                "y": self.objective.position.y,
                "alive": self.objective.alive,
            }

        return state

    # ================================================================
    # UTILITIES
    # ================================================================

    def _cleanup_dead_minions(self) -> None:
        self.minions = [
            minion
            for minion in self.minions
            if minion.alive
        ]

    def _update_entities(
        self,
        dt: float,
    ) -> None:
        for champion in self.champions.values():
            champion.update(dt)

        for team_towers in self.lane_towers.values():
            for tower in team_towers.values():
                tower.update(dt)

        for minion in self.minions:
            minion.update(dt)

        if self.objective is not None:
            self.objective.update(dt)

    def _update_wards(self, dt: float) -> None:
        expired = []
        for ward in self.wards:
            ward.duration = max(0.0, ward.duration - dt)
            if ward.duration == 0.0:
                expired.append(ward)
        for ward in expired:
            self.wards.remove(ward)
            self.events.append(f"{ward.team.value.upper()} vision ward expired.")

    @staticmethod
    def enemy_team(
        team: Team,
    ) -> Team:
        if team == Team.BLUE:
            return Team.RED

        return Team.BLUE

    def get_entities(self) -> List[Entity]:
        entities: List[Entity] = []

        entities.extend(
            self.champions.values()
        )

        entities.extend(
            tower
            for team_towers in self.lane_towers.values()
            for tower in team_towers.values()
        )

        entities.extend(
            self.bases.values()
        )

        entities.extend(
            self.minions
        )

        if self.objective is not None:
            entities.append(
                self.objective
            )

        return entities

    def print_state(self) -> None:
        print(
            f"\nTime: {self.time:.1f}s"
        )

        print(
            f"Tick: {self.tick_count}"
        )

        print(
            f"Minions: {len(self.minions)}"
        )

        for team in Team:
            champion = self.champions[team]
            tower = self.towers[team]
            base = self.bases[team]

            print(
                f"{team.value.upper():5} | "
                f"HP: {champion.hp:7.1f} | "
                f"Level: {champion.level} | "
                f"XP: {champion.xp:6.1f} | "
                f"Gold: {champion.gold:5d} | "
                f"Tower: {tower.hp:7.1f} | "
                f"Base: {base.hp:7.1f}"
            )
