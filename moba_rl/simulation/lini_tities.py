from dataclasses import dataclass
from enum import Enum


class Team(Enum):
    BLUE = "blue"
    RED = "red"


@dataclass
class Position:
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5


@dataclass
class Entity:
    id: int
    team: Team
    position: Position
    max_hp: float
    hp: float

    @property
    def alive(self) -> bool:
        return self.hp > 0

    def take_damage(self, amount: float) -> None:
        self.hp = max(0.0, self.hp - amount)


@dataclass
class Champion(Entity):
    attack_damage: float = 50.0
    attack_range: float = 120.0
    move_speed: float = 100.0
    armor: float = 0.0
    attack_upgrade_level: int = 0
    armor_upgrade_level: int = 0
    speed_upgrade_level: int = 0
    vitality_upgrade_level: int = 0

    attack_cooldown: float = 1.0
    attack_timer: float = 0.0

    level: int = 1
    xp: float = 0.0
    gold: int = 500

    kills: int = 0
    deaths: int = 0
    minion_kills: int = 0
    objectives_taken: int = 0

    respawn_timer: float = 0.0

    ability_1_cooldown: float = 5.0
    ability_1_timer: float = 0.0

    ability_2_cooldown: float = 8.0
    ability_2_timer: float = 0.0

    ability_3_cooldown: float = 12.0
    ability_3_timer: float = 0.0

    ward_cooldown: float = 45.0
    ward_timer: float = 0.0
    sweep_cooldown: float = 30.0
    sweep_timer: float = 0.0

    def take_damage(self, amount: float) -> None:
        if amount <= 0.0:
            return
        reduced_damage = max(1.0, amount - self.armor)
        super().take_damage(reduced_damage)

    def update(self, dt: float) -> None:
        self.attack_timer = max(
            0.0,
            self.attack_timer - dt,
        )

        self.ability_1_timer = max(
            0.0,
            self.ability_1_timer - dt,
        )

        self.ability_2_timer = max(
            0.0,
            self.ability_2_timer - dt,
        )

        self.ability_3_timer = max(
            0.0,
            self.ability_3_timer - dt,
        )
        self.ward_timer = max(0.0, self.ward_timer - dt)
        self.sweep_timer = max(0.0, self.sweep_timer - dt)

        if not self.alive:
            self.respawn_timer = max(
                0.0,
                self.respawn_timer - dt,
            )

    def move_towards(
        self,
        target: Position,
        dt: float,
        map_width: float,
        map_height: float,
    ) -> None:

        if not self.alive:
            return

        dx = target.x - self.position.x
        dy = target.y - self.position.y

        distance = (dx * dx + dy * dy) ** 0.5

        if distance == 0:
            return

        max_distance = self.move_speed * dt

        if distance <= max_distance:
            self.position.x = target.x
            self.position.y = target.y
            return

        direction_x = dx / distance
        direction_y = dy / distance

        self.position.x += direction_x * max_distance
        self.position.y += direction_y * max_distance

        self.position.x = max(
            0.0,
            min(map_width, self.position.x),
        )

        self.position.y = max(
            0.0,
            min(map_height, self.position.y),
        )

    def xp_to_next_level(self) -> float:
        return 100.0 * self.level

    def gain_xp(self, amount: float) -> bool:
        self.xp += amount
        leveled_up = False

        while self.xp >= self.xp_to_next_level():
            self.xp -= self.xp_to_next_level()
            self.level += 1

            self.max_hp += 100.0
            self.hp = min(
                self.max_hp,
                self.hp + 100.0,
            )

            self.attack_damage += 10.0

            leveled_up = True

        return leveled_up


@dataclass
class Tower(Entity):
    lane: str = "mid"
    attack_damage: float = 80.0
    attack_range: float = 250.0

    attack_cooldown: float = 1.0
    attack_timer: float = 0.0

    def update(self, dt: float) -> None:
        self.attack_timer = max(
            0.0,
            self.attack_timer - dt,
        )


@dataclass
class Base(Entity):
    pass


@dataclass
class Minion(Entity):
    lane: str = "mid"
    attack_damage: float = 10.0
    attack_range: float = 40.0
    move_speed: float = 60.0

    attack_cooldown: float = 1.5
    attack_timer: float = 0.0

    def update(self, dt: float) -> None:
        self.attack_timer = max(
            0.0,
            self.attack_timer - dt,
        )


@dataclass
class Objective(Entity):
    name: str = "neutral_objective"
    attack_damage: float = 40.0
    attack_range: float = 100.0
    attack_cooldown: float = 2.0
    attack_timer: float = 0.0

    reward_gold: int = 200
    reward_xp: float = 100.0

    def update(self, dt: float) -> None:
        self.attack_timer = max(
            0.0,
            self.attack_timer - dt,
        )


@dataclass
class VisionWard:
    id: int
    team: Team
    position: Position
    duration: float = 60.0
    max_duration: float = 60.0
