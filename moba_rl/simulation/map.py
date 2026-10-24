from dataclasses import dataclass
import random

from .lini_tities import Position, Team


@dataclass
class MapConfig:
    width: float = 2000.0
    height: float = 600.0

    lane_y: float = 300.0

    # Three parallel lanes. ``lane_y`` remains the middle-lane alias so old
    # callers and saved configurations keep their original coordinates.
    top_lane_y: float | None = None
    bottom_lane_y: float | None = None

    blue_base_x: float = 100.0
    red_base_x: float = 1900.0

    blue_tower_x: float = 500.0
    red_tower_x: float = 1500.0

    objective_x: float = 1000.0
    objective_y: float = 225.0

    @classmethod
    def procedural(cls, rng: random.Random) -> "MapConfig":
        """Generate a seeded, mirrored three-lane map."""
        width = rng.uniform(1800.0, 2200.0)
        height = rng.uniform(520.0, 760.0)
        lane_y = height * rng.uniform(0.46, 0.54)
        lane_spread = height * rng.uniform(0.25, 0.31)
        base_margin = width * rng.uniform(0.04, 0.07)
        tower_fraction = rng.uniform(0.24, 0.31)
        return cls(
            width=width,
            height=height,
            lane_y=lane_y,
            top_lane_y=max(0.0, lane_y - lane_spread),
            bottom_lane_y=min(height, lane_y + lane_spread),
            blue_base_x=base_margin,
            red_base_x=width - base_margin,
            blue_tower_x=width * tower_fraction,
            red_tower_x=width * (1.0 - tower_fraction),
            objective_x=width * rng.uniform(0.46, 0.54),
            objective_y=lane_y + rng.choice((-1.0, 1.0)) * lane_spread * 0.5,
        )


class MOBAmap:
    def __init__(self, config: MapConfig | None = None):
        self.config = config or MapConfig()

    def base_position(self, team: Team) -> Position:
        if team == Team.BLUE:
            return Position(self.config.blue_base_x, self.config.lane_y)
        return Position(self.config.red_base_x, self.config.lane_y)

    @property
    def lanes(self) -> tuple[str, str, str]:
        return ("top", "mid", "bottom")

    def lane_y(self, lane: str) -> float:
        """Return the vertical centerline for a named lane."""
        if lane == "top":
            return self.config.top_lane_y if self.config.top_lane_y is not None else self.config.height * 0.25
        if lane == "mid":
            return self.config.lane_y
        if lane == "bottom":
            return self.config.bottom_lane_y if self.config.bottom_lane_y is not None else self.config.height * 0.75
        raise ValueError(f"Unknown lane: {lane!r}")

    def lane_position(self, x: float, lane: str) -> Position:
        return Position(x, self.lane_y(lane))

    def tower_position(self, team: Team, lane: str = "mid") -> Position:
        lane_y = self.lane_y(lane)
        if team == Team.BLUE:
            return Position(self.config.blue_tower_x, lane_y)
        return Position(self.config.red_tower_x, lane_y)

    def objective_position(self) -> Position:
        return Position(self.config.objective_x, self.config.objective_y)

    def objective_spawn_positions(self) -> list[Position]:
        """Offer neutral objective sites in the central jungle between lanes."""
        top = self.lane_y("top")
        mid = self.lane_y("mid")
        bottom = self.lane_y("bottom")
        return [
            Position(self.config.width * 0.5, (top + mid) * 0.5),
            Position(self.config.width * 0.5, mid),
            Position(self.config.width * 0.5, (mid + bottom) * 0.5),
        ]

    def spawn_position(self, team: Team, lane: str = "mid") -> Position:
        base = self.base_position(team)
        return Position(base.x, self.lane_y(lane))
