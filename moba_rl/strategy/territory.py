from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TerritoryControlEnv(gym.Env):
    """Turn-based, non-combat 1v1 territory strategy environment.

    Each region has an owner, development level, fortification and resource
    yield. Players spend supplies to expand, develop or fortify, and use
    influence to sway neutral regions. Victory goes to the player with the
    strongest score when the turn limit is reached.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 1}
    WIDTH = 5
    HEIGHT = 5
    REGION_COUNT = WIDTH * HEIGHT
    HOLD = 0
    CLAIM = 1
    DEVELOP = 2
    FORTIFY = 3
    HARVEST = 4
    DIPLOMACY = 5
    ACTION_TYPES = ("hold", "claim", "develop", "fortify", "harvest", "diplomacy")

    def __init__(self, opponent=None, max_steps: int = 60, seed: int | None = None,
                 render_mode: str | None = None, record_trajectory: bool = False, **_: Any):
        super().__init__()
        self.opponent = opponent
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.initial_seed = seed
        self._has_reset = False
        self.render_mode = render_mode
        if render_mode not in {None, "human", "rgb_array", "ansi"}:
            raise ValueError("render_mode must be None, 'human', 'rgb_array', or 'ansi'")
        self.record_trajectory = record_trajectory
        self.action_space = spaces.Discrete(1 + len(self.ACTION_TYPES[1:]) * self.REGION_COUNT)
        self.observation_space = spaces.Box(0.0, 1.0, (self.REGION_COUNT * 5 + 11,), dtype=np.float32)
        self.steps = 0
        self.last_opponent_action: int | None = None
        self.game: TerritoryGame | None = None
        self.trajectory: list[dict[str, Any]] = []
        self._pygame = None
        self._screen = None
        self._clock = None

    @classmethod
    def encode(cls, verb: str, region: int = 0) -> int:
        if verb == "hold":
            return 0
        return cls.ACTION_TYPES[1:].index(verb) * cls.REGION_COUNT + int(region) + 1

    @classmethod
    def decode(cls, action: int) -> tuple[str, int | None]:
        action = int(action)
        if action == 0:
            return "hold", None
        verb_index, region = divmod(action - 1, cls.REGION_COUNT)
        if verb_index >= len(cls.ACTION_TYPES) - 1:
            raise ValueError(f"Invalid action {action}")
        return cls.ACTION_TYPES[verb_index + 1], region

    def reset(self, *, seed=None, options=None):
        reset_seed = seed
        if reset_seed is None and not self._has_reset:
            reset_seed = self.initial_seed
        super().reset(seed=reset_seed)
        self._has_reset = True
        game_seed = int(self.np_random.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        self.game = TerritoryGame(game_seed)
        self.steps = 0
        self.last_opponent_action = None
        self.trajectory = []
        if self.record_trajectory:
            self.trajectory.append({"step": 0, "state": self.game.get_state(), "events": []})
        return self._observation(), {"game_seed": game_seed, "steps": 0}

    def action_masks(self):
        assert self.game is not None
        return self.game.action_mask(0)

    def _opponent_choice(self) -> int:
        assert self.game is not None
        if self.opponent is not None:
            try:
                action = int(self.opponent.act(self.game))
                if self.game.action_mask(1)[action]:
                    return action
            except (AttributeError, IndexError, TypeError, ValueError):
                pass
        return self.game.heuristic_action(1, getattr(self, "opponent_style", "balanced"))

    def step(self, action: int):
        if self.game is None:
            raise RuntimeError("Environment must be reset before calling step().")
        action = int(action)
        mask = self.action_masks()
        if action < 0 or action >= len(mask) or not mask[action]:
            raise ValueError("Action is currently unavailable.")
        event_start = len(self.game.events)
        prior_score = self.game.score(0) - self.game.score(1)
        opponent_action = self._opponent_choice()
        if self.game.initiative == 0:
            self.game.resolve(action, 0)
            self.game.resolve(opponent_action, 1)
        else:
            self.game.resolve(opponent_action, 1)
            self.game.resolve(action, 0)
        self.last_opponent_action = opponent_action
        self.game.end_turn()
        self.steps += 1
        score_delta = (self.game.score(0) - self.game.score(1)) - prior_score
        reward = float(np.clip(score_delta / 100.0, -1.0, 1.0))
        terminated = self.game.game_over
        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            self.game.finish_by_score()
            terminated = True
            truncated = False
        info = {
            "steps": self.steps,
            "blue_score": self.game.score(0),
            "red_score": self.game.score(1),
            "blue_regions": self.game.region_count(0),
            "red_regions": self.game.region_count(1),
            "blue_action": action,
            "red_action": opponent_action,
            "winner": "blue" if self.game.winner == 0 else "red" if self.game.winner == 1 else None,
        }
        obs = self._observation()
        if self.record_trajectory:
            self.trajectory.append({"step": self.steps, "blue_action": action,
                "red_action": opponent_action, "reward": reward,
                "state": self.game.get_state(), "events": self.game.events[event_start:]})
        return obs, reward, terminated, truncated, info

    def _observation(self):
        assert self.game is not None
        features: list[float] = []
        for region in self.game.regions:
            features.extend([
                (region["owner"] + 1) / 2.0,
                region["development"] / 3.0,
                region["fortification"] / 3.0,
                region["yield_type"] / 2.0,
                region["influence"] / 9.0,
            ])
        own, enemy = self.game.players
        features.extend([
            own["supplies"] / 20.0, own["influence"] / 20.0,
            own["materials"] / 20.0, own["score"] / 250.0,
            enemy["supplies"] / 20.0, enemy["influence"] / 20.0,
            enemy["materials"] / 20.0, enemy["score"] / 250.0,
            self.steps / max(1, self.max_steps),
            self.game.initiative / 1.0,
            (self.game.objective_owner + 1.0) / 2.0,
        ])
        return np.clip(np.asarray(features, dtype=np.float32), 0.0, 1.0)

    def decision_target(self, action: int | None, team=None):
        if action is None or action == 0:
            return None
        verb, region = self.decode(action)
        return {"label": f"{verb} region {region + 1}", "region": region}

    def close(self):
        if self._pygame is not None and self._screen is not None:
            self._pygame.display.quit()
            self._screen = None

    def render(self):
        if self.game is None:
            return ""
        if self.render_mode == "ansi":
            return self.game.render()
        if self.render_mode not in {"human", "rgb_array"}:
            return None
        if self._pygame is None:
            try:
                import pygame
            except ImportError as exc:
                raise RuntimeError("Install the visualization extra to render with Pygame.") from exc
            self._pygame = pygame
            pygame.init()
        pygame = self._pygame
        surface = (pygame.display.set_mode((650, 650)) if self.render_mode == "human"
                   else pygame.Surface((650, 650)))
        self._screen = surface
        surface.fill((17, 25, 32))
        font = pygame.font.Font(None, 20)
        title = pygame.font.Font(None, 30)
        surface.blit(title.render(f"Territory Council  |  Turn {self.game.turn}/60", True, (228, 236, 242)), (28, 18))
        for index, region in enumerate(self.game.regions):
            x, y = 28 + (index % 5) * 120, 70 + (index // 5) * 112
            color = (35, 83, 119) if region["owner"] == 0 else (112, 52, 61) if region["owner"] == 1 else (48, 57, 65)
            rect = pygame.Rect(x, y, 108, 98)
            pygame.draw.rect(surface, color, rect, border_radius=10)
            pygame.draw.rect(surface, (220, 191, 116) if index == 12 else (87, 101, 112), rect, width=3 if index == 12 else 1, border_radius=10)
            owner = "BLUE" if region["owner"] == 0 else "RED" if region["owner"] == 1 else "OPEN"
            surface.blit(font.render(f"Region {index + 1}", True, (237, 242, 246)), (x + 9, y + 8))
            surface.blit(font.render(owner, True, (237, 242, 246)), (x + 9, y + 32))
            surface.blit(font.render(self.game.YIELDS[region["yield_type"]], True, (221, 202, 153)), (x + 9, y + 55))
            surface.blit(font.render(f"Dev {region['development']} · Guard {region['fortification']}", True, (203, 213, 220)), (x + 9, y + 76))
        if self.render_mode == "human":
            pygame.display.flip()
            if self._clock is None:
                self._clock = pygame.time.Clock()
            self._clock.tick(1)
            return None
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


class TerritoryGame:
    YIELDS = ("food", "materials", "influence")

    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.regions = [
            {"owner": -1, "development": 0, "fortification": 0,
             "yield_type": self.rng.randrange(3), "influence": 0}
            for _ in range(TerritoryControlEnv.REGION_COUNT)
        ]
        self.players = [
            {"supplies": 5, "materials": 3, "influence": 3, "score": 0},
            {"supplies": 5, "materials": 3, "influence": 3, "score": 0},
        ]
        # Mirrored starting holdings leave the central objective contested.
        self.regions[0]["owner"] = 0
        self.regions[-1]["owner"] = 1
        center = TerritoryControlEnv.REGION_COUNT // 2
        self.regions[center]["yield_type"] = 2
        self.objective_owner = -1
        self.initiative = 0
        self.turn = 0
        self.events: list[str] = ["Both factions establish a foothold at opposite corners."]
        self.game_over = False
        self.winner: int | None = None

    @staticmethod
    def neighbors(index: int) -> list[int]:
        x, y = index % 5, index // 5
        return [ny * 5 + nx for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1))
                if 0 <= nx < 5 and 0 <= ny < 5]

    def action_mask(self, player: int):
        mask = np.zeros(1 + 5 * 25, dtype=np.bool_)
        mask[0] = True
        profile = self.players[player]
        for idx, region in enumerate(self.regions):
            adjacent_owned = any(self.regions[n]["owner"] == player for n in self.neighbors(idx))
            mask[1 + idx] = region["owner"] == -1 and adjacent_owned and profile["supplies"] >= 2
            mask[1 + 25 + idx] = region["owner"] == player and region["development"] < 3 and profile["materials"] >= 2
            mask[1 + 50 + idx] = region["owner"] == player and region["fortification"] < 3 and profile["supplies"] >= 2
            mask[1 + 75 + idx] = region["owner"] == player
            mask[1 + 100 + idx] = region["owner"] != player and profile["influence"] >= 2
        return mask

    def resolve(self, action: int, player: int):
        verb, idx = TerritoryControlEnv.decode(action)
        p = self.players[player]
        team = "Blue" if player == 0 else "Red"
        if verb == "hold" or idx is None:
            self.events.append(f"{team} holds position and banks resources.")
            return
        r = self.regions[idx]
        if verb == "claim" and r["owner"] == -1 and any(self.regions[n]["owner"] == player for n in self.neighbors(idx)) and p["supplies"] >= 2:
            p["supplies"] -= 2
            r["owner"] = player
            r["influence"] = max(1, r["influence"])
            self.events.append(f"{team} claims region {idx + 1} ({self.YIELDS[r['yield_type']]} land).")
        elif verb == "develop" and r["owner"] == player and r["development"] < 3 and p["materials"] >= 2:
            p["materials"] -= 2
            r["development"] += 1
            self.events.append(f"{team} develops region {idx + 1} to level {r['development']}.")
        elif verb == "fortify" and r["owner"] == player and r["fortification"] < 3 and p["supplies"] >= 2:
            p["supplies"] -= 2
            r["fortification"] += 1
            self.events.append(f"{team} fortifies region {idx + 1}.")
        elif verb == "harvest" and r["owner"] == player:
            if r["yield_type"] == 0:
                p["supplies"] = min(20, p["supplies"] + 3 + r["development"])
            elif r["yield_type"] == 1:
                p["materials"] = min(20, p["materials"] + 3 + r["development"])
            else:
                p["influence"] = min(20, p["influence"] + 2 + r["development"])
            self.events.append(f"{team} gathers {self.YIELDS[r['yield_type']]} from region {idx + 1}.")
        elif verb == "diplomacy" and r["owner"] != player and p["influence"] >= 2:
            p["influence"] -= 2
            r["influence"] += 1
            if r["influence"] >= 3 + 2 * r["fortification"]:
                r["owner"] = player
                self.events.append(f"{team} secures region {idx + 1} through diplomacy.")
            else:
                self.events.append(f"{team} sends envoys to region {idx + 1}.")
        p["score"] += 1

    def end_turn(self):
        for player, p in enumerate(self.players):
            p["supplies"] = min(20, p["supplies"] + 1)
            p["materials"] = min(20, p["materials"] + 1)
            p["influence"] = min(20, p["influence"] + 1)
            for r in self.regions:
                if r["owner"] == player:
                    p["score"] += 2 + r["development"]
                    if r["yield_type"] == 0:
                        p["supplies"] = min(20, p["supplies"] + 1)
                    elif r["yield_type"] == 1:
                        p["materials"] = min(20, p["materials"] + 1)
                    else:
                        p["influence"] = min(20, p["influence"] + 1)
        center = 12
        owner = self.regions[center]["owner"]
        if owner != -1:
            self.objective_owner = owner
            self.players[owner]["score"] += 5
            self.events.append(f"{'Blue' if owner == 0 else 'Red'} controls the central council seat (+5 score).")
        self.turn += 1
        self.initiative = 1 - self.initiative
        if self.turn >= 60:
            self.finish_by_score()

    def score(self, player: int):
        return self.players[player]["score"] + 10 * sum(r["owner"] == player for r in self.regions) + 3 * sum(r["development"] for r in self.regions if r["owner"] == player)

    def region_count(self, player: int):
        return sum(r["owner"] == player for r in self.regions)

    def heuristic_action(self, player: int, style: str = "balanced"):
        mask = self.action_mask(player)
        legal = np.flatnonzero(mask)
        # Prefer expansion, then economic upgrades, then harvesting/scouting.
        priorities = {
            "balanced": (0, 25, 75, 100, 50),
            "expansionist": (0, 25, 75, 50, 100),
            "developer": (25, 75, 0, 50, 100),
        }.get(style, (0, 25, 75, 100, 50))
        for offset in priorities:
            candidates = [int(a) for a in legal if offset < a <= offset + 25]
            if candidates:
                return candidates[self.rng.randrange(len(candidates))]
        return 0

    def finish_by_score(self):
        scores = [self.score(0), self.score(1)]
        self.winner = 0 if scores[0] > scores[1] else 1 if scores[1] > scores[0] else None
        self.game_over = True
        self.events.append("The council tallies the final territorial score.")

    def get_state(self):
        return {
            "turn": self.turn, "game_over": self.game_over,
            "winner": "blue" if self.winner == 0 else "red" if self.winner == 1 else None,
            "map": {"width": 5, "height": 5}, "regions": self.regions,
            "players": {
                "blue": {**self.players[0], "score": self.score(0), "regions": self.region_count(0)},
                "red": {**self.players[1], "score": self.score(1), "regions": self.region_count(1)},
            },
            "objective_region": 12,
            "objective_owner": "blue" if self.objective_owner == 0 else "red" if self.objective_owner == 1 else "neutral",
        }

    def render(self):
        symbols = { -1: "·", 0: "B", 1: "R" }
        return "\n".join(" ".join(symbols[self.regions[y*5+x]["owner"]] for x in range(5)) for y in range(5))
