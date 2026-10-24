from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from moba_rl.environment import MOBAEnv
from moba_rl.simulation.game import Game
from moba_rl.simulation.lini_tities import Minion, Position, Team, VisionWard
from moba_rl.simulation.map import MapConfig, MOBAmap


def _apply_entity_state(entity, state):
    entity.position = Position(float(state["x"]), float(state["y"]))
    entity.hp = float(state["hp"])
    entity.max_hp = float(state["max_hp"])
    if "respawn_timer" in state:
        entity.respawn_timer = float(state["respawn_timer"])


def _game_from_state(state):
    game = Game(seed=0)
    game.map = MOBAmap(MapConfig(**state["map"]))
    game.procedural_map = bool(state.get("procedural_map", False))
    game.time = float(state["time"])
    game.game_over = bool(state["game_over"])
    game.winner = Team(state["winner"]) if state.get("winner") else None
    game.objective_respawn_timer = state.get("objective_respawn_timer")

    for team in Team:
        champion_state = state["champions"][team.value]
        champion = game.champions[team]
        _apply_entity_state(champion, champion_state)
        for field in (
            "level", "xp", "gold", "kills", "deaths", "attack_damage",
            "armor", "move_speed", "attack_upgrade_level", "armor_upgrade_level",
            "speed_upgrade_level", "vitality_upgrade_level",
        ):
            if field in champion_state:
                setattr(champion, field, champion_state[field])

        for lane, tower_state in state.get("lane_towers", {}).get(team.value, {}).items():
            if lane in game.lane_towers[team]:
                _apply_entity_state(game.lane_towers[team][lane], tower_state)
        base_state = state["bases"][team.value]
        _apply_entity_state(game.bases[team], base_state)

    objective_state = state.get("objective")
    if objective_state:
        _apply_entity_state(game.objective, objective_state)
    else:
        game.objective = None

    game.minions = [
        Minion(
            id=game._get_id(), team=Team(item["team"]),
            position=Position(float(item["x"]), float(item["y"])),
            hp=float(item["hp"]), max_hp=float(item["max_hp"]),
            lane=item.get("lane", "mid"),
        )
        for item in state.get("minion_entities", [])
    ]
    game.wards = [
        VisionWard(
            id=game._get_id(), team=Team(item["team"]),
            position=Position(float(item["x"]), float(item["y"])),
            duration=float(item.get("duration", 0.0)),
            max_duration=float(item.get("max_duration", 60.0)),
        )
        for item in state.get("wards", [])
    ]
    for team in Team:
        team_orders = state.get("wave_orders", {}).get(team.value, {})
        for lane, order in team_orders.items():
            if lane in game.wave_orders[team]:
                game.wave_orders[team][lane].update(order)
    return game


def main():
    parser = argparse.ArgumentParser(description="Play a saved MOBA-RL match trajectory.")
    parser.add_argument("replay", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    trajectory = replay.get("trajectory", [])
    if not trajectory:
        raise ValueError("Replay contains no trajectory frames")

    env = MOBAEnv(render_mode="human")
    env.metadata = {**env.metadata, "render_fps": args.fps}
    try:
        for frame in trajectory:
            env.game = _game_from_state(frame["state"])
            env.render()
            if env._screen is None:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
