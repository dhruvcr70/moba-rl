from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from moba_rl.agents.scripted import create_opponent
from moba_rl.environment import MOBAEnv
from moba_rl.simulation.lini_tities import Team


ROOT = Path(__file__).resolve().parents[2]

OPPONENTS = [
    "aggressive",
    "defensive",
    "objective",
    "balanced",
    "adaptive",
]


def evaluate(
    model,
    opponent_name,
    episodes,
    seed,
    progress_reward_scale=0.0,
    progress_target_mode="lane",
    damage_reward_scale=0.0,
    objective_reward_scale=0.0,
    structure_loss_penalty_scale=0.0,
    procedural_map=False,
    fog_of_war=False,
    strategic_actions=False,
    hierarchical_actions=False,
    advanced_decisions=False,
    ticks_per_decision=1,
    max_steps=5000,
    replay_dir=None,
):
    env = MOBAEnv(
        seed=seed,
        max_steps=max_steps,
        ticks_per_decision=ticks_per_decision,
        progress_reward_scale=progress_reward_scale,
        progress_target_mode=progress_target_mode,
        damage_reward_scale=damage_reward_scale,
        objective_reward_scale=objective_reward_scale,
        structure_loss_penalty_scale=structure_loss_penalty_scale,
        opponent=create_opponent(opponent_name),
        procedural_map=procedural_map,
        fog_of_war=fog_of_war,
        strategic_actions=strategic_actions,
        hierarchical_actions=hierarchical_actions,
        advanced_decisions=advanced_decisions,
        record_trajectory=replay_dir is not None,
    )

    wins = 0
    losses = 0
    draws = 0
    episode_lengths = []
    episode_rewards = []
    metrics = Counter()
    action_counts = Counter()

    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            observation, info = env.reset(seed=episode_seed)
            game_seed = info["game_seed"]

            terminated = False
            truncated = False
            steps = 0
            total_reward = 0.0

            while not terminated and not truncated:
                masks = get_action_masks(env)

                action, _ = model.predict(
                    observation,
                    action_masks=masks,
                    deterministic=True,
                )
                action_counts[int(action)] += 1

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                steps += 1
                total_reward += float(reward)

            winner = info.get("winner")

            if winner == "blue":
                wins += 1
            elif winner == "red":
                losses += 1
            else:
                draws += 1

            episode_lengths.append(steps)
            episode_rewards.append(total_reward)

            game = env.game
            blue = game.champions[Team.BLUE]
            red = game.champions[Team.RED]
            metrics["blue_objectives"] += blue.objectives_taken
            metrics["red_objectives"] += red.objectives_taken
            metrics["blue_towers_destroyed"] += sum(not tower.alive for tower in game.lane_towers[red.team].values())
            metrics["red_towers_destroyed"] += sum(not tower.alive for tower in game.lane_towers[blue.team].values())
            metrics["blue_kills"] += blue.kills
            metrics["red_kills"] += red.kills
            metrics["blue_deaths"] += blue.deaths
            metrics["red_deaths"] += red.deaths
            metrics["blue_final_gold"] += blue.gold
            metrics["red_final_gold"] += red.gold
            metrics["blue_final_level"] += blue.level
            metrics["red_final_level"] += red.level
            metrics["blue_ward_placements"] += sum("BLUE placed a vision ward" in event for event in game.events)
            metrics["red_ward_placements"] += sum("RED placed a vision ward" in event for event in game.events)
            metrics["blue_ward_sweeps"] += sum("BLUE swept enemy vision" in event for event in game.events)
            metrics["red_ward_sweeps"] += sum("RED swept enemy vision" in event for event in game.events)
            metrics["blue_wave_orders"] += sum("BLUE called a " in event for event in game.events)
            metrics["red_wave_orders"] += sum("RED called a " in event for event in game.events)

            if replay_dir is not None:
                replay_path = Path(replay_dir) / (
                    f"{opponent_name}_episode_{episode:04d}_seed_{episode_seed}.json"
                )
                replay_path.parent.mkdir(parents=True, exist_ok=True)
                with replay_path.open("w", encoding="utf-8") as replay_file:
                    json.dump({
                        "schema_version": 1,
                        "opponent": opponent_name,
                        "episode_seed": episode_seed,
                        "game_seed": game_seed,
                        "winner": winner,
                        "trajectory": env.trajectory,
                    }, replay_file)

    finally:
        env.close()

    return {
        "opponent": opponent_name,
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / episodes,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        **{
            f"mean_{name}": float(value / episodes)
            for name, value in metrics.items()
        },
        "blue_action_diversity": len(action_counts),
        "blue_advanced_action_rate": float(
            sum(count for action, count in action_counts.items() if action >= 24)
            / max(1, sum(action_counts.values()))
        ),
        "blue_action_counts": json.dumps(
            {str(action): action_counts[action] for action in sorted(action_counts)},
            separators=(",", ":"),
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="models/moba_ppo_final.zip",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--progress-reward-scale",
        type=float,
        default=0.0,
        help="Use the same movement-progress reward scale as training.",
    )
    parser.add_argument(
        "--progress-target-mode",
        choices=("lane", "objective_then_lane"),
        default="lane",
        help="Use the same movement target mode as training.",
    )
    parser.add_argument(
        "--damage-reward-scale",
        type=float,
        default=0.0,
        help="Use the same direct-damage reward scale as training.",
    )
    parser.add_argument(
        "--objective-reward-scale",
        type=float,
        default=0.0,
        help="Use the same objective-control reward scale as training.",
    )
    parser.add_argument(
        "--structure-loss-penalty-scale",
        type=float,
        default=0.0,
        help="Use the same Blue tower/base damage penalty as training.",
    )
    parser.add_argument(
        "--strategic-actions",
        action="store_true",
        help="Match a model trained with minion and shop actions.",
    )
    parser.add_argument(
        "--hierarchical-actions",
        action="store_true",
        help="Match a model trained with the 24-action hierarchical profile.",
    )
    parser.add_argument(
        "--advanced-decisions",
        action="store_true",
        help="Match a checkpoint trained with the 36-action advanced profile.",
    )
    parser.add_argument(
        "--ticks-per-decision", type=int, default=1,
        help="Match the number of simulation ticks per policy choice used in training.",
    )
    parser.add_argument(
        "--max-decisions", type=int, default=5000,
        help="Maximum policy choices per match before marking it truncated.",
    )
    parser.add_argument(
        "--procedural-maps",
        action="store_true",
        help="Evaluate on seeded procedural lane layouts.",
    )
    parser.add_argument(
        "--fog-of-war",
        action="store_true",
        help="Compatibility flag; fog settings are inferred from the checkpoint observation size.",
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=None,
        help="Optional directory for one JSON trajectory per evaluation episode.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="logs/evaluation_results.csv",
    )

    args = parser.parse_args()

    if args.progress_reward_scale < 0.0:
        parser.error("--progress-reward-scale must be non-negative")

    if args.damage_reward_scale < 0.0:
        parser.error("--damage-reward-scale must be non-negative")

    if args.objective_reward_scale < 0.0:
        parser.error("--objective-reward-scale must be non-negative")

    if args.structure_loss_penalty_scale < 0.0:
        parser.error("--structure-loss-penalty-scale must be non-negative")

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    model_path = ROOT / args.model

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = MaskablePPO.load(str(model_path))

    action_count = int(model.action_space.n)
    observation_count = int(model.observation_space.shape[0])
    profile_base = {12: 33, 17: 39, 24: 39, 36: 53}.get(action_count)
    if profile_base is None or observation_count not in {profile_base, profile_base + 4}:
        parser.error(
            f"Unsupported policy profile: {action_count} actions and "
            f"{observation_count} observations. Expected 12/33–37, 17/39–43, "
            "24/39–43, or 36/53–57."
        )
    strategic_actions = action_count in {17, 24, 36}
    hierarchical_actions = action_count in {24, 36}
    advanced_decisions = action_count == 36
    fog_of_war = observation_count == profile_base + 4
    if args.ticks_per_decision <= 0 or args.max_decisions <= 0:
        parser.error("--ticks-per-decision and --max-decisions must be positive")

    print()
    print("=" * 72)
    print("MOBA-RL EVALUATION")
    print("=" * 72)
    print(f"Model: {model_path}")
    print(f"Episodes per opponent: {args.episodes}")
    print(f"Policy profile: {action_count} actions / {observation_count} observations")
    print(f"Ticks per decision: {args.ticks_per_decision}")
    print("=" * 72)

    results = []

    for index, opponent_name in enumerate(OPPONENTS):
        result = evaluate(
            model=model,
            opponent_name=opponent_name,
            episodes=args.episodes,
            seed=args.seed + index * 10_000,
            progress_reward_scale=args.progress_reward_scale,
            progress_target_mode=args.progress_target_mode,
            damage_reward_scale=args.damage_reward_scale,
            objective_reward_scale=args.objective_reward_scale,
            structure_loss_penalty_scale=(
                args.structure_loss_penalty_scale
            ),
            procedural_map=args.procedural_maps,
            fog_of_war=fog_of_war,
            strategic_actions=strategic_actions,
            hierarchical_actions=hierarchical_actions,
            advanced_decisions=advanced_decisions,
            ticks_per_decision=args.ticks_per_decision,
            max_steps=args.max_decisions,
            replay_dir=(
                str(ROOT / args.replay_dir)
                if args.replay_dir is not None
                else None
            ),
        )

        results.append(result)

        print()
        print(f"Opponent: {result['opponent']}")
        print(f"Win rate: {result['win_rate']:.2%}")
        print(f"Wins: {result['wins']}")
        print(f"Losses: {result['losses']}")
        print(f"Draws: {result['draws']}")
        print(
            "Mean episode length: "
            f"{result['mean_episode_length']:.1f}"
        )
        print(
            "Mean episode reward: "
            f"{result['mean_episode_reward']:.3f}"
        )

    overall_win_rate = float(
        np.mean([result["win_rate"] for result in results])
    )

    print()
    print("=" * 72)
    print(f"OVERALL MEAN WIN RATE: {overall_win_rate:.2%}")
    print("=" * 72)

    csv_path = ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "model", "opponent", "episodes", "wins", "losses", "draws",
        "win_rate", "mean_episode_length", "mean_episode_reward",
        "mean_blue_objectives", "mean_red_objectives",
        "mean_blue_towers_destroyed", "mean_red_towers_destroyed",
        "mean_blue_kills", "mean_red_kills", "mean_blue_deaths", "mean_red_deaths",
        "mean_blue_final_gold", "mean_red_final_gold",
        "mean_blue_final_level", "mean_red_final_level",
        "mean_blue_ward_placements", "mean_red_ward_placements",
        "mean_blue_ward_sweeps", "mean_red_ward_sweeps",
        "mean_blue_wave_orders", "mean_red_wave_orders",
        "blue_action_diversity", "blue_advanced_action_rate", "blue_action_counts",
    ]
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            existing_header = next(csv.reader(file), [])
        if existing_header != fieldnames:
            csv_path = csv_path.with_name(f"{csv_path.stem}_v2{csv_path.suffix}")
            print(f"Existing CSV schema differs; writing the expanded metrics to {csv_path.name}.")
    write_header = not csv_path.exists()

    with csv_path.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow(fieldnames)

        timestamp = datetime.now().isoformat(timespec="seconds")

        for result in results:
            writer.writerow([
                timestamp,
                str(model_path),
                result["opponent"],
                result["episodes"],
                result["wins"],
                result["losses"],
                result["draws"],
                result["win_rate"],
                result["mean_episode_length"],
                result["mean_episode_reward"],
                result.get("mean_blue_objectives", 0.0),
                result.get("mean_red_objectives", 0.0),
                result.get("mean_blue_towers_destroyed", 0.0),
                result.get("mean_red_towers_destroyed", 0.0),
                result.get("mean_blue_kills", 0.0),
                result.get("mean_red_kills", 0.0),
                result.get("mean_blue_deaths", 0.0),
                result.get("mean_red_deaths", 0.0),
                result.get("mean_blue_final_gold", 0.0),
                result.get("mean_red_final_gold", 0.0),
                result.get("mean_blue_final_level", 0.0),
                result.get("mean_red_final_level", 0.0),
                result.get("mean_blue_ward_placements", 0.0),
                result.get("mean_red_ward_placements", 0.0),
                result.get("mean_blue_ward_sweeps", 0.0),
                result.get("mean_red_ward_sweeps", 0.0),
                result.get("mean_blue_wave_orders", 0.0),
                result.get("mean_red_wave_orders", 0.0),
                result["blue_action_diversity"],
                result["blue_advanced_action_rate"],
                result["blue_action_counts"],
            ])

    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
