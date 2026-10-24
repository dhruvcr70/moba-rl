from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from moba_rl.strategy.territory import TerritoryControlEnv

ROOT = Path(__file__).resolve().parents[2]
STYLES = ("balanced", "expansionist", "developer")


class RotatingTerritoryOpponents(gym.Wrapper):
    def __init__(self, env, seed: int):
        super().__init__(env)
        self.rng = random.Random(seed)

    def reset(self, *, seed=None, options=None):
        self.env.unwrapped.opponent_style = self.rng.choice(STYLES)
        return self.env.reset(seed=seed, options=options)


def evaluate(model, episodes: int, seed: int):
    results = []
    for style in STYLES:
        wins = 0
        scores = []
        for episode in range(episodes):
            env = TerritoryControlEnv(seed=seed + episode)
            env.opponent_style = style
            observation, _ = env.reset(seed=seed + episode)
            while True:
                action, _ = model.predict(observation, action_masks=get_action_masks(env), deterministic=True)
                observation, _, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    wins += info["winner"] == "blue"
                    scores.append(info["blue_score"] - info["red_score"])
                    break
            env.close()
        results.append({"opponent": style, "episodes": episodes,
            "wins": int(wins), "win_rate": wins / max(1, episodes),
            "mean_score_difference": sum(scores) / max(1, len(scores))})
    return results


def main():
    parser = argparse.ArgumentParser(description="Train a MaskablePPO territory-control agent.")
    parser.add_argument("--run-name", default="territory_ppo")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.eval_episodes <= 0 or args.checkpoint_freq <= 0:
        parser.error("timesteps, eval-episodes, and checkpoint-freq must be positive")

    model_dir = ROOT / "models" / "territory"
    checkpoint_dir = ROOT / "checkpoints" / "territory" / args.run_name
    log_dir = ROOT / "logs" / "territory" / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(RotatingTerritoryOpponents(
        TerritoryControlEnv(seed=args.seed, max_steps=60), args.seed))
    model = MaskablePPO("MlpPolicy", env, seed=args.seed, device=args.device,
        verbose=1, tensorboard_log=str(log_dir),
        learning_rate=3e-4, n_steps=1024, batch_size=256, gamma=0.99,
        ent_coef=0.02)
    checkpoint = CheckpointCallback(save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir), name_prefix=args.run_name)
    print(f"Training territory-control PPO on {model.device}; outputs: {model_dir}")
    model.learn(total_timesteps=args.timesteps, callback=checkpoint,
        tb_log_name=args.run_name)
    destination = model_dir / args.run_name
    model.save(str(destination))
    results = evaluate(model, args.eval_episodes, args.seed + 100_000)
    report = log_dir / "evaluation.csv"
    with report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved model: {destination}.zip")
    print(f"Saved evaluation: {report}")
    for row in results:
        print(f"{row['opponent']}: win rate {row['win_rate']:.1%}, "
              f"mean score difference {row['mean_score_difference']:.1f}")


if __name__ == "__main__":
    main()
