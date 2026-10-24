from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from moba_rl.agents.scripted import create_opponent
from moba_rl.agents.self_play import SelfPlayOpponent
from moba_rl.environment import MOBAEnv
from moba_rl.training.callbacks import (
    TrainingMetricsCallback,
    WandbTrackingCallback,
)


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"
BEST_MODEL_DIR = MODEL_DIR / "best"

OPPONENT_NAMES = [
    "aggressive",
    "defensive",
    "objective",
    "balanced",
]


class RotatingOpponentWrapper(gym.Wrapper):
    """Choose a scripted opponent at the beginning of every episode."""

    def __init__(self, env, opponent_names, seed=42, curriculum=False,
                 opponent_weights=None, mixed_maps=False):
        super().__init__(env)
        self.opponent_names = list(opponent_names)
        self.rng = np.random.default_rng(seed)
        self.current_opponent = None
        self.self_play_model = None
        self.self_play_models = []
        self.curriculum = bool(curriculum)
        self.training_progress = 0.0
        self.opponent_weights = dict(opponent_weights or {})
        self.mixed_maps = bool(mixed_maps)

    def set_training_progress(self, progress):
        self.training_progress = float(np.clip(progress, 0.0, 1.0))

    def set_self_play_model(self, model):
        self.self_play_model = model
        self.self_play_models = [model]

    def add_self_play_model(self, model, pool_size=4):
        self.self_play_models.append(model)
        live_model = self.self_play_model
        historical = [item for item in self.self_play_models if item is not live_model]
        historical = historical[-max(1, int(pool_size)):]
        self.self_play_models = [live_model, *historical]

    def eligible_opponents(self):
        if not self.curriculum:
            return list(self.opponent_names)

        if self.training_progress < 0.15:
            stage = ["aggressive"]
        elif self.training_progress < 0.35:
            stage = ["aggressive", "defensive"]
        elif self.training_progress < 0.60:
            stage = ["aggressive", "defensive", "objective"]
        elif self.training_progress < 0.80:
            stage = list(OPPONENT_NAMES)
        else:
            stage = list(self.opponent_names)
        return [name for name in stage if name in self.opponent_names]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        eligible = self.eligible_opponents()
        weights = np.asarray([self.opponent_weights.get(name, 1.0) for name in eligible], dtype=float)
        if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("Eligible opponent weights must be finite, non-negative, and have a positive sum")
        self.current_opponent = str(self.rng.choice(eligible, p=weights / weights.sum()))

        # Randomize map layout by episode while retaining seeded reproducibility.
        procedural_map = bool(self.rng.integers(0, 2)) if self.mixed_maps else None
        if procedural_map is not None:
            self.env.unwrapped.procedural_map = procedural_map

        # MOBAEnv owns the actual opponent policy.
        if self.current_opponent == "self_play":
            if self.self_play_model is None:
                raise RuntimeError(
                    "Self-play was selected before its policy was assigned."
                )
            model = self.rng.choice(self.self_play_models or [self.self_play_model])
            opponent = SelfPlayOpponent(model)
        else:
            opponent = create_opponent(self.current_opponent)
        self.env.unwrapped.opponent = opponent

        observation, info = self.env.reset(
            seed=seed,
            options=options,
        )

        info["training_opponent"] = self.current_opponent
        if procedural_map is not None:
            info["training_procedural_map"] = procedural_map
        return observation, info


class CurriculumAndSelfPlayCallback(BaseCallback):
    """Advance curriculum stages and save frozen self-play opponents."""

    def __init__(
        self,
        opponent_wrapper,
        total_timesteps,
        curriculum=False,
        self_play=False,
        snapshot_freq=10_000,
        pool_size=4,
        snapshot_dir=None,
        verbose=0,
    ):
        super().__init__(verbose)
        self.opponent_wrapper = opponent_wrapper
        self.total_timesteps = max(1, int(total_timesteps))
        self.curriculum = bool(curriculum)
        self.self_play = bool(self_play)
        self.snapshot_freq = max(1, int(snapshot_freq))
        self.pool_size = max(1, int(pool_size))
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.next_snapshot = self.snapshot_freq

    def _on_training_start(self):
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.curriculum:
            progress = self.num_timesteps / self.total_timesteps
            if hasattr(self.opponent_wrapper, "env_method"):
                self.opponent_wrapper.env_method(
                    "set_training_progress", progress
                )
            else:
                self.opponent_wrapper.set_training_progress(progress)

        if (
            self.self_play
            and self.snapshot_dir is not None
            and self.num_timesteps >= self.next_snapshot
        ):
            snapshot_base = self.snapshot_dir / (
                f"opponent_{self.num_timesteps:09d}"
            )
            self.model.save(str(snapshot_base))
            snapshot = MaskablePPO.load(
                str(snapshot_base),
                device="cpu",
            )
            if hasattr(self.opponent_wrapper, "env_method"):
                self.opponent_wrapper.env_method(
                    "add_self_play_model", snapshot, self.pool_size
                )
            else:
                self.opponent_wrapper.add_self_play_model(
                    snapshot,
                    pool_size=self.pool_size,
                )
            if self.verbose:
                print(f"Added self-play opponent: {snapshot_base}.zip")
            while self.next_snapshot <= self.num_timesteps:
                self.next_snapshot += self.snapshot_freq

        return True


class MultiOpponentEvalCallback(BaseCallback):
    """Evaluate against every scripted opponent and save the best model."""

    def __init__(
        self,
        eval_freq,
        episodes_per_opponent,
        seed,
        best_model_path,
        csv_path,
        progress_reward_scale=0.0,
        progress_target_mode="lane",
        damage_reward_scale=0.0,
        objective_reward_scale=0.0,
        objective_capture_reward=2.0,
        structure_loss_penalty_scale=0.0,
        procedural_map=False,
        fog_of_war=False,
        strategic_actions=False,
        hierarchical_actions=False,
        advanced_decisions=False,
        ticks_per_decision=1,
        best_model_opponents=None,
        tracker=None,
        verbose=1,
    ):
        super().__init__(verbose)

        self.eval_freq = max(1, int(eval_freq))
        self.episodes_per_opponent = max(
            1, int(episodes_per_opponent)
        )
        self.seed = int(seed)
        self.best_model_path = Path(best_model_path)
        self.csv_path = Path(csv_path)
        self.progress_reward_scale = float(progress_reward_scale)
        self.progress_target_mode = progress_target_mode
        self.damage_reward_scale = float(damage_reward_scale)
        self.objective_reward_scale = float(objective_reward_scale)
        self.objective_capture_reward = float(objective_capture_reward)
        self.structure_loss_penalty_scale = float(
            structure_loss_penalty_scale
        )
        self.procedural_map = bool(procedural_map)
        self.fog_of_war = bool(fog_of_war)
        self.strategic_actions = bool(strategic_actions or hierarchical_actions)
        self.hierarchical_actions = bool(hierarchical_actions)
        self.advanced_decisions = bool(advanced_decisions)
        self.ticks_per_decision = max(1, int(ticks_per_decision))
        self.best_model_opponents = list(best_model_opponents or OPPONENT_NAMES)
        self.tracker = tracker
        self.best_win_rate = -1.0
        self.next_eval_timestep = self.eval_freq

    def _on_training_start(self):
        self.best_model_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        self.csv_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        if not self.csv_path.exists():
            with self.csv_path.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow([
                    "timesteps",
                    "opponent",
                    "episodes",
                    "wins",
                    "losses",
                    "draws",
                    "win_rate",
                    "mean_reward",
                    "mean_length",
                    "overall_win_rate",
                ])

    def _evaluate(self, opponent_name):
        env = MOBAEnv(
            seed=self.seed,
            max_steps=5000,
            ticks_per_decision=self.ticks_per_decision,
            progress_reward_scale=self.progress_reward_scale,
            progress_target_mode=self.progress_target_mode,
            damage_reward_scale=self.damage_reward_scale,
            objective_reward_scale=self.objective_reward_scale,
            objective_capture_reward=self.objective_capture_reward,
            structure_loss_penalty_scale=(
                self.structure_loss_penalty_scale
            ),
            opponent=create_opponent(opponent_name),
            procedural_map=self.procedural_map,
            fog_of_war=self.fog_of_war,
            strategic_actions=self.strategic_actions,
            hierarchical_actions=self.hierarchical_actions,
            advanced_decisions=self.advanced_decisions,
        )

        wins = 0
        losses = 0
        draws = 0
        rewards = []
        lengths = []

        try:
            for episode in range(self.episodes_per_opponent):
                observation, info = env.reset(
                    seed=(
                        self.seed
                        + self.num_timesteps
                        + episode
                    )
                )

                terminated = False
                truncated = False
                episode_reward = 0.0
                steps = 0

                while not terminated and not truncated:
                    # Fetch a fresh mask for the current state.
                    from sb3_contrib.common.maskable.utils import (
                        get_action_masks,
                    )

                    masks = get_action_masks(env)

                    action, _ = self.model.predict(
                        observation,
                        action_masks=masks,
                        deterministic=True,
                    )

                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(action)

                    episode_reward += float(reward)
                    steps += 1

                winner = info.get("winner")

                if winner == "blue":
                    wins += 1
                elif winner == "red":
                    losses += 1
                else:
                    draws += 1

                rewards.append(episode_reward)
                lengths.append(steps)

        finally:
            env.close()

        return {
            "opponent": opponent_name,
            "episodes": self.episodes_per_opponent,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / self.episodes_per_opponent,
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),
        }

    def _on_step(self):
        if self.num_timesteps < self.next_eval_timestep:
            return True
        while self.next_eval_timestep <= self.num_timesteps:
            self.next_eval_timestep += self.eval_freq

        results = [
            self._evaluate(name)
            for name in OPPONENT_NAMES
        ]

        overall_win_rate = float(
            np.mean([result["win_rate"] for result in results])
        )

        mean_reward = float(
            np.mean([result["mean_reward"] for result in results])
        )
        rates = {result["opponent"]: result["win_rate"] for result in results}
        selection_win_rate = float(np.mean([
            rates[name] for name in self.best_model_opponents
        ]))

        print()
        print("=" * 72)
        print(f"EVALUATION AT {self.num_timesteps:,} TIMESTEPS")
        print("=" * 72)

        with self.csv_path.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            for result in results:
                print(
                    f"{result['opponent']:>10}: "
                    f"win={result['win_rate']:.1%}, "
                    f"reward={result['mean_reward']:.2f}, "
                    f"length={result['mean_length']:.0f}"
                )

                writer.writerow([
                    self.num_timesteps,
                    result["opponent"],
                    result["episodes"],
                    result["wins"],
                    result["losses"],
                    result["draws"],
                    result["win_rate"],
                    result["mean_reward"],
                    result["mean_length"],
                    overall_win_rate,
                ])

        self.logger.record(
            "evaluation/mean_win_rate",
            overall_win_rate,
        )
        self.logger.record(
            "evaluation/mean_episode_reward",
            mean_reward,
        )

        print(f"Mean win rate: {overall_win_rate:.2%}")
        print(
            "Checkpoint selection win rate "
            f"({', '.join(self.best_model_opponents)}): {selection_win_rate:.2%}"
        )
        print(f"Mean episode reward: {mean_reward:.3f}")
        if self.tracker is not None:
            tracked = {
                "evaluation/mean_win_rate": overall_win_rate,
                "evaluation/mean_episode_reward": mean_reward,
            }
            tracked.update({
                f"evaluation/{result['opponent']}/win_rate": result["win_rate"]
                for result in results
            })
            self.tracker.log(tracked)

        # Allow a run to select checkpoints for its declared target matchups.
        if selection_win_rate > self.best_win_rate:
            self.best_win_rate = selection_win_rate

            self.model.save(str(self.best_model_path))

            print(
                f"New best model saved: "
                f"{self.best_model_path}.zip"
            )

        print("=" * 72)

        return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log training and evaluation metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="moba-rl",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel rollout environments (1–16).",
    )
    parser.add_argument(
        "--vec-env",
        choices=("dummy", "subproc"),
        default="dummy",
        help="Use in-process batched or subprocess-parallel environments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="moba_ppo",
        help="Name used for this run's checkpoints, model, and evaluation log.",
    )
    parser.add_argument(
        "--progress-reward-scale",
        type=float,
        default=0.0,
        help="Reward scale for movement toward the current strategic target.",
    )
    parser.add_argument(
        "--progress-target-mode",
        choices=("lane", "objective_then_lane"),
        default="lane",
        help="Movement target used for progress reward shaping.",
    )
    parser.add_argument(
        "--damage-reward-scale",
        type=float,
        default=0.0,
        help="Reward per point of direct damage dealt to enemy targets.",
    )
    parser.add_argument(
        "--objective-reward-scale",
        type=float,
        default=0.0,
        help="Additional zero-sum reward for capturing or denying the neutral objective.",
    )
    parser.add_argument(
        "--objective-capture-reward",
        type=float,
        default=2.0,
        help="Built-in event reward on neutral-objective captures (default 2.0).",
    )
    parser.add_argument(
        "--structure-loss-penalty-scale",
        type=float,
        default=0.0,
        help="Penalty per point of damage taken by the Blue tower/base.",
    )
    parser.add_argument(
        "--strategic-actions",
        action="store_true",
        help="Enable minion farming and shop-purchase actions (17-action policy).",
    )
    parser.add_argument(
        "--hierarchical-actions",
        action="store_true",
        help="Use high-level engage, push, objective, farm, shop, defend, and retreat actions (24-action policy).",
    )
    parser.add_argument(
        "--advanced-decisions",
        action="store_true",
        help="Add lane rotations, wave control, vision, objective zoning, and fight posture choices (36-action policy).",
    )
    parser.add_argument(
        "--ticks-per-decision",
        type=int,
        default=1,
        help="Simulation ticks advanced per policy choice (50 ticks = five simulated seconds).",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Introduce scripted opponents in stages as training progresses.",
    )
    parser.add_argument(
        "--self-play-pool-size",
        type=int,
        default=4,
        help="Maximum number of frozen self-play snapshots kept in memory.",
    )
    parser.add_argument(
        "--self-play-snapshot-freq",
        type=int,
        default=10_000,
        help="Timesteps between frozen self-play policy snapshots.",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Include the current policy as a mirrored RED-side training opponent.",
    )
    parser.add_argument(
        "--procedural-maps",
        action="store_true",
        help="Use seeded procedural lane layouts for training and evaluation.",
    )
    parser.add_argument(
        "--mixed-maps",
        action="store_true",
        help="Randomly use fixed or procedural maps each training episode; evaluation remains fixed-map.",
    )
    parser.add_argument(
        "--opponent-weights",
        nargs="+",
        metavar="NAME=WEIGHT",
        help="Sampling weights for scripted opponents, e.g. balanced=2 defensive=2 aggressive=1 objective=1.",
    )
    parser.add_argument(
        "--fog-of-war",
        action="store_true",
        help="Train with partial observations and explicit visibility features.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--best-model-opponents",
        nargs="+",
        choices=OPPONENT_NAMES,
        default=None,
        help="Opponents whose mean win rate selects the best checkpoint (defaults to all scripted opponents).",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="PPO entropy coefficient; modestly larger values can discourage premature action collapse.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
    )

    args = parser.parse_args()

    if not args.run_name or Path(args.run_name).name != args.run_name:
        parser.error("--run-name must be a non-empty filename-safe name")

    if args.progress_reward_scale < 0.0:
        parser.error("--progress-reward-scale must be non-negative")

    if args.damage_reward_scale < 0.0:
        parser.error("--damage-reward-scale must be non-negative")

    if args.objective_reward_scale < 0.0:
        parser.error("--objective-reward-scale must be non-negative")

    if args.objective_capture_reward < 0.0:
        parser.error("--objective-capture-reward must be non-negative")

    if args.structure_loss_penalty_scale < 0.0:
        parser.error("--structure-loss-penalty-scale must be non-negative")

    if args.ent_coef < 0.0:
        parser.error("--ent-coef must be non-negative")

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")

    if args.ticks_per_decision <= 0:
        parser.error("--ticks-per-decision must be positive")

    if args.advanced_decisions and not args.fog_of_war:
        parser.error("--advanced-decisions currently requires --fog-of-war so its 57-feature policy can be loaded by the arena.")

    if not 1 <= args.n_envs <= 16:
        parser.error("--n-envs must be between 1 and 16")

    if args.vec_env == "subproc" and args.n_envs < 2:
        parser.error("--vec-env subproc requires --n-envs of at least 2")

    if args.vec_env == "subproc" and args.self_play:
        parser.error(
            "--self-play currently requires --vec-env dummy because the live "
            "CUDA policy cannot safely be shared with Windows subprocesses."
        )

    if args.self_play_pool_size <= 0:
        parser.error("--self-play-pool-size must be positive")

    if args.self_play_snapshot_freq <= 0:
        parser.error("--self-play-snapshot-freq must be positive")

    if args.mixed_maps and args.procedural_maps:
        parser.error("Use either --mixed-maps or --procedural-maps, not both")

    opponent_weights = {name: 1.0 for name in OPPONENT_NAMES}
    if args.opponent_weights:
        opponent_weights = {}
        try:
            for item in args.opponent_weights:
                name, raw_weight = item.split("=", 1)
                if name not in OPPONENT_NAMES or name in opponent_weights:
                    raise ValueError
                weight = float(raw_weight)
                if not np.isfinite(weight) or weight < 0:
                    raise ValueError
                opponent_weights[name] = weight
            if sum(opponent_weights.values()) <= 0:
                raise ValueError
        except ValueError:
            parser.error("--opponent-weights must contain unique known names with finite non-negative weights and a positive total")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_model_path = BEST_MODEL_DIR / f"{args.run_name}_best"
    evaluation_csv_path = (
        LOG_DIR / "multi_opponent_evaluation.csv"
        if args.run_name == "moba_ppo"
        else LOG_DIR / f"{args.run_name}_multi_opponent_evaluation.csv"
    )

    training_opponents = list(OPPONENT_NAMES)
    if args.self_play:
        training_opponents.append("self_play")

    best_model_opponents = args.best_model_opponents or list(OPPONENT_NAMES)

    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    experiment_dir = ROOT / "experiments" / args.run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    source_paths = sorted((ROOT / "moba_rl").rglob("*.py"))
    run_config_path = experiment_dir / "config.json"
    run_config_path.write_text(json.dumps({
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": [sys.executable, "-m", "moba_rl.training.train_ppo", *sys.argv[1:]],
        "parameters": vars(args),
        "resolved_opponent_weights": opponent_weights,
        "training_opponents": training_opponents,
        "best_model_opponents": best_model_opponents,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": version("torch"),
            "stable_baselines3": version("stable-baselines3"),
            "sb3_contrib": version("sb3-contrib"),
            "gymnasium": version("gymnasium"),
            "cuda_available": bool(__import__("torch").cuda.is_available()),
            "cuda_device": (__import__("torch").cuda.get_device_name(0)
                            if __import__("torch").cuda.is_available() else None),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
        },
    }, indent=2), encoding="utf-8")

    def make_training_env(env_index):
        base_env = MOBAEnv(
            seed=args.seed + env_index,
            max_steps=5000,
            ticks_per_decision=args.ticks_per_decision,
            progress_reward_scale=args.progress_reward_scale,
            progress_target_mode=args.progress_target_mode,
            damage_reward_scale=args.damage_reward_scale,
            objective_reward_scale=args.objective_reward_scale,
            objective_capture_reward=args.objective_capture_reward,
            structure_loss_penalty_scale=args.structure_loss_penalty_scale,
            procedural_map=args.procedural_maps,
            fog_of_war=args.fog_of_war,
            strategic_actions=(args.strategic_actions or args.hierarchical_actions or args.advanced_decisions),
            hierarchical_actions=args.hierarchical_actions,
            advanced_decisions=args.advanced_decisions,
            opponent=create_opponent("aggressive"),
        )
        rotating = RotatingOpponentWrapper(
            base_env,
            training_opponents,
            seed=args.seed + env_index,
            curriculum=args.curriculum,
            opponent_weights=opponent_weights,
            mixed_maps=args.mixed_maps,
        )
        return Monitor(rotating)

    env_fns = [
        (lambda env_index=index: make_training_env(env_index))
        for index in range(args.n_envs)
    ]
    vec_env_class = DummyVecEnv if args.vec_env == "dummy" else SubprocVecEnv
    train_env = vec_env_class(env_fns)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.n_envs),
        save_path=str(CHECKPOINT_DIR),
        name_prefix=args.run_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    tracker_callback = None
    if args.wandb:
        tracker_callback = WandbTrackingCallback(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.run_name,
            config={
                "timesteps": args.timesteps,
                "seed": args.seed,
                "opponents": training_opponents,
                "curriculum": args.curriculum,
                "self_play": args.self_play,
                "procedural_maps": args.procedural_maps,
                "mixed_maps": args.mixed_maps,
                "opponent_weights": opponent_weights,
                "best_model_opponents": best_model_opponents,
                "ent_coef": args.ent_coef,
                "fog_of_war": args.fog_of_war,
                "strategic_actions": (args.strategic_actions or args.hierarchical_actions or args.advanced_decisions),
                "hierarchical_actions": args.hierarchical_actions,
                "advanced_decisions": args.advanced_decisions,
                "ticks_per_decision": args.ticks_per_decision,
                "progress_reward_scale": args.progress_reward_scale,
                "damage_reward_scale": args.damage_reward_scale,
                "objective_reward_scale": args.objective_reward_scale,
                "objective_capture_reward": args.objective_capture_reward,
            },
        )

    evaluation_callback = MultiOpponentEvalCallback(
        eval_freq=args.eval_freq,
        episodes_per_opponent=args.eval_episodes,
        seed=args.seed + 1000,
        best_model_path=best_model_path,
        csv_path=evaluation_csv_path,
        progress_reward_scale=args.progress_reward_scale,
        progress_target_mode=args.progress_target_mode,
        damage_reward_scale=args.damage_reward_scale,
        objective_reward_scale=args.objective_reward_scale,
        objective_capture_reward=args.objective_capture_reward,
        structure_loss_penalty_scale=args.structure_loss_penalty_scale,
        procedural_map=args.procedural_maps,
        fog_of_war=args.fog_of_war,
        strategic_actions=(args.strategic_actions or args.hierarchical_actions or args.advanced_decisions),
        hierarchical_actions=args.hierarchical_actions,
        advanced_decisions=args.advanced_decisions,
        ticks_per_decision=args.ticks_per_decision,
        best_model_opponents=best_model_opponents,
        tracker=tracker_callback,
        verbose=1,
    )

    metrics_callback = TrainingMetricsCallback()
    curriculum_callback = CurriculumAndSelfPlayCallback(
        opponent_wrapper=train_env,
        total_timesteps=args.timesteps,
        curriculum=args.curriculum,
        self_play=args.self_play,
        snapshot_freq=args.self_play_snapshot_freq,
        pool_size=args.self_play_pool_size,
        snapshot_dir=MODEL_DIR / "opponent_pool" / args.run_name,
        verbose=1 if args.self_play else 0,
    )

    callbacks = CallbackList([
        *([tracker_callback] if tracker_callback is not None else []),
        checkpoint_callback,
        evaluation_callback,
        metrics_callback,
        curriculum_callback,
    ])

    print()
    print("=" * 72)
    print("MOBA-RL: MIXED-OPPONENT PPO TRAINING")
    print("=" * 72)
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Rollout environments: {args.n_envs} ({args.vec_env})")
    print(f"Seed: {args.seed}")
    print(f"Progress reward scale: {args.progress_reward_scale:g}")
    print(f"Damage reward scale: {args.damage_reward_scale:g}")
    print(f"Objective reward scale: {args.objective_reward_scale:g}")
    print(f"Objective capture event reward: {args.objective_capture_reward:g}")
    print(f"Procedural maps: {args.procedural_maps}; mixed maps: {args.mixed_maps}")
    print(f"Fog of war: {args.fog_of_war}")
    print(f"Strategic actions (farm/shop): {args.strategic_actions or args.hierarchical_actions}")
    print(f"Hierarchical actions: {args.hierarchical_actions}")
    print(f"Advanced decisions: {args.advanced_decisions}")
    print(f"Simulation ticks per decision: {args.ticks_per_decision}")
    print(
        "Structure loss penalty scale: "
        f"{args.structure_loss_penalty_scale:g}"
    )
    print(f"Training opponents: {', '.join(training_opponents)}")
    print(f"Opponent sampling weights: {opponent_weights}")
    print(f"Weights & Biases: {args.wandb}")
    print(f"Curriculum: {args.curriculum}")
    print(f"Self-play opponent pool: {args.self_play}")
    print(f"Evaluation episodes per opponent: {args.eval_episodes}")
    print(f"Checkpoint selection opponents: {', '.join(best_model_opponents)}")
    print(f"PPO entropy coefficient: {args.ent_coef:g}")
    print(f"Experiment config: {run_config_path}")
    print("Algorithm: MaskablePPO")
    print("=" * 72)

    rollout_steps = max(1, 1024 // args.n_envs)
    rollout_size = rollout_steps * args.n_envs
    batch_size = min(256, rollout_size)
    while rollout_size % batch_size:
        batch_size -= 1

    try:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=rollout_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(LOG_DIR),
            seed=args.seed,
            device="auto",
        )

        if args.self_play:
            train_env.env_method("set_self_play_model", model)

        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=False,
        )

        final_path = MODEL_DIR / f"{args.run_name}_final"
        model.save(str(final_path))

        print()
        print("=" * 72)
        print("TRAINING COMPLETE")
        print(f"Final model: {final_path}.zip")
        print(
            "Best model: "
            f"{best_model_path}.zip"
        )
        print(
            "Evaluation CSV: "
            f"{evaluation_csv_path}"
        )
        print("=" * 72)

    finally:
        train_env.close()


if __name__ == "__main__":
    main()
