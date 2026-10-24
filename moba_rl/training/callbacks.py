from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    """
    Records useful MOBA-specific statistics in TensorBoard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:

        infos = self.locals.get("infos", [])

        for info in infos:

            if "winner" in info:
                winner = info["winner"]

                if winner == "blue":
                    self.logger.record(
                        "game/agent_win",
                        1,
                    )

                elif winner == "red":
                    self.logger.record(
                        "game/agent_win",
                        0,
                    )

            if "steps" in info:
                self.logger.record(
                    "game/steps",
                    info["steps"],
                )

            if "time" in info:
                self.logger.record(
                    "game/time",
                    info["time"],
                )

            if "blue_action_diversity" in info:
                self.logger.record(
                    "strategy/blue_action_diversity",
                    info["blue_action_diversity"],
                )
                self.logger.record(
                    "strategy/blue_advanced_action_rate",
                    info["blue_advanced_action_rate"],
                )
                for metric in (
                    "blue_objectives_taken", "red_objectives_taken",
                    "blue_towers_destroyed", "red_towers_destroyed",
                    "blue_ward_placements", "red_ward_placements",
                    "blue_wave_orders", "red_wave_orders",
                ):
                    self.logger.record(f"strategy/{metric}", info[metric])
                for name, count in info.get("blue_action_counts", {}).items():
                    self.logger.record(
                        f"strategy/action_count/{name}", count
                    )

        return True

class WandbTrackingCallback(BaseCallback):
    """Optional Weights & Biases logger for SB3 and match metrics."""

    def __init__(
        self,
        project,
        run_name,
        config,
        entity=None,
        log_freq=1024,
        verbose=0,
    ):
        super().__init__(verbose)
        self.project = project
        self.run_name = run_name
        self.config = dict(config)
        self.entity = entity
        self.log_freq = max(1, int(log_freq))
        self.wandb = None
        self.run = None

    def _on_training_start(self):
        try:
            import os
            import tempfile
            from pathlib import Path

            temp_root = Path("logs/wandb/tmp").resolve()
            temp_root.mkdir(parents=True, exist_ok=True)
            tempfile.tempdir = str(temp_root)
            os.environ["TMP"] = str(temp_root)
            os.environ["TEMP"] = str(temp_root)
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B tracking requested; install with `pip install -e '.[tracking]'`."
            ) from exc
        self.wandb = wandb
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            config=self.config,
            sync_tensorboard=True,
            save_code=True,
        )

    def log(self, values):
        if self.run is not None and values:
            # TensorBoard synchronization owns W&B's global step. Passing an
            # explicit step here emits a warning and can conflict with it.
            self.wandb.log(values)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode:
                metrics = {
                    "train/episode_reward": float(episode["r"]),
                    "train/episode_length": int(episode["l"]),
                }
                opponent = info.get("training_opponent")
                if opponent:
                    metrics["train/opponent"] = opponent
                winner = info.get("winner")
                if winner:
                    metrics["train/win"] = float(winner == "blue")
                for key in (
                    "blue_action_diversity", "blue_advanced_action_rate",
                    "blue_objectives_taken", "red_objectives_taken",
                    "blue_towers_destroyed", "red_towers_destroyed",
                    "blue_ward_placements", "red_ward_placements",
                    "blue_wave_orders", "red_wave_orders",
                ):
                    if key in info:
                        metrics[f"strategy/{key}"] = float(info[key])
                action_counts = info.get("blue_action_counts", {})
                total_actions = max(1, sum(action_counts.values()))
                for action_name, count in action_counts.items():
                    metrics[f"strategy/action_share/{action_name}"] = count / total_actions
                self.log(metrics)

        if self.n_calls % self.log_freq == 0:
            values = {}
            for key, value in self.logger.name_to_value.items():
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                if scalar == scalar:  # Exclude NaNs.
                    values[key] = scalar
            self.log(values)
        return True

    def _on_training_end(self):
        if self.run is not None:
            self.run.finish()
            self.run = None
