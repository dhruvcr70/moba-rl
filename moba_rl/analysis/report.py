from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from moba_rl.actions import action_name

Z95 = 1.959963984540054


def _wilson(wins: int, games: int) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    den = 1 + Z95 * Z95 / games
    center = (p + Z95 * Z95 / (2 * games)) / den
    margin = Z95 * math.sqrt(p * (1 - p) / games + Z95 * Z95 / (4 * games * games)) / den
    return max(0.0, center - margin), min(1.0, center + margin)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dependencies():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Analysis needs the dev extras: python -m pip install -e '.[dev]'"
        ) from exc
    return pd, plt


def analyze_evaluations(csv_path: Path, out_dir: Path, pd, plt):
    frame = pd.read_csv(csv_path)
    required = {"model", "opponent", "episodes", "wins", "losses", "draws"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing)}")

    aggregate = {
        "episodes": ("episodes", "sum"),
        "wins": ("wins", "sum"),
        "losses": ("losses", "sum"),
        "draws": ("draws", "sum"),
        "mean_episode_length": ("mean_episode_length", "mean"),
        "mean_episode_reward": ("mean_episode_reward", "mean"),
    }
    decision_metrics = [
        name for name in frame.columns
        if (name.startswith("mean_") and name not in {
            "mean_episode_length", "mean_episode_reward"
        }) or name in {"blue_action_diversity", "blue_advanced_action_rate"}
    ]
    for name in decision_metrics:
        aggregate[name] = (name, "mean")
    summary = (
        frame.groupby(["model", "opponent"], as_index=False)
        .agg(**aggregate)
    )
    summary["win_rate"] = summary["wins"] / summary["episodes"].clip(lower=1)
    summary.to_csv(out_dir / "evaluation_summary.csv", index=False)
    if decision_metrics:
        summary[["model", "opponent", "episodes", "win_rate", *decision_metrics]].to_csv(
            out_dir / "decision_quality.csv", index=False
        )

    pivot = summary.pivot(index="opponent", columns="model", values="win_rate")
    axis = pivot.plot(kind="bar", figsize=(11, 6), ylim=(0, 1), rot=25)
    axis.set_ylabel("Win rate")
    axis.set_title("Win rate by opponent")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Model", loc="best")
    axis.figure.tight_layout()
    axis.figure.savefig(out_dir / "win_rates.png", dpi=160)
    plt.close(axis.figure)
    return summary


def analyze_training(csv_path: Path, out_dir: Path, pd, plt):
    frame = pd.read_csv(csv_path)
    if "timesteps" not in frame or "overall_win_rate" not in frame:
        raise ValueError(
            "Training evaluation CSV must contain timesteps and overall_win_rate"
        )
    points = frame[["timesteps", "overall_win_rate"]].drop_duplicates()
    if "opponent" in frame:
        points = frame.groupby("timesteps", as_index=False)["overall_win_rate"].first()
    points.to_csv(out_dir / "training_progress.csv", index=False)
    axis = points.plot(
        x="timesteps",
        y="overall_win_rate",
        figsize=(10, 5),
        legend=False,
    )
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Mean scripted-opponent win rate")
    axis.set_ylim(0, 1)
    axis.set_title("Evaluation progress during training")
    axis.grid(alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(out_dir / "training_progress.png", dpi=160)
    plt.close(axis.figure)
    return {
        "checkpoints": int(points["timesteps"].nunique()),
        "episodes_per_opponent_min": int(frame["episodes"].min()) if "episodes" in frame else None,
        "episodes_per_opponent_max": int(frame["episodes"].max()) if "episodes" in frame else None,
    }


def analyze_replays(replay_dir: Path, out_dir: Path, pd, plt):
    replay_files = sorted(replay_dir.rglob("*.json"))
    if not replay_files:
        raise ValueError(f"No replay JSON files found under {replay_dir}")

    episode_rows = []
    action_rows = []
    for replay_path in replay_files:
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        trajectory = replay.get("trajectory", [])
        blue_actions = {}
        red_actions = {}
        event_count = 0
        objective_captures = 0
        blue_objective_captures = 0
        red_objective_captures = 0
        blue_ward_placements = 0
        red_ward_placements = 0
        blue_ward_sweeps = 0
        red_ward_sweeps = 0
        blue_wave_orders = 0
        red_wave_orders = 0
        for item in trajectory:
            action = item.get("blue_action")
            if action is not None:
                blue_actions[int(action)] = blue_actions.get(int(action), 0) + 1
            action = item.get("red_action")
            if action is not None:
                red_actions[int(action)] = red_actions.get(int(action), 0) + 1
            events = item.get("events", [])
            event_count += len(events)
            objective_captures += sum(
                "captured the neutral objective" in event for event in events
            )
            blue_objective_captures += sum(
                "BLUE captured the neutral objective" in event for event in events
            )
            red_objective_captures += sum(
                "RED captured the neutral objective" in event for event in events
            )
            blue_ward_placements += sum("BLUE placed a vision ward" in event for event in events)
            red_ward_placements += sum("RED placed a vision ward" in event for event in events)
            blue_ward_sweeps += sum("BLUE swept enemy vision" in event for event in events)
            red_ward_sweeps += sum("RED swept enemy vision" in event for event in events)
            blue_wave_orders += sum("BLUE called a " in event for event in events)
            red_wave_orders += sum("RED called a " in event for event in events)

        final_state = trajectory[-1].get("state", {}) if trajectory else {}
        champions = final_state.get("champions", {})
        blue_state = champions.get("blue", {})
        red_state = champions.get("red", {})
        towers = final_state.get("lane_towers", {})
        blue_towers_lost = sum(
            not item.get("alive", True)
            for item in towers.get("blue", {}).values()
        )
        red_towers_lost = sum(
            not item.get("alive", True)
            for item in towers.get("red", {}).values()
        )
        decisions = [item.get("blue_action") for item in trajectory if item.get("blue_action") is not None]

        episode_rows.append({
            "file": replay_path.name,
            "opponent": replay.get("opponent"),
            "winner": replay.get("winner"),
            "episode_seed": replay.get("episode_seed"),
            "steps": max(0, len(trajectory) - 1),
            "event_count": event_count,
            "objective_captures": objective_captures,
            "blue_objective_captures": blue_objective_captures,
            "red_objective_captures": red_objective_captures,
            "blue_kills": blue_state.get("kills", 0),
            "red_kills": red_state.get("kills", 0),
            "blue_deaths": blue_state.get("deaths", 0),
            "red_deaths": red_state.get("deaths", 0),
            "blue_final_gold": blue_state.get("gold", 0),
            "red_final_gold": red_state.get("gold", 0),
            "blue_final_level": blue_state.get("level", 1),
            "red_final_level": red_state.get("level", 1),
            "blue_towers_lost": blue_towers_lost,
            "red_towers_lost": red_towers_lost,
            "blue_ward_placements": blue_ward_placements,
            "red_ward_placements": red_ward_placements,
            "blue_ward_sweeps": blue_ward_sweeps,
            "red_ward_sweeps": red_ward_sweeps,
            "blue_wave_orders": blue_wave_orders,
            "red_wave_orders": red_wave_orders,
            "blue_action_diversity": len(set(decisions)),
            "blue_advanced_action_rate": sum(int(action) >= 24 for action in decisions) / max(1, len(decisions)),
        })
        for side, counts in (("blue", blue_actions), ("red", red_actions)):
            for action, count in counts.items():
                action_rows.append({
                    "file": replay_path.name,
                    "side": side,
                    "action": action,
                    "action_name": action_name(action),
                    "count": count,
                })

    episodes = pd.DataFrame(episode_rows)
    actions = pd.DataFrame(action_rows)
    episodes.to_csv(out_dir / "replay_summary.csv", index=False)
    actions.to_csv(out_dir / "replay_action_counts.csv", index=False)
    if not actions.empty:
        aggregate = actions.groupby(["side", "action"], as_index=False)["count"].sum()
        labels = actions[["action", "action_name"]].drop_duplicates().set_index("action")
        aggregate["label"] = aggregate["action"].map(labels["action_name"])
        pivot = aggregate.pivot(index="label", columns="side", values="count").fillna(0)
        axis = pivot.plot(kind="bar", figsize=(11, 6), rot=0)
        axis.set_xlabel("Action")
        axis.set_ylabel("Action count across recorded matches")
        axis.set_title("Replay action frequencies")
        axis.grid(axis="y", alpha=0.25)
        axis.figure.tight_layout()
        axis.figure.savefig(out_dir / "replay_actions.png", dpi=160)
        plt.close(axis.figure)
    return episodes


def analyze_benchmarks(paths: list[Path], out_dir: Path, pd, plt):
    """Summarize per-match paired benchmark exports across map conditions."""
    frames = []
    metadata = []
    required = {
        "opponent", "opponent_type", "candidate_side", "seed", "candidate_result",
        "candidate_objectives", "candidate_towers_destroyed", "candidate_kills",
        "candidate_action_diversity", "procedural_map",
    }
    for path in paths:
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Benchmark CSV {path} is missing columns: {sorted(missing)}")
        frame["map_condition"] = frame["procedural_map"].map(
            lambda value: "procedural" if str(value).lower() in {"true", "1"} else "fixed"
        )
        frame["benchmark_file"] = path.name
        frames.append(frame)
        manifest_path = path.with_name(path.stem + "_manifest.json")
        metadata.append({"file": path.name, "sha256": _sha256(path), "matches": len(frame),
                         "seed_min": int(frame["seed"].min()), "seed_max": int(frame["seed"].max()),
                         "map_conditions": sorted(frame["map_condition"].unique().tolist()),
                         "manifest_file": manifest_path.name if manifest_path.is_file() else None,
                         "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
                         "manifest": json.loads(manifest_path.read_text(encoding="utf-8"))
                         if manifest_path.is_file() else None})

    data = pd.concat(frames, ignore_index=True)
    rows = []
    for (map_condition, opponent, opponent_type, side), group in data.groupby(
            ["map_condition", "opponent", "opponent_type", "candidate_side"], sort=True):
        games = len(group)
        wins = int((group["candidate_result"] == "win").sum())
        draws = int((group["candidate_result"] == "draw").sum())
        losses = games - wins - draws
        low, high = _wilson(wins, games - draws)
        rows.append({
            "map_condition": map_condition,
            "opponent": opponent,
            "opponent_type": opponent_type,
            "candidate_side": side,
            "games": games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate_excluding_draws": wins / max(1, games - draws),
            "win_rate_95_ci_low": low,
            "win_rate_95_ci_high": high,
            "mean_objectives": group["candidate_objectives"].mean(),
            "mean_towers_destroyed": group["candidate_towers_destroyed"].mean(),
            "mean_kills": group["candidate_kills"].mean(),
            "mean_action_diversity": group["candidate_action_diversity"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)
    (out_dir / "benchmark_inputs.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Keep baseline and historical-policy results visible, split by side and map.
    plot = summary.copy()
    def short_name(name: str) -> str:
        normalized = str(name).replace("\\", "/")
        if "opponent_000030000" in normalized:
            return "self-play 30k"
        if "opponent_000050000" in normalized:
            return "self-play 50k"
        return normalized.rsplit("/", 1)[-1].removesuffix(".zip")

    plot["label"] = plot.apply(
        lambda row: f"{short_name(row['opponent'])}\n{row['candidate_side'].upper()} · {row['map_condition'][0].upper()}", axis=1
    )
    axis = plot.set_index("label")["win_rate_excluding_draws"].plot(
        kind="bar", figsize=(max(12, len(plot) * 0.68), 5.5), ylim=(0, 1), rot=25,
        color=["#4e8cff" if side == "blue" else "#ff6b79" for side in plot["candidate_side"]],
    )
    axis.set_ylabel("Win rate (draws excluded)")
    axis.set_title("Held-out match results by opponent, seat, and map condition")
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(out_dir / "benchmark_win_rates.png", dpi=160)
    plt.close(axis.figure)

    paired = data[data["map_condition"].isin(["fixed", "procedural"])].copy()
    paired["score"] = paired["candidate_result"].map({"win": 1.0, "draw": 0.5, "loss": 0.0})
    pivot = paired.pivot_table(index=["opponent", "candidate_side", "seed"],
                               columns="map_condition", values="score", aggfunc="first")
    if {"fixed", "procedural"}.issubset(pivot.columns):
        matched = pivot.dropna(subset=["fixed", "procedural"]).reset_index()
        map_rows = []
        for (opponent, side), group in matched.groupby(["opponent", "candidate_side"], sort=True):
            map_rows.append({
                "opponent": opponent,
                "candidate_side": side,
                "matched_seeds": len(group),
                "fixed_map_win_score": group["fixed"].mean(),
                "procedural_map_win_score": group["procedural"].mean(),
                "fixed_minus_procedural": (group["fixed"] - group["procedural"]).mean(),
            })
        map_comparison = pd.DataFrame(map_rows)
    else:
        map_comparison = pd.DataFrame(columns=["opponent", "candidate_side", "matched_seeds",
            "fixed_map_win_score", "procedural_map_win_score", "fixed_minus_procedural"])
    map_comparison.to_csv(out_dir / "paired_map_comparison.csv", index=False)

    report_lines = [
        "## Match benchmark evidence", "",
        "These are deterministic policy evaluations over fixed seeds. Wilson intervals use decisive games; draws are reported separately. Scripted opponents are evaluated with the candidate in Blue, while learned historical opponents are evaluated with the candidate in both seats. The adaptive baseline is held out: the saved training code lists aggressive, defensive, objective, and balanced as its scripted training opponents.",
        "",
        "| Map | Opponent | Type | Candidate seat | Games | W-D-L | Win rate (95% CI) | Mean objectives | Mean towers destroyed | Mean action diversity |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rate = row["win_rate_excluding_draws"]
        report_lines.append(
            f"| {row['map_condition']} | {row['opponent']} | {row['opponent_type']} | {row['candidate_side']} | {row['games']} | {row['wins']}-{row['draws']}-{row['losses']} | {rate:.1%} ({row['win_rate_95_ci_low']:.1%}–{row['win_rate_95_ci_high']:.1%}) | {row['mean_objectives']:.2f} | {row['mean_towers_destroyed']:.2f} | {row['mean_action_diversity']:.2f} |"
        )
    report_lines += [
        "",
        "![Benchmark win rates](benchmark_win_rates.png)",
        "",
        "The matched map comparison is in `paired_map_comparison.csv`. It compares fixed and procedural outcomes on the overlapping seed IDs and reports the paired mean score difference (win=1, draw=0.5, loss=0); it is descriptive and does not adjust for multiple opponents.",
        "",
        "Interpretation limits: these results describe the tested checkpoint and simulator settings, not human-level play. Historical policy snapshots are from the same training lineage, so they are useful self-play opponents but not independent external policies. The benchmark records objective captures and structure damage to expose wins that come from a narrow strategy.",
        "",
    ]
    return summary, report_lines


def main():
    parser = argparse.ArgumentParser(description="Summarize MOBA-RL evaluation and replay data.")
    parser.add_argument("--evaluations", type=Path, help="Evaluation CSV produced by evaluate.py")
    parser.add_argument("--training-csv", type=Path, help="Training callback evaluation CSV")
    parser.add_argument("--replays", type=Path, help="Directory containing saved replay JSON files")
    parser.add_argument("--benchmarks", type=Path, nargs="+", help="One or more per-match CSV files from evaluation.benchmark")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if not any((args.evaluations, args.training_csv, args.replays, args.benchmarks)):
        parser.error("provide at least one evaluation, benchmark, training, or replay input")

    pd, plt = _load_dependencies()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = ["# MOBA-RL analysis report", ""]
    training_summary = None
    if args.evaluations:
        summary = analyze_evaluations(args.evaluations, args.out_dir, pd, plt)
        report_lines += [
            "## Evaluation summary",
            "",
            summary.to_markdown(index=False),
            "",
            "![Win rates by opponent](win_rates.png)",
            "",
        ]
    if args.training_csv:
        training_summary = analyze_training(args.training_csv, args.out_dir, pd, plt)
        report_lines += [
            "## Training progress",
            "",
            "![Training evaluation progress](training_progress.png)",
            "",
            f"The source contains {training_summary['checkpoints']} checkpoints and {training_summary['episodes_per_opponent_min']}–{training_summary['episodes_per_opponent_max']} evaluation episode(s) per opponent per checkpoint. Treat the curve as a progress trace, not a precise estimate of final win rate or training-seed variance.",
            "",
        ]
    if args.replays:
        episodes = analyze_replays(args.replays, args.out_dir, pd, plt)
        report_lines += [
            "## Replay summary",
            "",
            episodes.to_markdown(index=False),
            "",
            "Action counts are also saved to `replay_action_counts.csv`.",
            "",
        ]
    if args.benchmarks:
        _, benchmark_lines = analyze_benchmarks(args.benchmarks, args.out_dir, pd, plt)
        data_rows = pd.concat([pd.read_csv(path) for path in args.benchmarks], ignore_index=True)
        objective_games = int((data_rows["candidate_objectives"] > 0).sum())
        objective_line = (
            f"Across these benchmark inputs, the candidate captured the neutral objective in {objective_games}/{len(data_rows)} matches."
            if objective_games else
            f"Across these benchmark inputs, the candidate captured the neutral objective in 0/{len(data_rows)} matches; its wins came through other simulator outcomes, chiefly structure destruction."
        )
        report_lines += [objective_line, ""]
        report_lines += benchmark_lines
    (args.out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Analysis outputs written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
