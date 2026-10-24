"""Paired, seeded evaluation against scripted and historical policies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from moba_rl.agents.scripted import OPPONENTS, create_opponent
from moba_rl.agents.self_play import SelfPlayOpponent
from moba_rl.environment import MOBAEnv
from moba_rl.simulation.lini_tities import Team

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "models/moba_advanced_big_20261023_final.zip"
Z95 = 1.959963984540054


def wilson(wins: int, games: int) -> tuple[float, float]:
    if games == 0:
        return 0.0, 0.0
    p = wins / games
    den = 1 + Z95 * Z95 / games
    center = (p + Z95 * Z95 / (2 * games)) / den
    margin = Z95 * math.sqrt(p * (1 - p) / games + Z95 * Z95 / (4 * games * games)) / den
    return max(0.0, center - margin), min(1.0, center + margin)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_profile(model: MaskablePPO) -> tuple[int, bool]:
    actions = int(model.action_space.n)
    observations = int(model.observation_space.shape[0])
    base = {12: 33, 17: 39, 24: 39, 36: 53}.get(actions)
    if base is None or observations not in {base, base + 4}:
        raise ValueError(f"Unsupported policy profile: {actions} actions / {observations} observations")
    return actions, observations == base + 4


def play_match(candidate, candidate_path: str, opponent_name: str, opponent_model,
               candidate_side: str, seed: int, max_decisions: int, procedural: bool,
               fog: bool, ticks_per_decision: int) -> dict:
    action_count, model_fog = model_profile(candidate)
    advanced = action_count == 36
    opponent = create_opponent(opponent_name) if opponent_model is None else SelfPlayOpponent(opponent_model, deterministic=True)
    env = MOBAEnv(
        opponent=opponent, seed=seed, max_steps=max_decisions,
        ticks_per_decision=ticks_per_decision, simulation_dt=0.1,
        procedural_map=procedural, fog_of_war=fog or model_fog,
        hierarchical_actions=action_count in {24, 36},
        strategic_actions=action_count in {17, 24, 36},
        advanced_decisions=advanced,
    )
    obs, info = env.reset(seed=seed)
    other_model = opponent_model
    if candidate_side == "red":
        # Blue is the pool policy and the candidate takes the mirrored Red seat.
        if other_model is None:
            raise ValueError("Seat swaps require a learned-policy opponent")
        env.opponent = SelfPlayOpponent(candidate, deterministic=True)
    policy = candidate if candidate_side == "blue" else other_model
    action_counts: Counter[int] = Counter()
    terminated = truncated = False
    total_reward = 0.0
    decisions = 0
    while not terminated and not truncated:
        masks = get_action_masks(env)
        action, _ = policy.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)
        if candidate_side == "blue":
            action_counts[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        if candidate_side == "red" and env.last_opponent_action is not None:
            action_counts[int(env.last_opponent_action)] += 1
        total_reward += float(reward)
        decisions += 1

    winner = info.get("winner")
    candidate_won = winner == candidate_side
    game = env.game
    side = Team.BLUE if candidate_side == "blue" else Team.RED
    champ = game.champions[side]
    enemy = game.champions[game.enemy_team(side)]
    enemy_towers = game.lane_towers[game.enemy_team(side)]
    row = {
        "candidate": candidate_path,
        "opponent": opponent_name,
        "opponent_type": "policy" if opponent_model is not None else "scripted",
        "candidate_side": candidate_side,
        "seed": seed,
        "game_seed": info.get("game_seed"),
        "winner": winner or "draw",
        "candidate_result": "win" if candidate_won else "loss" if winner else "draw",
        "decisions": decisions,
        "simulated_seconds": round(float(game.time), 2),
        "candidate_reward": round(total_reward if candidate_side == "blue" else 0.0, 5),
        "candidate_kills": champ.kills,
        "candidate_deaths": champ.deaths,
        "candidate_objectives": champ.objectives_taken,
        "candidate_towers_destroyed": sum(not tower.alive for tower in enemy_towers.values()),
        "candidate_final_gold": champ.gold,
        "opponent_final_gold": enemy.gold,
        "candidate_action_diversity": len(action_counts),
        "candidate_actions": json.dumps(dict(sorted(action_counts.items())), separators=(",", ":")),
        "procedural_map": procedural,
        "ticks_per_decision": ticks_per_decision,
    }
    env.close()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Candidate checkpoint, relative to project root")
    parser.add_argument("--opponent-models", nargs="*", default=[
        "models/opponent_pool/moba_advanced_big_20261023/opponent_000030000.zip",
        "models/opponent_pool/moba_advanced_big_20261023/opponent_000050000.zip",
    ], help="Historical opponent checkpoints; each is tested with candidate on both sides")
    parser.add_argument("--episodes", type=int, default=50, help="Seeded matches per opponent and seat")
    parser.add_argument("--seed", type=int, default=900000, help="First held-out seed (same seeds are paired across seats)")
    parser.add_argument("--max-decisions", type=int, default=5000)
    parser.add_argument("--ticks-per-decision", type=int, default=50)
    parser.add_argument("--procedural-maps", action="store_true", help="Use fixed held-out procedural map seeds")
    parser.add_argument("--scripted-opponents", nargs="+", choices=sorted(OPPONENTS), default=None,
        help="Limit scripted baselines to these names (defaults to all scripted opponents)")
    parser.add_argument("--scripted-only", action="store_true", help="Skip historical policy side-swap matches")
    parser.add_argument("--csv", default=None, help="Per-match CSV output path")
    parser.add_argument("--summary", default=None, help="Aggregate JSON output path")
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_decisions <= 0 or args.ticks_per_decision <= 0:
        parser.error("episodes, max decisions, and ticks per decision must be positive")

    candidate_path = (ROOT / args.model).resolve()
    if not candidate_path.is_file():
        parser.error(f"Candidate model not found: {candidate_path}")
    candidate = MaskablePPO.load(str(candidate_path), device="cpu")
    action_count, fog = model_profile(candidate)
    scripted_names = args.scripted_opponents or sorted(OPPONENTS)
    opponents: list[tuple[str, MaskablePPO | None]] = [(name, None) for name in scripted_names]
    if not args.scripted_only:
        for name in args.opponent_models:
            path = (ROOT / name).resolve()
            if not path.is_file():
                parser.error(f"Opponent model not found: {path}")
            old = MaskablePPO.load(str(path), device="cpu")
            if tuple(old.observation_space.shape) != tuple(candidate.observation_space.shape) or old.action_space.n != candidate.action_space.n:
                parser.error(f"Incompatible policy profile: {path}")
            opponents.append((name, old))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = (ROOT / args.csv).resolve() if args.csv else ROOT / "logs" / f"benchmark_{stamp}.csv"
    summary_path = (ROOT / args.summary).resolve() if args.summary else ROOT / "logs" / f"benchmark_{stamp}_summary.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    plans = sum(args.episodes * (1 if model is None else 2) for _, model in opponents)
    print(f"Candidate: {candidate_path.name} ({action_count} actions; fog={fog})")
    print(f"Matches: {plans} across {len(opponents)} opponents; paired held-out seeds start at {args.seed}", flush=True)
    done = 0
    for opponent_name, old_model in opponents:
        sides = ("blue",) if old_model is None else ("blue", "red")
        for side in sides:
            for index in range(args.episodes):
                seed = args.seed + index
                row = play_match(candidate, candidate_path.relative_to(ROOT).as_posix(), opponent_name,
                    old_model, side, seed, args.max_decisions, args.procedural_maps, fog,
                    args.ticks_per_decision)
                rows.append(row)
                done += 1
                with csv_path.open("a", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    if done == 1:
                        writer.writeheader()
                    writer.writerow(row)
                if done % 5 == 0 or done == plans:
                    print(f"[{done}/{plans}] {opponent_name} / {side}: {row['candidate_result']} seed={seed}", flush=True)

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["opponent"], []).append(row)
    summaries = {}
    for name, group in groups.items():
        wins = sum(row["candidate_result"] == "win" for row in group)
        draws = sum(row["candidate_result"] == "draw" for row in group)
        low, high = wilson(wins, len(group) - draws)
        summaries[name] = {"games": len(group), "wins": wins, "draws": draws,
            "losses": len(group) - wins - draws, "win_rate_excluding_draws": wins / max(1, len(group) - draws),
            "win_rate_including_draws_as_half": (wins + 0.5 * draws) / max(1, len(group)),
            "win_rate_95_ci_wilson": [round(low, 4), round(high, 4)],
            "mean_candidate_objectives": round(sum(int(r["candidate_objectives"]) for r in group) / len(group), 3),
            "mean_candidate_towers_destroyed": round(sum(int(r["candidate_towers_destroyed"]) for r in group) / len(group), 3),
            "mean_candidate_kills": round(sum(int(r["candidate_kills"]) for r in group) / len(group), 3),
            "mean_candidate_action_diversity": round(sum(int(r["candidate_action_diversity"]) for r in group) / len(group), 3),
            "by_side": {side: {
                "games": sum(r["candidate_side"] == side for r in group),
                "wins": sum(r["candidate_side"] == side and r["candidate_result"] == "win" for r in group),
                "draws": sum(r["candidate_side"] == side and r["candidate_result"] == "draw" for r in group),
                "win_rate_excluding_draws": (
                    sum(r["candidate_side"] == side and r["candidate_result"] == "win" for r in group)
                    / max(1, sum(r["candidate_side"] == side and r["candidate_result"] != "draw" for r in group))
                ),
            }
                for side in sorted({r["candidate_side"] for r in group})}}
    aggregate = {"candidate": str(candidate_path.relative_to(ROOT)), "seed_start": args.seed,
        "paired_fixed_seeds": True, "procedural_maps": args.procedural_maps,
        "episodes_per_opponent_and_side": args.episodes, "ticks_per_decision": args.ticks_per_decision,
        "matches": len(rows), "opponents": summaries, "match_csv": str(csv_path.relative_to(ROOT))}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    manifest_path = summary_path.with_name(summary_path.stem.replace("_summary", "") + "_manifest.json")
    source_files = sorted((ROOT / "moba_rl").rglob("*.py"))
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": "python -m moba_rl.evaluation.benchmark " + " ".join(sys.argv[1:]),
        "parameters": vars(args),
        "candidate": {
            "path": str(candidate_path.relative_to(ROOT)),
            "sha256": sha256_file(candidate_path),
            "actions": action_count,
            "observation_features": int(candidate.observation_space.shape[0]),
        },
        "opponent_checkpoints": [
            {"path": str((ROOT / name).relative_to(ROOT)), "sha256": sha256_file(ROOT / name)}
            for name, model in opponents if model is not None
        ],
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": package_version("torch"),
            "stable_baselines3": package_version("stable-baselines3"),
            "sb3_contrib": package_version("sb3-contrib"),
            "gymnasium": package_version("gymnasium"),
            "numpy": package_version("numpy"),
        },
        "outputs": {
            "summary": str(summary_path.relative_to(ROOT)),
            "summary_sha256": sha256_file(summary_path),
            "match_csv": str(csv_path.relative_to(ROOT)),
            "match_csv_sha256": sha256_file(csv_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "match_csv": str(csv_path),
        "manifest": str(manifest_path), "results": summaries}, indent=2))


if __name__ == "__main__":
    main()
