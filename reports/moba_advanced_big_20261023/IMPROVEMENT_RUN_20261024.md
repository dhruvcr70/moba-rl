# Fixed-map matchup improvement run — 2026-10-24

## What changed

Added reproducible training controls in `moba_rl/training/train_ppo.py`:

- `--mixed-maps` samples either the fixed or procedural layout on each training episode.
- `--opponent-weights` changes the sampling balance among the four scripted opponents.
- Each training reset uses the wrapper's seeded RNG, so map and opponent selection are reproducible.
- Training-time evaluation remains on the fixed map, preserving a direct measure of the weak condition.

Trained a new 50,000-timestep run with seed 101, 36-action decisions, fog of war, self-play snapshots, a 50/50 fixed/procedural map mix, and weights `balanced=2 defensive=2 aggressive=1 objective=1`. The original checkpoint was not overwritten. The run's best checkpoint was selected at 40,000 steps by the existing mean-win-rate callback.

## Paired held-out results

Both checkpoints were evaluated on the same 50 seeds within each map condition. The candidate was Blue in all matches. The 920000 seed range was fixed-map; 930000 was procedural.

| Map | Opponent | Original | New best | New minus old |
|---|---|---:|---:|---:|
| Fixed | Balanced | 8/50 (16%) | 16/50 (32%) | +16 points |
| Fixed | Defensive | 0/50 (0%) | 0/50 (0%) | 0 points |
| Procedural | Balanced | 34/50 (68%) | 30/50 (60%) | -8 points |
| Procedural | Defensive | 30/50 (60%) | 18/50 (36%) | -24 points |

The fixed-map Balanced result is a promising direction but not conclusive: the paired exact two-sided sign test on the 50 shared seeds gives p=0.134. Defensive performance did not improve on the fixed map. Procedural performance was lower for the new policy, especially against Defensive. These results do not support replacing the original checkpoint.

## Behavior diagnosis

The new policy's mean distinct-action count was only 2.0 per match in all four final comparisons, down from 5.1–6.0 for the original policy. Its action log was dominated by `contest_objective` and `wait`. It did capture the neutral objective, averaging 1.0 per fixed-map match and 1.0–1.62 on procedural maps, but this narrow behavior coincided with poorer procedural win rates. The next iteration should address policy collapse and defensive-opponent counterplay before increasing training duration or claiming a general improvement.

At 50,000 steps, the trainer's 5-episode-per-opponent fixed-map check reported 100% against Aggressive, 0% against Defensive, 100% against Objective, and 100% against Balanced. This small checkpoint sample overestimated the paired held-out results and shows why the larger seeded benchmark is needed.

## Artifacts

- Follow-up fixed-map diagnostic (50 new seeds each): `logs/benchmark_fixed_followup_50*`.
- Paired fixed-map comparisons: `logs/benchmark_fixed_comparison_{original,mixed}_50*`.
- Paired procedural-map comparisons: `logs/benchmark_procedural_comparison_{original,mixed}_50*`.
- Training checkpoint: `models/best/fixed_matchup_mixed_seed101_best.zip`.
- Final training checkpoint: `models/fixed_matchup_mixed_seed101_final.zip`.
- Training evaluation history: `logs/fixed_matchup_mixed_seed101_multi_opponent_evaluation.csv`.
- Candidate hashes, runtime details, and source hashes are in each benchmark manifest JSON.
