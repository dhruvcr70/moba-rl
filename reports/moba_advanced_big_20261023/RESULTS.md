# MOBA-RL evaluation results

## Evaluation setup

- Candidate: `models/moba_advanced_big_20261023_final.zip` (MaskablePPO, 36 actions, 57 observation features, 50,176 training timesteps).
- 450 deterministic matches: 225 on the fixed map and 225 on procedural maps.
- Both map conditions use the same held-out seeds, 900000–900024.
- Each map condition tests five scripted opponents for 25 matches each. It also tests two historical self-play checkpoints with the candidate in both seats, 25 matches per seat.
- `adaptive` is held out from the saved training opponent list. The historical checkpoints come from the candidate's own training run and are not independent agents.
- Per-match results, summaries, and reproducibility manifests are saved under `logs/benchmark_20261024_*_25*`.

## Findings

The candidate's win rate depends heavily on the map condition against two scripted baselines:

| Opponent | Fixed map | Procedural maps | Fixed minus procedural, paired seeds |
|---|---:|---:|---:|
| Balanced | 2/25 (8%) | 20/25 (80%) | -72 percentage points |
| Defensive | 1/25 (4%) | 13/25 (52%) | -48 percentage points |
| Held-out adaptive | 25/25 (100%) | 25/25 (100%) | 0 points |

Across all 450 matches, the candidate captured the neutral objective **0 times**. It won primarily by destroying structures. Its mean distinct action count was about 4.8 per match, so the high win rates against several opponents should not be read as broad strategic mastery.

Against the historical 50,000-step checkpoint, the candidate won 10/25 matches as Blue and 22/25 as Red on the fixed map; on procedural maps it won 14/25 as Blue and 23/25 as Red. This seat difference is large and should be investigated before making claims about balanced self-play.

## What the results support

- The checkpoint beats the held-out adaptive baseline on both tested map conditions in this 25-seed sample.
- It performs poorly against balanced and defensive baselines on the fixed map, despite better results against those baselines on procedural maps.
- The neutral-objective strategy is not demonstrated by this checkpoint.
- The training-time CSV has eight checkpoints but only one evaluation episode per opponent per checkpoint. It shows progress, but it is too small to establish stable learning across training seeds.

See [`report.md`](report.md) for the full per-opponent table, confidence intervals, training curve, and plots. The benchmark inputs and run manifests are recorded in [`benchmark_inputs.json`](benchmark_inputs.json); the matched map comparison is in [`paired_map_comparison.csv`](paired_map_comparison.csv).

## Recommended next improvement cycle

Train across a deliberate mix of fixed and procedural layouts, increase objective-focused feedback only after checking that the reward does not encourage degenerate behavior, and evaluate each candidate against the same held-out seeds in both seats. Use at least three independent training seeds and at least 10 evaluation episodes per opponent at each checkpoint; keep a separate final test seed range for model selection.

## Follow-up status

The first mixed-map, defensive-weighted run and a paired 50-seed comparison on each map condition are now complete. Fixed-map Balanced performance rose from 16% to 32%, but fixed-map Defensive stayed at 0%; procedural rates fell and the new policy collapsed to two actions. The original checkpoint remains the recommended demo model. See [`IMPROVEMENT_RUN_20261024.md`](IMPROVEMENT_RUN_20261024.md) for results, artifacts, and the diagnosis.
