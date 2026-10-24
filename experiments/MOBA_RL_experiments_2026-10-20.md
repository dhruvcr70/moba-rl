# MOBA-RL experiments — 2026-10-20

All full evaluations used 20 episodes per opponent, seed base 1000, and the same per-opponent seed offsets. `models/moba_ppo_final.zip` was left untouched. Every newly trained model was a separate named run.

| Model / experiment | Aggressive | Defensive | Objective | Balanced | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing `models/moba_ppo_final.zip` | 8/20 | 0/20 | 0/20 | 0/20 | 10.00% |
| Fresh training, `gamma=0.999` | 0/20 | 0/20 | 0/20 | 0/20 | 0.00% |
| Lane progress, scale 5 | 20/20 | 0/20 | 0/20 | 0/20 | 25.00% |
| Objective-first progress, scale 5 | 0/20 | 1/20 | 0/20 | 0/20 | 1.25% |
| Lane progress 5 + direct damage 0.002 | 20/20 | 6/20 | 0/20 | 0/20 | **32.50%** |
| Objective-first + direct damage 0.002 | 1/20 | 1/20 | 0/20 | 0/20 | 2.50% |
| Lane progress 5 + damage 0.002 + structure penalty 0.001 | 20/20 | 0/20 | 0/20 | 0/20 | 25.00% |
| Lane progress 5 + damage 0.005, seed 42 | 20/20 | 20/20 | 0/20 | 0/20 | **50.00%** |
| Lane progress 5 + damage 0.005, seed 43 | 20/20 | 1/20 | 0/20 | 0/20 | 26.25% |
| Objective-first + damage 0.005, seed 42 | 20/20 | 1/20 | 0/20 | 0/20 | 26.25% |

## Findings

- The baseline agent spent many decisions moving or doing nothing and seldom attacked a strategic target.
- Raising gamma alone did not help.
- Lane-progress shaping taught forward movement and raised the measured win rate to 25%, but it did not solve defense or objective play.
- Adding a small reward for direct damage to the enemy champion, tower, base, or objective improved the lane-progress model to 32.5%. It won 6/20 against the defensive opponent, plus 20/20 against aggressive.
- Objective-first movement performed poorly, even when paired with direct-damage reward. Keep `lane` as the default target mode.
- Penalizing Blue tower/base damage erased the defensive matchup gains in the tested run. Do not turn that option on by default.
- All reward-shaping options default to 0, preserving the prior environment behavior. Target mode defaults to `lane`.
- The lane-progress 5 + damage 0.005 run reached 50% with training seed 42, but its seed 43 repeat reached only 26.25%; both runs scored 20/20 against aggressive and 0/20 against objective and balanced. The gain is seed-sensitive and does not address those two opponents. The objective-first + damage 0.005 run also scored 26.25%.

## Best tested settings

The highest single-run result is lane progress plus direct damage at scale 0.005 (50%, training seed 42). It did not repeat at seed 43 (26.25%). The scale 0.002 candidate scored 32.5% at seed 42 and 26.25% at seed 43. All settings remain preliminary. The highest run used:

- `progress_reward_scale=5.0`
- `progress_target_mode=lane`
- `damage_reward_scale=0.005`
- `structure_loss_penalty_scale=0.0`

Models: `models/damage_reward_005_final.zip` (single-run high) and `models/damage_reward_005_seed43_final.zip` (repeat).

## Reproduce the 0.005 seed 42 training run

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m moba_rl.training.train_ppo `
  --run-name damage_reward_005 `
  --timesteps 50000 `
  --seed 42 `
  --eval-freq 50000 `
  --eval-episodes 1 `
  --checkpoint-freq 50000 `
  --progress-reward-scale 5.0 `
  --progress-target-mode lane `
  --damage-reward-scale 0.005
```

The run name keeps the model, checkpoints, and evaluation log separate from the original model.

Evaluate with the same shaping settings and 20 episodes per opponent:

```powershell
.\.venv\Scripts\python.exe -m moba_rl.evaluation.evaluate `
  --model models/damage_reward_005_final.zip `
  --episodes 20 `
  --seed 1000 `
  --progress-reward-scale 5.0 `
  --progress-target-mode lane `
  --damage-reward-scale 0.005 `
  --csv logs/damage_reward_005_evaluation_2026-10-20.csv
```

## Current verification

`.\.venv\Scripts\python.exe -m pytest`: 47 passed.

The next investigation should focus on observing and responding to opponent strategy: the learned policy rushes lane and loses to objective and balanced styles across both tested seeds. Avoid claiming the 50% score as a general improvement until additional seeds or a training change improve those matchups.


## Project foundation progress (2026-10-20)

The reward-scale investigation above remains on hold. The project now also has an installable package, setup and usage documentation, deterministic procedural maps, optional fog-of-war observations, expanded strategic actions, curriculum/self-play training support, replay recording and analysis, Pygame rendering, optional W&B tracking, a FastAPI/WebSocket inference service, a React/Tailwind arena, and Docker deployment files.

Verification after these additions: `pytest` reports 67 passed. The React production build succeeds and `npm audit` reports zero vulnerabilities. The report generator produced evaluation, training-progress, and replay summaries. A saved fog-of-war policy loaded into the API and completed an agent-controlled step. W&B completed an offline 1,024-timestep smoke run. The combined integration model uses 37 observation features and 12 actions; strategic-action models use 39 features, or 43 when combined with fog of war, and 17 actions.

Deployment checks still needing suitable hardware/tools: CUDA training, online W&B synchronization, and Docker image builds. The current machine has CPU-only PyTorch and no Docker CLI. The API test client also reports an upstream Starlette/httpx deprecation warning; this does not fail tests.

## CUDA self-play and curriculum run (2026-10-20)

Run: `gpu_selfplay_curriculum_50k`, seed 42, 50,000 requested steps (50,176 collected due to 1,024-step PPO rollouts), curriculum and self-play enabled, with procedural maps and fog of war. PPO auto-selected CUDA on the NVIDIA T600 Laptop GPU using PyTorch 2.9.1+cu128.

The checkpoint evaluations at 10k, 20k, 30k, 40k, and 50k steps each reported a 0% win rate. A separate final evaluation on seeds 10,000–10,009 also reported 0/10 wins against each scripted opponent (aggressive, defensive, objective, balanced), 40 matches total. Mean rewards in the final evaluation were -2.645, -4.304, -5.604, and -5.713 respectively. This experiment did not improve win rate and should not replace the baseline. Reward-scale tuning remained unchanged.

Artifacts: `models/gpu_selfplay_curriculum_50k_final.zip`, `models/best/gpu_selfplay_curriculum_50k_best.zip`, `logs/gpu_selfplay_curriculum_50k_multi_opponent_evaluation.csv`, and `logs/gpu_selfplay_curriculum_50k_final_10seed_eval.csv`. Training device verification and the full 67-test suite passed under CUDA PyTorch.

## Replay diagnosis (2026-10-20)

Recorded one held-out procedural/fog match against each scripted opponent for the 50k self-play policy and one standard-map match against each for the original baseline. These four-match-per-policy traces are diagnostic samples, not performance estimates.

The self-play policy selected `move_left` on 5,224 of 5,260 turns (99.3%; per-match range 98.8–99.6%), used no attacks on champions, towers, or the objective, and walked to x=0 in every match. Objective captures were 0, 0, 1, and 0 by opponent (the one capture was by the scripted Red opponent). Red minions destroyed Blue's base in all four traces. The policy gained some passive XP/gold from lane activity, but did not use the farm/shop actions because this model has the default 12-action space.

In the baseline traces, Blue moved right 2,720 times, moved left 846 times, moved up 1,851 times, and attacked the enemy champion 38 times in one match. It still lost all four single-match checks; the baseline remains preferable only as the comparison point, not a proven winner. The sharp retreat/action collapse appears specific to the new fog/procedural self-play run and merits checking for policy-observation/reward effects.

Likely contributing condition: with default reward scales set to zero, most lane progress and structure damage happens autonomously through minions, while survival gives a small per-step positive reward. That makes passive survival competitive with taking risky strategic actions. Reward-scale tuning is still on hold. Next controlled experiment should keep these reward settings fixed, enable `--strategic-actions`, and compare action distributions/replays under otherwise matched training settings before committing to another full run.

Replay files: `logs/gpu_selfplay_curriculum_50k_diagnostic_replays/` and `logs/baseline_diagnostic_replays/`. Reports: `reports/gpu_selfplay_curriculum_50k_diagnostic/` and `reports/baseline_diagnostic/`.

## Reward-scale follow-up (2026-10-20, CUDA)

Resumed the held reward-scale investigation with matched 50k-step PPO runs on standard maps, no fog/procedural maps/self-play, progress target `lane`, and progress reward scale 5.0. PyTorch auto-selected the NVIDIA T600. The coefficient varied between candidates; each model used a separate seed and run name. Full evaluations used 20 matches per opponent with the same seed base (1000).

| Direct-damage scale | Training seed(s) | Wins / 80 per seed | Mean across seeds | Aggressive | Defensive | Objective | Balanced |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005 | 42 (prior), 43 (prior), 44 (CUDA repeat) | 82 / 240 total | 34.17% | 60/60 | 22/60 | 0/60 | 0/60 |
| 0.01 | 45 (CUDA) | 22 / 80 | 27.50% | 20/20 | 2/20 | 0/20 | 0/20 |

Seed 44 at scale 0.005 scored 26.25%: aggressive 20/20, defensive 1/20, objective 0/20, balanced 0/20. Seed 45 at 0.01 scored 27.50%: aggressive 20/20, defensive 2/20, objective 0/20, balanced 0/20. The higher coefficient did not provide convincing improvement; its single seed is close to the lower coefficient's repeats. Scale 0.005 remains the best candidate by multi-seed mean, but results are seed-sensitive and do not generalize to objective or balanced opponents. Keep reward scales experimental; do not promote a default from this evidence.

Artifacts: `models/damage_reward_005_seed44_cuda_final.zip`, `models/damage_reward_010_seed45_cuda_final.zip`, `logs/damage_reward_005_seed44_cuda_evaluation_2026-10-20.csv`, and `logs/damage_reward_010_seed45_cuda_evaluation_2026-10-20.csv`.

## Strategic-action, held-out evaluation, and vectorization follow-up (2026-10-20)

Added configurable vector rollout environments (`--n-envs`, `--vec-env dummy|subproc`) and an adaptive scripted opponent used only during evaluation. Multi-environment curriculum and self-play callbacks were smoke-tested with two in-process environments; four subprocess environments were smoke-tested separately. CUDA PPO correctly selected the NVIDIA T600. Subprocess self-play is intentionally rejected on Windows because workers cannot safely share the live CUDA policy; use `dummy` for multi-env self-play.

| Run | Configuration | Result |
| --- | --- | --- |
| `throughput_dummy_10k` | 10,240 steps, 1 dummy env, seed 800 | 38.03 s; 327 logged FPS |
| `throughput_subproc4_10k` | Same workload, 4 subprocess envs | 26.45 s; 704 logged FPS |
| `strategic_heldout_cuda50k` | 50,000 steps, 4 subprocess envs, curriculum, procedural maps, fog, 17 actions, seed 46 | Completed on CUDA; 25% mean training-time evaluation win rate (5 games per training opponent) |

The throughput comparison is a single-machine measurement, not a general hardware claim. Four subprocess workers reduced elapsed time by about 30% (38.03 s to 26.45 s, or 1.44x faster); their reported PPO FPS was 704 versus 327 for one dummy environment. The strategic model's checkpoint evaluation was 5/5 against aggressive and 0/5 against defensive, objective, and balanced opponents.

The final model was then evaluated for 5 episodes against each of the four scripted opponents plus the held-out adaptive opponent on procedural maps with fog of war (25 matches, seeds 50,000 onward). Results: aggressive 5/5, defensive 0/5, objective 0/5, balanced 0/5, adaptive 0/5; overall 20%. Replay review found 33,285 of 35,128 Blue actions (94.7%) were left/right movement, 824 (2.3%) were farming, 105 bought boots, and none were direct attacks. The model is not strategically competent yet; its action-use collapse is the next research bottleneck, and no reward setting should be promoted based on these results.

Artifacts: `models/strategic_heldout_cuda50k_final.zip`, `logs/strategic_heldout_procedural_eval.csv`, `logs/strategic_heldout_replays/`, and the generated report/plots in `reports/strategic_heldout/`. The frontend production build succeeded. 62 simulation/environment tests passed. API session create/step/cleanup passed through direct in-process calls; HTTP TestClient and local socket checks could not be completed in this managed Windows environment because requests hang and socket binding is denied (`WinError 10013`).

To isolate the collapse, ran `strategic_shaped_isolation_50k`: 50,000 steps, seed 47, four subprocess environments, strategic actions and curriculum enabled, standard maps/no fog, lane progress scale 5.0, direct-damage scale 0.005. Training-time evaluation reached 5/5 wins against aggressive and 5/5 against defensive. A separate 25-match evaluation (five per opponent including adaptive) scored 5/5 aggressive, 5/5 defensive, and 0/5 objective, balanced, and adaptive, for 40% overall. This is a clear gain on the two tested matchups, but no evidence yet of broad strategic competence. Keep the exact run settings experimental until repeated across seeds; the remaining objective/balanced/adaptive losses need targeted work.

Artifacts: `models/strategic_shaped_isolation_50k_final.zip` and `logs/strategic_shaped_isolation_eval.csv`.

## Objective-control rewards and hierarchical actions (2026-10-20)

Added an optional `objective_reward_scale` to training and evaluation. It applies a zero-sum bonus when either team captures the neutral objective and defaults to 0. Replay reports now split Blue and Red capture counts. A coefficient of 3 did not reproduce: seed 48 reached 48% over 25 standard-map matches, while seed 49 reached 0% on a 10-match spot evaluation. Coefficient 10 produced 100% on one seed's 20 standard-map training-evaluation matches, but only 28% over 25 procedural-map matches; Red captured the objective in all ten objective/adaptive matches. The reward alone did not solve objective play.

Added an optional `--hierarchical-actions` profile with five high-level actions: engage champion, push lane, contest objective, farm lane, and return to shop. At that point it had 22 discrete actions. Macros are team-relative and supported by self-play; enemy pursuit in fog uses visible or remembered locations. The objective macro attacks a visible nearby enemy who is contesting before continuing the objective attempt. Existing default and 17-action models retain their action spaces. A 2,048-step CUDA smoke run completed, and the hierarchical 43-observation/22-action model loaded into an API match, returned 22 labels/masks, took an agent-controlled step, and closed cleanly in-process.

Three 50k-step CUDA seeds were trained with the same hierarchy, curriculum, procedural maps, fog, progress scale 5.0 toward `objective_then_lane`, direct damage scale 0.005, and objective capture/denial scale 10.0. External evaluation used five episodes against each of five styles (aggressive, defensive, objective, balanced, adaptive) on procedural maps with fog:

| Seed | Training-time win rate vs four training styles | External wins / 25 | External win rate |
| ---: | ---: | ---: | ---: |
| 52 | 75% | 22 / 25 | 88% |
| 53 | 50% | 6 / 25 | 24% |
| 54 | 25% | 5 / 25 | 20% |
| Combined | — | 33 / 75 | 44% |

Combined per-opponent results were aggressive 6/15, defensive 15/15, objective 5/15, balanced 2/15, and adaptive 5/15. Seed 52 is a strong single run but the seed spread is too large to call the result robust. Replay summaries also show macro-action collapse: seed 52 selected `engage_champion` 30,883 times; seed 53 selected `contest_objective` 26,793 times; seed 54 selected it 31,506 times. Capturing the objective by itself does not guarantee a win: seed 54's Blue side captured several objectives but still lost every match except against defensive. The next research bottleneck is learning when to switch strategic modes and measuring progress over more seeds, rather than raising the objective bonus further.

Artifacts: `models/hier_selfplay_combat_seed52_final.zip`, `models/hier_selfplay_combat_seed53_final.zip`, `models/hier_objective_defense_seed54_final.zip`, their evaluation CSVs and replay folders in `logs/`, and reports in `reports/hier_selfplay_combat_seed52/`, `reports/hier_selfplay_combat_seed53/`, and `reports/hier_objective_defense_seed54/`.

Verification for the hierarchy work: `compileall` succeeded; a 2,048-step training smoke run completed; the API's in-process agent session matched the new model's 43-feature observations, 22-action mask, and 22 action names; and the evaluator and analysis pipeline completed. A Pygame RGB frame from a 24-step procedural/fog match was saved as `reports/hierarchical_pygame_demo.png`. The pytest suite was not rerun after these code changes. Live browser/API checks remain blocked by localhost socket restrictions in this managed Windows session, and Docker is unavailable here.

Next substantive research work: investigate action-mask/observation/reward dynamics causing movement-only behavior, then compare fixes across multiple seeds and held-out procedural/fog scenarios. The repo includes the Pygame viewer, React arena, FastAPI/WebSocket service, and Docker deployment files; end-to-end browser/live API deployment and Docker image build still need execution in an environment that allows localhost sockets and has Docker installed.

## Defensive macro actions (2026-10-21)

Extended the hierarchical action profile from 22 to 24 actions. `defend_lane` attacks a visible champion threatening the home tower, clears nearby opposing minions, or moves back to protect the lane; after the tower falls, the destination falls back to the home base. `retreat_to_base` moves the champion back to base without spending gold. Both are available to BLUE and RED through the shared team-relative executor and action masks. The API reports the 24-action profile, and the React setup text lists defend and retreat. Existing 22-action checkpoints are still available as files but cannot be loaded with the new 24-action setting; use the matching profile when loading those older models.

Trained `hierarchical_defense_retreat_20261021` for 50,000 steps on the NVIDIA T600 with seed 52, curriculum and self-play, procedural maps, fog, progress scale 5.0 toward `objective_then_lane`, direct damage scale 0.005, and objective control scale 10.0. The 10k, 20k, 30k, 40k, and 50k checkpoint win rates against the four training opponents were 85%, 25%, 35%, 55%, and 75%. The best checkpoint was saved at 10k; the final model was not selected as best.

Fresh held-out evaluation of the best checkpoint used 10 episodes per opponent on procedural maps with fog (50 matches, seeds starting at 60,000): aggressive 10/10, defensive 10/10, objective 10/10, balanced 4/10, adaptive 10/10; total 44/50 (88%). The balanced matchup remains weak, and this is one training seed, so it is a candidate result rather than evidence of robust improvement. Artifacts: `models/best/hierarchical_defense_retreat_20261021_best.zip`, `models/hierarchical_defense_retreat_20261021_final.zip`, `logs/hierarchical_defense_retreat_20261021_multi_opponent_evaluation.csv`, and `logs/hierarchical_defense_retreat_20261021_heldout_aligned.csv`.

## Current project scope (2026-10-24)

This log preserves earlier milestones, including experiments that were not promoted and deployment work that was not completed. Docker was subsequently dropped from the project scope; the old Docker references above are historical and do not describe a supported deployment path. The current local demo uses the FastAPI/WebSocket service and React arena. Training updates one PPO policy against scripted, saved, or self-play snapshot opponents; it is not distributed training or two independently learning teams. See the root `README.md` and `reports/moba_advanced_big_20261023/RESULTS.md` for the current setup and measured policy limitations.
