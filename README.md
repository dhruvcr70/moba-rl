# MOBA-RL

MOBA-RL is a three-lane, League-inspired simulation for reinforcement-learning research. Blue and Red champions make decisions on a seeded map with top, mid, and bottom lanes; lane minions, towers and bases; champion combat and abilities; upgrades; and a neutral objective. The web arena centers play around strategic orders and exposes the map state and recent outcomes so a person can understand what each agent is doing.

The simulation advances in short ticks. In the strategic arena, each player can commit to an order for five simulated seconds before choosing again, or select real-time mode where the game clock runs live and orders refresh every half-second. The seven main orders are engage, push, contest the objective, farm, recall and shop, defend, and retreat. Tactical actions remain available for direct control. It is a League-inspired research simulation, not a production-scale commercial MOBA.

The Gymnasium environment supports flat and hierarchical action spaces, scripted opponents, procedural maps, fog of war, reward shaping, and trajectory recording. Advanced orders add lane rotations, freeze/slow/fast wave control, warding and sweeps, objective zoning, all-in and poke plans, and cooldown conservation. PPO and MaskablePPO training, evaluation, checkpointing, and optional experiment tracking are provided in `moba_rl/training/`. Existing 24-action/43-feature policies remain compatible; advanced policies use 36 actions and 57 features. The repository also contains an experimental territory-control environment, separate from the primary MOBA game.

## Project status and evidence

This is an end-to-end local research prototype. One trainable PPO policy learns against scripted opponents, saved policies, or self-play snapshots; the two teams are not independently updated learners, and training is not distributed across machines. The simulation supports continuous real-time ticks, while the agents choose orders at intervals. It is not a full commercial MOBA engine.

The evaluation pipeline supports fixed and procedural maps, held-out opponents, seat swaps, replay analysis, and reproducibility manifests. The current results also show important limitations: the main checkpoint won 2/25 fixed-map matches against Balanced and 1/25 against Defensive in the reported comparison, and captured the neutral objective 0 times across 450 matches. A later experimental run improved one fixed-map matchup but regressed on some procedural-map matchups, so it was not promoted. See [the results summary](reports/moba_advanced_big_20261023/RESULTS.md) and [the follow-up report](reports/moba_advanced_big_20261023/IMPROVEMENT_RUN_20261024.md) before describing policy strength.

The supported demo runs locally through FastAPI/WebSockets and the React arena. TensorBoard logging is included; Weights & Biases is optional. Docker deployment is outside the current project scope.

## Setup (Windows PowerShell)

Requires Python 3.11, 3.12, or 3.13.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

The trainer selects CUDA automatically when PyTorch detects a compatible GPU. Check the current installation with:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

## Train a MOBA policy

Trained policy checkpoint archives are not included in this public source release. Train a policy with the command below, or supply your own compatible `.zip` checkpoint to the evaluation and arena commands. The committed reports document earlier runs; their model files are not available in this repository.

For a small first run with the full strategic action set:

```powershell
python -m moba_rl.training.train_ppo `
  --run-name moba_advanced `
  --timesteps 100000 `
  --hierarchical-actions `
  --advanced-decisions `
  --ticks-per-decision 50 `
  --fog-of-war `
  --n-envs 4
```

The trainer uses PyTorch's automatic device selection, so it will use CUDA when available. Training options also include curriculum, self-play, procedural maps, and experiment tracking; run `python -m moba_rl.training.train_ppo --help` to see the flags available in the installed version. Checkpoints and logs are written under `models/`, `checkpoints/`, and `logs/`.

The arena lists only compatible checkpoints that you have trained or placed under `models/`. Existing hierarchical policies use 24 actions and 43 observation features; advanced policies use 36 actions and 57 features. Selecting a checkpoint enables the matching action set and fog-of-war features. The arena runs loaded policies on CPU so they can be inspected without consuming GPU memory needed for training.

## Run the strategic arena

Open one PowerShell window at the repository root and start the API:

```powershell
python -m uvicorn moba_rl.api.app:app --host 127.0.0.1 --port 8005
```

In a second window:

```powershell
cd frontend
npm install
npm run dev
```

Open the local address printed by Vite (normally `http://localhost:5173`). Choose a scripted Red opponent and create a match. In human-control mode, choose a lane order, advanced strategic order, or explicit item purchase. To watch a compatible trained policy, select it under Blue Policy and use **Next order** or **Run continuously**. With decision view enabled, the arena pauses after each order. The map, visible wards, wave plans, champion and tower health, context cards, policy action probabilities, event log, and order history show game conditions and outcomes.

For a continuously advancing match, enable **Real-time combat** before creating it. The game clock advances in 0.1-second simulation ticks and the interface updates every 0.5 seconds. Trained policies commit to an order for up to five simulated seconds, matching their training interval; an order can be reconsidered sooner if it becomes unavailable. The order log records changes in plan rather than every live update. Human orders persist until replaced.

If the frontend uses a different API address, set `VITE_API_URL` before starting Vite, for example:

```powershell
$env:VITE_API_URL = "http://127.0.0.1:8005"
npm run dev
```

The arena also shows the latest completed opponent evaluations from `logs/`. During a match, **Download replay** exports the trajectory as JSON. Replays can be inspected frame by frame with the Pygame viewer:

```powershell
python -m moba_rl.visualization.replay logs\my_match.json --fps 10
```

## Evaluate and analyze experiments

Evaluation reads the policy dimensions from the checkpoint and builds the matching action and observation profile. Match `--ticks-per-decision` to the setting used for the policy; five-second orders use `50` ticks:

```powershell
python -m moba_rl.evaluation.evaluate `
  --model path\to\your_policy.zip `
  --episodes 5 `
  --procedural-maps `
  --ticks-per-decision 50 `
  --replay-dir logs\latest_replays `
  --csv logs\latest_evaluation.csv
```

Evaluation output includes wins, objective captures, towers destroyed, champion kills/deaths, final gold and levels, ward and wave-plan usage, action diversity, and advanced-action rate. Create a report and charts with:

For match-level evidence, run the paired benchmark. It evaluates all five scripted baselines and can also compare optional self-play checkpoints using the same held-out seeds for both policy seats. Checkpoints are not bundled in this repository; use `--scripted-only` unless you provide compatible policies. It writes every game to a CSV and a summary with Wilson 95% confidence intervals. `--episodes 50` is the default; a quick smoke run can use fewer games.

```powershell
python -m moba_rl.evaluation.benchmark `
  --model path\to\your_policy.zip `
  --episodes 50 `
  --seed 900000 `
  --procedural-maps `
  --scripted-only
```

The benchmark outputs a per-match CSV, summary JSON, and reproducibility manifest under `logs/`. The manifest records the full command, seed/map settings, Python and library versions, and SHA-256 hashes of the candidate, opponent checkpoints, core benchmark/environment source files, and output files. Pass `--csv` and `--summary` to choose output locations, or `--scripted-only` to omit historical policy seat swaps.

To compare the fixed map with procedural maps using the same held-out seed range, run two benchmark jobs with separate output names. Use `--episodes 50` for a larger comparison; the committed report uses 25 matches per opponent and map condition, paired by seed.

```powershell
python -m moba_rl.evaluation.benchmark `
  --episodes 50 `
  --seed 900000 `
  --csv logs\benchmark_fixed.csv `
  --summary logs\benchmark_fixed_summary.json

python -m moba_rl.evaluation.benchmark `
  --episodes 50 `
  --seed 900000 `
  --procedural-maps `
  --csv logs\benchmark_procedural.csv `
  --summary logs\benchmark_procedural_summary.json
```

Build a report from both map conditions and the training-time evaluation log:

```powershell
python -m moba_rl.analysis.report `
  --benchmarks logs\benchmark_fixed.csv logs\benchmark_procedural.csv `
  --training-csv logs\moba_advanced_big_20261023_multi_opponent_evaluation.csv `
  --out-dir reports\moba_advanced_big_20261023
```

The checked-in run report is in `reports/moba_advanced_big_20261023/`; start with `RESULTS.md` for the main findings. Its 25-seed paired map comparison includes `adaptive`, a held-out scripted opponent not listed in the training opponent pool. The candidate won 8% of fixed-map matches against the balanced baseline and 4% against the defensive baseline, versus 80% and 52% respectively on procedural maps. It captured the neutral objective in 0 of 450 matches. These results expose map sensitivity and a strategic-objective weakness despite stronger aggregate wins against other opponents; they are the main targets for the next policy improvement cycle.

A follow-up fixed-map check on 50 new seeds (910000–910049) confirmed the matchup weakness: the same checkpoint won 7/50 (14%) against `balanced` and 0/50 against `defensive`. Full per-match data and a hash manifest are in `logs/benchmark_fixed_followup_50*`. `train_ppo.py` supports `--mixed-maps` and `--opponent-weights` for follow-up runs. For example:

```powershell
python -m moba_rl.training.train_ppo `
  --run-name fixed_matchup_mixed_seed101 `
  --seed 101 `
  --timesteps 50000 `
  --advanced-decisions `
  --fog-of-war `
  --mixed-maps `
  --opponent-weights balanced=2 defensive=2 aggressive=1 objective=1 `
  --self-play `
  --eval-episodes 5
```

The training evaluation callback stays on the fixed map, so it tracks the weakness being targeted while training episodes alternate between fixed and procedural layouts. Compare any new checkpoint against the saved candidate on an untouched held-out seed range before treating an increase as evidence of improvement.

The first mixed-map follow-up run and paired 50-seed comparisons are documented in `reports/moba_advanced_big_20261023/IMPROVEMENT_RUN_20261024.md`. It raised fixed-map Balanced wins from 16% to 32%, but left fixed-map Defensive at 0% and lowered procedural-map rates. The new checkpoint is therefore kept as an experiment and does not replace the original model.

Treat the training curve as descriptive: the saved callback CSV has one evaluation episode per opponent at each of eight checkpoints, so it does not estimate training variance across independent training seeds. The final checkpoint contains 50,176 PPO timesteps.

```powershell
python -m moba_rl.analysis.report `
  --evaluations logs\latest_evaluation.csv `
  --training-csv logs\hierarchical_defense_retreat_20261021_multi_opponent_evaluation.csv `
  --replays logs\latest_replays `
  --out-dir reports\latest
```

The replay viewer renders the saved map, active lane plans, wards, units, and structures. Analysis writes evaluation summaries, decision-quality metrics, action-frequency charts, and replay summaries to the report folder. The arena’s **Latest evaluations** panel reads saved training evaluation CSVs from `logs/`.

## Training metrics

TensorBoard logs can be opened with:

```powershell
python -m tensorboard.main --logdir logs
```

Evaluation artifacts are written with each run when the corresponding evaluation options are enabled. Compare policies using the same seeds, opponent set, and episode count.

Training supports TensorBoard by default and optional Weights & Biases logging through `--wandb`. To make a follow-up run comparable, keep the environment and reward flags fixed, use a new run name, record the seed, and evaluate at least 10 episodes per opponent. Repeat the run with multiple training seeds before attributing gains to a code or reward change. Set `WANDB_API_KEY` in the shell instead of saving credentials in project files.

```powershell
$env:WANDB_API_KEY = "<your key>"
python -m moba_rl.training.train_ppo `
  --run-name advanced_seed101 `
  --seed 101 `
  --timesteps 50000 `
  --n-envs 4 `
  --hierarchical-actions `
  --advanced-decisions `
  --ticks-per-decision 50 `
  --fog-of-war `
  --curriculum `
  --self-play `
  --eval-episodes 10 `
  --wandb `
  --wandb-project moba-rl
```

Change the run name and seed for each independent replicate. The training callback CSV is useful for watching progress, but its small per-checkpoint evaluation sample should not be treated as final benchmark evidence.

The GitHub Actions workflow runs the Python suite and frontend production build on pushes and pull requests.

To run those checks locally from the repository root:

```powershell
python -m pytest
Push-Location frontend
npm ci
npm run build
Pop-Location
```

## Repository layout

- `moba_rl/simulation/`: map, champions, minions, towers, objective, combat, and match rules.
- `moba_rl/environment.py`: Gymnasium environment, observations, action masks, and strategic orders.
- `moba_rl/agents/scripted.py`: scripted opponent baselines.
- `moba_rl/training/`: PPO training and evaluation entry points.
- `moba_rl/api/`: FastAPI and WebSocket match service.
- `frontend/`: React strategic arena.
- `moba_rl/strategy/territory.py`: separate experimental territory-control game.
- `tests/`: automated checks for the simulation and environment.
