from __future__ import annotations

import asyncio
import csv
import json
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from moba_rl.actions import ACTION_NAMES
from moba_rl.agents.scripted import OPPONENTS, create_opponent
from moba_rl.environment import MOBAEnv
from moba_rl.simulation.lini_tities import Team


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = (ROOT / "models").resolve()
ORDER_TICKS = 50
TICK_SECONDS = 0.1
ORDER_SECONDS = ORDER_TICKS * TICK_SECONDS
MAX_DECISIONS = 500
REALTIME_TICKS_PER_DECISION = 5
REALTIME_INTERVAL_SECONDS = REALTIME_TICKS_PER_DECISION * TICK_SECONDS
REALTIME_MAX_DECISIONS = 10_000
REALTIME_MAX_IDENTICAL_ORDERS = 3
# Policies were trained to commit for ORDER_TICKS (five simulated seconds).
# Re-querying them on every 0.5-second realtime beat caused visible command jitter.
REALTIME_POLICY_DECISION_STEPS = max(1, ORDER_TICKS // REALTIME_TICKS_PER_DECISION)


def model_decision_spec(path: Path) -> tuple[int, int] | None:
    """Read the checkpoint action and observation dimensions."""
    try:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
        actions = int(data.get("action_space", {}).get("n", 0))
        shape = data.get("observation_space", {}).get("_shape", [])
        features = int(shape[0]) if shape else 0
        return (actions, features) if (actions, features) in {(24, 43), (36, 57)} else None
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None


def is_strategic_model(path: Path) -> bool:
    return model_decision_spec(path) is not None


@dataclass
class MatchSession:
    env: MOBAEnv
    observation: np.ndarray
    info: dict
    model: MaskablePPO | None = None
    lock: asyncio.Lock | None = None
    current_blue_action: int | None = None
    last_blue_policy_action: int | None = None
    blue_policy_action_repeats: int = 0
    last_policy_preferences: list[dict] | None = None
    policy_control: bool = False
    policy_one_step: bool = False
    decision_view: bool = True
    paused: bool = False
    realtime: bool = False

    def _policy_preferences(self) -> list[dict]:
        if self.model is None:
            return []
        try:
            obs_tensor, _ = self.model.policy.obs_to_tensor(self.observation)
            distribution = self.model.policy.get_distribution(
                obs_tensor,
                action_masks=self.env.action_masks(),
            )
            probabilities = distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
            legal = np.flatnonzero(self.env.action_masks())
            ranked = sorted(legal, key=lambda index: float(probabilities[index]), reverse=True)[:4]
            return [
                {"action": int(index), "action_name": ACTION_NAMES[int(index)],
                 "probability": float(probabilities[index])}
                for index in ranked
            ]
        except (AttributeError, RuntimeError, ValueError, IndexError):
            return self.last_policy_preferences or []

    def _visible_context(self, team: Team) -> list[str]:
        game = self.env.game
        champion = game.champions[team]
        lane = game.lane_for_team(team)
        enemy_team = game.enemy_team(team)
        enemy = game.champions[enemy_team]
        ally_wave = sum(m.alive and m.team == team and m.lane == lane for m in game.minions)
        enemy_wave = sum(m.alive and m.team == enemy_team and m.lane == lane for m in game.minions)
        context = [
            f"{lane.title()} lane · {champion.hp / max(1, champion.max_hp):.0%} health · level {champion.level}",
            f"{champion.gold} gold · {ally_wave} allied / {enemy_wave} enemy minions in lane",
        ]
        wave = game.wave_order_for(team, lane)
        context.append(f"{lane.title()} wave plan · {wave.replace('_', ' ')}")
        owned_wards = sum(ward.team == team for ward in game.wards)
        context.append(f"Vision · {owned_wards}/2 wards · ward cooldown {champion.ward_timer:.0f}s · sweep cooldown {champion.sweep_timer:.0f}s")
        context.append(f"Upgrades · blade {champion.attack_upgrade_level}/5 · armor {champion.armor_upgrade_level}/5 · boots {champion.speed_upgrade_level}/5 · vitality {champion.vitality_upgrade_level}/5")
        enemy_tower = game.tower_for_lane(enemy_team, lane)
        if not self.env.fog_of_war or self.env._position_visible(enemy_tower.position, team):
            context.append(f"{enemy_team.value.title()} {lane} tower · {enemy_tower.hp:.0f} HP")
        else:
            context.append(f"{enemy_team.value.title()} {lane} tower outside current vision")
        objective = game.objective
        if objective.alive:
            context.append(f"Neutral objective available · {objective.hp:.0f} HP")
        elif game.objective_respawn_timer is not None:
            context.append(f"Objective respawns in {game.objective_respawn_timer:.0f}s")
        if not self.env.fog_of_war or self.env._position_visible(enemy.position, team):
            context.append(f"Opponent visible · {enemy.hp / max(1, enemy.max_hp):.0%} health")
        return context

    def _public_state(self) -> dict:
        state = self.env.game.get_state()
        if self.env.fog_of_war:
            game = self.env.game
            enemy = game.champions[Team.RED]
            if not self.env._position_visible(enemy.position, Team.BLUE):
                state["champions"]["red"].update({
                    "x": None, "y": None, "hp": None, "gold": None,
                    "level": None, "visible": False,
                })
                state["minion_entities"] = [
                    item for item in state["minion_entities"]
                    if item["team"] == "blue" or self.env._position_visible(
                        type(enemy.position)(item["x"], item["y"]), Team.BLUE
                    )
                ]
            for lane, tower in game.lane_towers[Team.RED].items():
                if not self.env._position_visible(tower.position, Team.BLUE):
                    state["lane_towers"]["red"][lane]["hp"] = None
            state["wards"] = [
                ward for ward in state.get("wards", [])
                if ward["team"] == "blue" or self.env._position_visible(
                    type(game.objective.position)(ward["x"], ward["y"]), Team.BLUE
                )
            ]
            red_base = game.bases[Team.RED]
            if not self.env._position_visible(red_base.position, Team.BLUE):
                state["bases"]["red"]["hp"] = None
        state["champions"]["blue"]["visible"] = True
        state["champions"]["red"].setdefault("visible", True)
        return state

    def payload(self):
        def describe(team: Team, action: int | None):
            return {
                "team": team.value,
                "action": action,
                "action_name": ACTION_NAMES.get(action) if action is not None else None,
                "target": self.env.decision_target(action, team),
                "observed_context": self._visible_context(team),
            }

        preferences = self._policy_preferences()
        if preferences:
            self.last_policy_preferences = preferences
        return {
            "observation": self.observation.tolist(),
            "action_mask": self.env.action_masks().tolist(),
            "action_names": {str(index): name for index, name in ACTION_NAMES.items()
                             if index < self.env.action_space.n},
            "info": self.info,
            "state": self._public_state(),
            "steps": self.env.steps,
            "done": bool(self.env.game.game_over or self.env.steps >= self.env.max_steps),
            "paused": self.paused,
            "decision_view": self.decision_view,
            "order_seconds": REALTIME_INTERVAL_SECONDS if self.realtime else ORDER_SECONDS,
            "realtime": self.realtime,
            "advanced_decisions": self.env.advanced_decisions,
            "events": self.env.game.events[-12:],
            "decisions": {
                "blue": describe(Team.BLUE, self.current_blue_action),
                "red": describe(Team.RED, self.env.last_opponent_action),
            },
            "policy_preferences": preferences,
        }


class CreateSessionRequest(BaseModel):
    opponent: str = "balanced"
    seed: int | None = None
    model: str | None = None
    red_model: str | None = None
    procedural_maps: bool = False
    fog_of_war: bool = False
    decision_view: bool = True
    advanced_decisions: bool = False
    realtime: bool = False


class StepRequest(BaseModel):
    action: int | None = Field(default=None, ge=0)
    use_agent: bool = False
    deterministic: bool = True


class RedPolicyOpponent:
    """Run a trained policy from Red's mirrored observation/action view."""

    def __init__(self, model: MaskablePPO, *, deterministic: bool = True,
                 avoid_repeated_orders: bool = False):
        self.model = model
        self.deterministic = deterministic
        self.avoid_repeated_orders = avoid_repeated_orders
        self.last_policy_action: int | None = None
        self.policy_action_repeats = 0
        self.cached_action: int | None = None
        self.steps_until_decision = 0

    def act_for_env(self, env: MOBAEnv) -> int:
        masks = env.action_masks(Team.RED)
        if (self.steps_until_decision > 0 and self.cached_action is not None
                and masks[self.cached_action]):
            self.steps_until_decision -= 1
            action = self.cached_action
            if action == 1:
                action = 2
            elif action == 2:
                action = 1
            env._execute_opponent_action(action)
            return self.cached_action
        observation = env._get_observation(Team.RED)
        masks = masks.copy()
        if (self.avoid_repeated_orders
                and self.policy_action_repeats >= REALTIME_MAX_IDENTICAL_ORDERS
                and self.last_policy_action is not None):
            masks[self.last_policy_action] = False
        action, _ = self.model.predict(
            observation,
            action_masks=masks,
            deterministic=self.deterministic,
        )
        policy_action = int(action)
        if self.avoid_repeated_orders:
            if policy_action == self.last_policy_action:
                self.policy_action_repeats += 1
            else:
                self.policy_action_repeats = 1
            self.last_policy_action = policy_action
        self.cached_action = policy_action
        if self.avoid_repeated_orders:
            self.steps_until_decision = REALTIME_POLICY_DECISION_STEPS - 1
        action = policy_action
        if action == 1:
            action = 2
        elif action == 2:
            action = 1
        env._execute_opponent_action(action)
        return action


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, MatchSession] = {}

    @staticmethod
    def _model_path(name: str) -> Path:
        path = (MODEL_DIR / name).resolve()
        if path != MODEL_DIR and MODEL_DIR not in path.parents:
            raise HTTPException(status_code=400, detail="Model must be inside models/.")
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Model not found: {name}")
        return path

    def create(self, request: CreateSessionRequest) -> tuple[str, MatchSession]:
        opponent_name = request.opponent.strip().lower()
        if opponent_name not in OPPONENTS:
            raise HTTPException(status_code=422,
                detail=f"Unknown opponent. Choose from {sorted(OPPONENTS)}.")

        advanced = bool(request.advanced_decisions)
        policy_name = request.model or request.red_model
        if policy_name:
            spec = model_decision_spec(self._model_path(policy_name))
            if spec is None:
                raise HTTPException(status_code=422, detail="That checkpoint does not match a supported arena policy.")
            advanced = spec[0] == 36

        env = MOBAEnv(
            opponent=create_opponent(opponent_name),
            seed=request.seed,
            max_steps=REALTIME_MAX_DECISIONS if request.realtime else MAX_DECISIONS,
            ticks_per_decision=REALTIME_TICKS_PER_DECISION if request.realtime else ORDER_TICKS,
            simulation_dt=TICK_SECONDS,
            procedural_map=request.procedural_maps,
            fog_of_war=request.fog_of_war or bool(request.model) or bool(request.red_model),
            hierarchical_actions=True,
            advanced_decisions=advanced,
            record_trajectory=True,
        )
        observation, info = env.reset(seed=request.seed)
        model = None
        try:
            if request.model:
                model_path = self._model_path(request.model)
                model = MaskablePPO.load(str(model_path), device="cpu")
                if tuple(model.observation_space.shape) != tuple(env.observation_space.shape):
                    raise HTTPException(status_code=422,
                        detail=f"That policy expects {model.observation_space.shape}; this game provides {env.observation_space.shape}.")
                if model.action_space.n != env.action_space.n:
                    raise HTTPException(status_code=422,
                        detail=f"That policy expects {model.action_space.n} actions; this game provides {env.action_space.n}.")
            if request.red_model:
                red_path = self._model_path(request.red_model)
                red_model = MaskablePPO.load(str(red_path), device="cpu")
                if tuple(red_model.observation_space.shape) != tuple(env.observation_space.shape):
                    raise HTTPException(status_code=422,
                        detail=f"That opponent policy expects {red_model.observation_space.shape}; this game provides {env.observation_space.shape}.")
                if red_model.action_space.n != env.action_space.n:
                    raise HTTPException(status_code=422,
                        detail=f"That opponent policy expects {red_model.action_space.n} actions; this game provides {env.action_space.n}.")
                env.opponent = RedPolicyOpponent(
                    red_model,
                    deterministic=not request.realtime,
                    avoid_repeated_orders=request.realtime,
                )
        except Exception:
            env.close()
            raise

        session_id = uuid.uuid4().hex
        session = MatchSession(env=env, observation=observation, info=info,
            model=model, lock=asyncio.Lock(), decision_view=request.decision_view,
            realtime=request.realtime)
        self.sessions[session_id] = session
        return session_id, session

    def get(self, session_id: str) -> MatchSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Match session not found.")
        return session

    def step(self, session: MatchSession, request: StepRequest, *, validate_mask: bool = True):
        if session.env.game.game_over or session.env.steps >= session.env.max_steps:
            raise HTTPException(status_code=409, detail="This match has finished.")
        if request.use_agent:
            if session.model is None:
                raise HTTPException(status_code=409, detail="No policy model is loaded.")
            reuse_current_order = (
                session.realtime
                and session.current_blue_action is not None
                and session.blue_policy_action_repeats > 0
                and session.env.action_masks()[session.current_blue_action]
            )
            if reuse_current_order:
                action = int(session.current_blue_action)
                session.blue_policy_action_repeats -= 1
            else:
                masks = session.env.action_masks()
                action, _ = session.model.predict(
                    session.observation,
                    action_masks=masks,
                    deterministic=(request.deterministic and not session.realtime),
                )
                action = int(action)
                if session.realtime:
                    session.blue_policy_action_repeats = REALTIME_POLICY_DECISION_STEPS - 1
        else:
            if request.action is None:
                raise HTTPException(status_code=422, detail="Provide an action or set use_agent=true.")
            action = int(request.action)
        masks = session.env.action_masks()
        if action < 0 or action >= len(masks):
            raise HTTPException(status_code=422, detail="Action is outside the action space.")
        if validate_mask and not masks[action]:
            raise HTTPException(status_code=422, detail="Action is currently unavailable.")
        observation, reward, terminated, truncated, info = session.env.step(action)
        session.observation = observation
        session.info = info
        session.current_blue_action = action
        return {**session.payload(), "action": action, "reward": float(reward),
                "terminated": bool(terminated), "truncated": bool(truncated)}


manager = SessionManager()
app = FastAPI(title="MOBA-RL Strategic Arena API", version="0.2.0")

allowed_origins = [origin.strip() for origin in os.getenv(
    "MOBA_RL_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5176,http://127.0.0.1:5176",
).split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"])


@app.get("/health")
def health():
    return {"status": "ok", "game": "moba_strategic_orders", "sessions": len(manager.sessions)}


@app.get("/models")
def list_models():
    if not MODEL_DIR.exists():
        return {"models": [], "advanced_models": []}
    compatible = [path for path in MODEL_DIR.rglob("*.zip") if is_strategic_model(path)]
    return {
        "models": sorted(path.relative_to(MODEL_DIR).as_posix() for path in compatible),
        "advanced_models": sorted(path.relative_to(MODEL_DIR).as_posix() for path in compatible if model_decision_spec(path) == (36, 57)),
    }


@app.get("/config")
def config():
    return {
        "game": "three_lane_moba",
        "opponents": sorted(OPPONENTS),
        "actions": ACTION_NAMES,
        "profiles": {
            "basic": {"action_count": 12, "observation_features": 33},
            "strategic": {"action_count": 17, "observation_features": 39},
            "hierarchical": {"action_count": 24, "observation_features": 39},
            "advanced": {"action_count": 36, "observation_features": 53},
        },
        "fog_feature_count": 4,
        "strategic_actions": list(range(17, 24)),
        "advanced_actions": list(range(24, 36)),
        "simulation_tick_seconds": TICK_SECONDS,
        "order_seconds": ORDER_SECONDS,
        "realtime_order_seconds": REALTIME_INTERVAL_SECONDS,
        "realtime_ticks_per_order": REALTIME_TICKS_PER_DECISION,
        "lanes": ["top", "mid", "bottom"],
    }


@app.get("/experiments")
def list_experiments():
    """Return the latest usable evaluation checkpoint for each training run."""
    log_dir = ROOT / "logs"
    experiments = []
    for path in sorted(
        log_dir.glob("*_multi_opponent_evaluation.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ) if log_dir.exists() else []:
        try:
            with path.open("r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            valid = [row for row in rows if row.get("opponent") and row.get("timesteps")]
            if not valid:
                continue
            latest_step = max(int(row["timesteps"]) for row in valid)
            latest = [row for row in valid if int(row["timesteps"]) == latest_step]
            opponents = []
            for row in latest:
                try:
                    opponents.append({
                        "opponent": row["opponent"],
                        "win_rate": float(row["win_rate"]),
                        "episodes": int(row["episodes"]),
                        "wins": int(row["wins"]),
                        "losses": int(row["losses"]),
                        "draws": int(row["draws"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            if not opponents:
                continue
            overall = next((
                float(row["overall_win_rate"])
                for row in latest if row.get("overall_win_rate")
            ), sum(row["win_rate"] for row in opponents) / len(opponents))
            experiments.append({
                "name": path.stem.removesuffix("_multi_opponent_evaluation"),
                "timesteps": latest_step,
                "overall_win_rate": overall,
                "opponents": opponents,
                "updated_at": path.stat().st_mtime,
            })
        except (OSError, csv.Error, ValueError):
            continue
    return {"experiments": experiments[:12]}


@app.post("/sessions")
def create_session(request: CreateSessionRequest):
    session_id, session = manager.create(request)
    return {"session_id": session_id, **session.payload()}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    return manager.get(session_id).payload()


@app.get("/sessions/{session_id}/replay")
def download_replay(session_id: str):
    session = manager.get(session_id)
    game = session.env.game
    payload = {
        "schema_version": 1,
        "source": "arena_session",
        "episode_seed": session.info.get("game_seed"),
        "winner": game.winner.value if game.winner else None,
        "advanced_decisions": session.env.advanced_decisions,
        "trajectory": session.env.trajectory,
    }
    return JSONResponse(
        payload,
        headers={"Content-Disposition": f'attachment; filename="moba_replay_{session_id[:12]}.json"'},
    )


@app.post("/sessions/{session_id}/step")
async def step_session(session_id: str, request: StepRequest):
    session = manager.get(session_id)
    async with session.lock:
        return manager.step(session, request)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    session = manager.sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="Match session not found.")
    session.env.close()
    return {"deleted": True}


@app.websocket("/ws/sessions/{session_id}")
async def websocket_session(websocket: WebSocket, session_id: str):
    session = manager.sessions.get(session_id)
    if session is None:
        await websocket.close(code=4404, reason="Match session not found")
        return
    await websocket.accept()
    if session.realtime:
        session.paused = False
        session.policy_control = session.model is not None
        session.policy_one_step = False
        await websocket.send_json(session.payload())
        loop = asyncio.get_running_loop()
        next_tick = loop.time() + REALTIME_INTERVAL_SECONDS
        try:
            while True:
                message = None
                timeout = max(0.0, next_tick - loop.time())
                try:
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass
                if message is not None:
                    if message.get("pause"):
                        session.paused = True
                        await websocket.send_json(session.payload())
                        continue
                    if message.get("resume"):
                        session.paused = False
                        next_tick = loop.time() + REALTIME_INTERVAL_SECONDS
                        await websocket.send_json(session.payload())
                    if message.get("step_agent") or message.get("use_agent"):
                        if session.model is None:
                            await websocket.send_json({"error": "No policy model is loaded."})
                        else:
                            session.policy_control = True
                            session.paused = False
                    if message.get("action") is not None:
                        session.current_blue_action = int(message["action"])
                        session.policy_control = False
                        session.paused = False
                        await websocket.send_json(session.payload())
                if session.paused:
                    continue
                now = loop.time()
                if now < next_tick:
                    continue
                next_tick = now + REALTIME_INTERVAL_SECONDS
                if session.env.game.game_over or session.env.steps >= session.env.max_steps:
                    await websocket.send_json(session.payload())
                    break
                if manager.sessions.get(session_id) is not session:
                    break
                request = (StepRequest(use_agent=True, deterministic=False)
                    if session.policy_control
                    else StepRequest(action=session.current_blue_action or 0))
                async with session.lock:
                    result = manager.step(session, request, validate_mask=False)
                await websocket.send_json(result)
                if result["done"]:
                    break
        except WebSocketDisconnect:
            return
        except HTTPException as exc:
            await websocket.send_json({"error": exc.detail})
        return
    if session.model is not None:
        session.policy_control = True
        if session.decision_view:
            session.policy_one_step = True
    elif session.decision_view:
        session.paused = True
    await websocket.send_json(session.payload())
    loop = asyncio.get_running_loop()
    tick_interval = ORDER_SECONDS
    next_tick = loop.time() + tick_interval
    try:
        while True:
            message = None
            if session.paused:
                message = await websocket.receive_json()
            else:
                timeout = max(0.0, next_tick - loop.time())
                if timeout > 0.0:
                    try:
                        message = await asyncio.wait_for(websocket.receive_json(), timeout=timeout)
                    except asyncio.TimeoutError:
                        pass
            if message is not None:
                if message.get("pause"):
                    session.paused = True
                    session.policy_control = False
                    session.policy_one_step = False
                    await websocket.send_json(session.payload())
                    continue
                if message.get("step_agent"):
                    if session.model is None:
                        await websocket.send_json({"error": "No policy model is loaded."})
                    else:
                        session.paused = False
                        session.policy_control = True
                        session.policy_one_step = True
                        next_tick = loop.time() + tick_interval
                if "use_agent" in message:
                    wants_policy = bool(message["use_agent"])
                    if wants_policy and session.model is None:
                        await websocket.send_json({"error": "No policy model is loaded."})
                    else:
                        session.paused = False
                        session.policy_control = wants_policy
                        session.policy_one_step = bool(message.get("one_step", False))
                        next_tick = loop.time() + tick_interval
                if message.get("action") is not None:
                    session.current_blue_action = int(message["action"])
                    session.policy_control = False
                    session.paused = False
                    session.policy_one_step = session.decision_view
                    next_tick = loop.time() + tick_interval
            if session.paused:
                await websocket.send_json(session.payload())
                continue
            now = loop.time()
            if now < next_tick:
                continue
            next_tick += tick_interval
            if session.env.game.game_over or session.env.steps >= session.env.max_steps:
                await websocket.send_json(session.payload())
                break
            if manager.sessions.get(session_id) is not session:
                break
            request = StepRequest(use_agent=True, deterministic=True) if session.policy_control else StepRequest(action=session.current_blue_action or 0)
            async with session.lock:
                result = manager.step(session, request, validate_mask=False)
                if session.policy_one_step:
                    session.policy_control = False
                    session.policy_one_step = False
                    if session.decision_view:
                        session.paused = True
                        result["paused"] = True
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return
    except HTTPException as exc:
        await websocket.send_json({"error": exc.detail})
