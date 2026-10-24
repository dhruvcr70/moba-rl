from fastapi.testclient import TestClient

from moba_rl.api.app import app, manager


def test_api_creates_controls_and_closes_match_session():
    client = TestClient(app)
    created = client.post("/sessions", json={"opponent": "balanced", "seed": 9})
    assert created.status_code == 200
    session_id = created.json()["session_id"]
    assert len(created.json()["observation"]) == 33
    assert len(created.json()["action_mask"]) == 12

    stepped = client.post(
        f"/sessions/{session_id}/step",
        json={"action": 2},
    )
    assert stepped.status_code == 200
    assert stepped.json()["action"] == 2
    assert stepped.json()["steps"] == 1
    assert isinstance(stepped.json()["state"]["champions"]["blue"]["hp"], float)

    deleted = client.delete(f"/sessions/{session_id}")
    assert deleted.status_code == 200
    assert session_id not in manager.sessions


def test_api_rejects_unavailable_actions():
    client = TestClient(app)
    created = client.post("/sessions", json={"seed": 10})
    session_id = created.json()["session_id"]
    try:
        response = client.post(
            f"/sessions/{session_id}/step",
            json={"action": 11},
        )
        assert response.status_code == 422
    finally:
        client.delete(f"/sessions/{session_id}")


def test_api_websocket_streams_match_updates():
    client = TestClient(app)
    created = client.post("/sessions", json={"seed": 11})
    session_id = created.json()["session_id"]
    try:
        with client.websocket_connect(f"/ws/sessions/{session_id}") as websocket:
            initial = websocket.receive_json()
            assert initial["steps"] == 0
            websocket.send_json({"action": 2})
            updated = websocket.receive_json()
            assert updated["steps"] == 1
            assert updated["action"] == 2
    finally:
        client.delete(f"/sessions/{session_id}")


def test_api_exposes_advanced_profile_and_replay_download():
    client = TestClient(app)
    created = client.post(
        "/sessions",
        json={"opponent": "balanced", "seed": 13, "advanced_decisions": True},
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]
    try:
        payload = created.json()
        assert payload["advanced_decisions"] is True
        assert len(payload["observation"]) == 53
        assert len(payload["action_mask"]) == 36
        assert len(payload["action_names"]) == 36

        stepped = client.post(f"/sessions/{session_id}/step", json={"action": 27})
        assert stepped.status_code == 200
        assert stepped.json()["state"]["wave_orders"]["blue"]["mid"]["mode"] == "freeze"

        replay = client.get(f"/sessions/{session_id}/replay")
        assert replay.status_code == 200
        assert replay.headers["content-disposition"].startswith("attachment;")
        assert len(replay.json()["trajectory"]) == 2
    finally:
        client.delete(f"/sessions/{session_id}")


def test_api_experiment_list_has_stable_shape():
    response = TestClient(app).get("/experiments")
    assert response.status_code == 200
    assert isinstance(response.json()["experiments"], list)


def test_api_config_describes_each_supported_profile():
    response = TestClient(app).get("/config")
    assert response.status_code == 200
    profiles = response.json()["profiles"]
    assert profiles["basic"] == {"action_count": 12, "observation_features": 33}
    assert profiles["advanced"] == {"action_count": 36, "observation_features": 53}
