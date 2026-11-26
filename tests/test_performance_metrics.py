"""Contrôle les mesures sans modifier les scores ni perdre l'ancien historique."""

import sqlite3
from contextlib import closing
from time import perf_counter
from uuid import UUID, uuid4

import gradio as gr
import pytest
from fastapi.testclient import TestClient

from app import main
from app.storage import ApiCallStore


def test_inference_measurement_preserves_prediction(scorer, client_data):
    expected = scorer.model.predict(client_data)
    measurements = {}
    actual = scorer.model.predict(client_data, measurements=measurements)
    assert actual == expected
    assert measurements["inference_ms"] >= 0


@pytest.mark.parametrize("valid", [True, False])
def test_api_measurements_are_correlated_and_validation_precedes_inference(
    scorer, client_data, tmp_path, monkeypatch, valid,
):
    database = tmp_path / "calls.sqlite3"
    monkeypatch.setenv("API_CALLS_DB_PATH", str(database))
    monkeypatch.setattr(main, "DEMO_CSV", tmp_path / "absent.csv")
    demo = main.create_demo(scorer.model)
    predict = next(block.fn for block in demo.fns.values() if block.fn.__name__ == "predict")
    try:
        if valid:
            result = predict(client_data)
        else:
            del client_data["OWN_CAR_AGE"]
            with pytest.raises(gr.Error, match="OWN_CAR_AGE"):
                predict(client_data)
        with closing(sqlite3.connect(database)) as connection:
            connection.row_factory = sqlite3.Row
            row = dict(connection.execute("SELECT * FROM api_calls").fetchone())
        assert str(UUID(row["request_id"])) == row["request_id"]
        assert row["cpu_time_ms"] >= 0
        assert row["execution_device"] == "cpu"
        if valid:
            assert row["request_id"] == result["request_id"]
            assert 0 <= row["inference_ms"] <= row["duration_ms"]
            assert row["score"] == result["score"] == 0.25
        else:
            assert row["status"] == "rejected"
            assert row["inference_ms"] is None
            assert row["score"] is None and row["input_json"] is None
            scorer.pipeline.predict_proba.assert_not_called()
    finally:
        demo.close(verbose=False)


def test_old_database_migration_preserves_rows(tmp_path):
    database = tmp_path / "old.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE api_calls (
                    request_id TEXT PRIMARY KEY, timestamp_utc TEXT, operation TEXT,
                    status TEXT, duration_ms REAL, input_json TEXT, score REAL,
                    prediction INTEGER, threshold REAL, model_version TEXT, reason TEXT
                )
            """)
            connection.execute("""
                INSERT INTO api_calls VALUES (
                    'historique', '2026-11-01', 'schema', 'success', 1,
                    NULL, NULL, NULL, NULL, NULL, NULL
                )
            """)
    store = ApiCallStore(database)
    for _ in range(2):
        store.record(operation="schema", status="success", started_at="2026-11-21",
                     started=perf_counter(), measurements={"cpu_time_ms": 1})
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 3
        assert connection.execute(
            "SELECT inference_ms,cpu_time_ms FROM api_calls WHERE request_id='historique'"
        ).fetchone() == (None, None)


@pytest.fixture
def measured_api(scorer, tmp_path, monkeypatch, client_data):
    """Une prédiction déjà enregistrée, puis une API HTTP sur une base isolée."""
    database = tmp_path / "measurements.sqlite3"
    monkeypatch.setenv("API_CALLS_DB_PATH", str(database))
    monkeypatch.setattr(main, "DEMO_CSV", tmp_path / "absent.csv")
    request_id = str(uuid4())
    ApiCallStore(database).record(
        operation="predict", status="success", started_at="2026-11-21T09:00:00+00:00",
        started=perf_counter(), request_id=request_id, inputs=client_data,
        result=scorer.model.predict(client_data),
    )
    with TestClient(main.create_app(scorer.model)) as api:
        payload = {"request_id": request_id, "latency_ms": 123.5}
        yield api, database, payload


def test_client_latency_is_saved_once_without_changing_prediction(measured_api):
    api, database, payload = measured_api
    for _ in range(2):
        assert api.post("/measurements", json=payload).status_code == 200
    assert api.post("/measurements", json={**payload, "latency_ms": 999.0}).status_code == 409
    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            "SELECT latency_ms,score "
            "FROM api_calls"
        ).fetchone()
        assert row == (123.5, 0.25)
        assert connection.execute("SELECT count(*) FROM api_calls").fetchone()[0] == 1



@pytest.mark.parametrize("change", [
    {"latency_ms": -1.0}, {"latency_ms": True}, {"latency_ms": "123"},
    {"request_id": "invalide"},
])
def test_invalid_client_measurement_is_rejected(measured_api, change):
    api, database, payload = measured_api
    assert api.post("/measurements", json={**payload, **change}).status_code == 422
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT latency_ms FROM api_calls").fetchone() == (None,)


def test_measurement_requires_existing_prediction(measured_api):
    api, _, payload = measured_api
    assert api.post("/measurements", json={**payload, "request_id": str(uuid4())}).status_code == 404


def test_measurement_storage_failure_is_explicit(measured_api, monkeypatch):
    api, _, payload = measured_api

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("indisponible")

    monkeypatch.setattr(ApiCallStore, "record_latency", fail)
    assert api.post("/measurements", json=payload).status_code == 503
