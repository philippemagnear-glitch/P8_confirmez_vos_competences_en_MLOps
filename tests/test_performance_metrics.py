"""Contrôle les mesures sans modifier les scores ni perdre l'ancien historique."""

import sqlite3
from contextlib import closing
from uuid import UUID, uuid4

import gradio as gr
import pytest
from gradio_client.exceptions import AppError

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
        assert row["model_backend"] == "onnx"
        if valid:
            assert row["request_id"] == result["request_id"]
            assert row["inference_ms"] >= 0
            assert row["preparation_ms"] >= 0
            assert row["score"] == result["score"] == 0.25
        else:
            assert row["status"] == "rejected"
            assert row["inference_ms"] is None
            assert row["score"] is None and row["input_json"] is None
            scorer.session.run.assert_not_called()
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
        store.record_api_call(operation="schema", status="success", started_at="2026-11-21",
                              measurements={"cpu_time_ms": 1})
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 3
        assert connection.execute(
            "SELECT inference_ms,cpu_time_ms FROM api_calls WHERE request_id='historique'"
        ).fetchone() == (None, None)


@pytest.fixture
def measured_api(scorer, serve_api, client_data):
    """Une prédiction enregistrée et un client Gradio sur une base isolée."""
    with serve_api(scorer.model) as server:
        database = server.db_path
        request_id = str(uuid4())
        ApiCallStore(database).record_api_call(
            operation="predict", status="success", started_at="2026-11-21T09:00:00+00:00",
            request_id=request_id, inputs=client_data,
            result=scorer.model.predict(client_data),
        )
        payload = {"request_id": request_id, "latency_ms": 123.5}
        yield server.client, database, payload


def test_client_latency_is_saved_once_without_changing_prediction(measured_api):
    api, database, payload = measured_api
    for _ in range(2):
        result = api.submit(measurement=payload, api_name="/measurements").result(timeout=20)
        assert result == {"status": "saved", "request_id": payload["request_id"]}
    with pytest.raises(AppError, match="Une mesure existe déjà"):
        api.submit(
            measurement={**payload, "latency_ms": 999.0}, api_name="/measurements",
        ).result(timeout=20)
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
    with pytest.raises(AppError, match="latency_ms|request_id"):
        api.submit(measurement={**payload, **change}, api_name="/measurements").result(timeout=20)
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT latency_ms FROM api_calls").fetchone() == (None,)


def test_measurement_requires_existing_prediction(measured_api):
    api, _, payload = measured_api
    with pytest.raises(AppError, match="Prédiction réussie introuvable"):
        api.submit(
            measurement={**payload, "request_id": str(uuid4())}, api_name="/measurements",
        ).result(timeout=20)


def test_measurement_storage_failure_is_explicit(measured_api, monkeypatch):
    api, _, payload = measured_api

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("indisponible")

    monkeypatch.setattr(ApiCallStore, "record_latency", fail)
    with pytest.raises(AppError, match="Stockage SQLite indisponible"):
        api.submit(measurement=payload, api_name="/measurements").result(timeout=20)
