"""Contrôle les mesures sans modifier les scores ni perdre l'ancien historique."""

import math
import sqlite3
from contextlib import closing
from time import perf_counter
from uuid import UUID

import gradio as gr
import pytest

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
        assert math.isfinite(row["cpu_percent"]) and row["cpu_percent"] >= 0
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
                     started=perf_counter(), measurements={"cpu_time_ms": 1, "cpu_percent": 50})
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM api_calls").fetchone()[0] == 3
        assert connection.execute(
            "SELECT inference_ms,cpu_time_ms FROM api_calls WHERE request_id='historique'"
        ).fetchone() == (None, None)
