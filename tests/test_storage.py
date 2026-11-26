"""T11–T13 : contrôle des traces SQLite et de leur persistance dans Docker.

Les appels HTTP utilisent un client synthétique complet. T13 nécessite une image
déjà construite, indiquée par SCORING_DOCKER_IMAGE ; sinon pytest signale qu'il
est ignoré. Le job deploy exécute obligatoirement ce contrôle sur son image.
"""

import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from uuid import UUID

import pytest
from gradio_client.exceptions import AppError

from app import main
from verifier_persistance_docker import verify_persistence


def read_last(path):
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        return dict(connection.execute(
            "SELECT * FROM api_calls ORDER BY rowid DESC LIMIT 1"
        ).fetchone())


def test_T11_successful_prediction_is_complete(serve_api, scorer, client_data):
    with serve_api(scorer.model) as server:
        result = server.client.submit(client=client_data, api_name="/predict").result(timeout=20)
        row = read_last(server.db_path)
    assert str(UUID(row["request_id"])) == row["request_id"]
    assert datetime.fromisoformat(row["timestamp_utc"]).utcoffset() == timedelta(0)
    assert row["operation"] == "predict" and row["status"] == "success"
    assert json.loads(row["input_json"]) == client_data
    assert json.loads(row["input_json"])["OWN_CAR_AGE"] is None
    assert json.loads(row["input_json"])["CNT_CHILDREN"] == 0
    for field in ("score", "prediction", "threshold", "model_version"):
        assert row[field] == result[field]
    assert row["reason"] is None


def test_T11_schema_is_recorded(serve_api, scorer):
    with serve_api(scorer.model) as server:
        server.client.submit(api_name="/schema").result(timeout=20)
        row = read_last(server.db_path)
    assert row["operation"] == "schema" and row["status"] == "success"
    assert row["input_json"] is None and row["score"] is None


def test_T12_rejected_input_has_no_payload_or_score(serve_api, scorer, client_data):
    del client_data["OWN_CAR_AGE"]
    with serve_api(scorer.model) as server:
        with pytest.raises(AppError, match="OWN_CAR_AGE"):
            server.client.submit(client=client_data, api_name="/predict").result(timeout=20)
        row = read_last(server.db_path)
    assert row["status"] == "rejected" and row["operation"] == "predict"
    assert "OWN_CAR_AGE" in row["reason"]
    for field in ("input_json", "score", "prediction", "threshold", "model_version"):
        assert row[field] is None
    scorer.session.run.assert_not_called()


def test_T12_server_failure_never_stores_exception_payload(serve_api, scorer, client_data):
    secret = "DONNEES_PRIVEES_DANS_EXCEPTION"
    scorer.session.failure = RuntimeError(secret + json.dumps(client_data))
    with serve_api(scorer.model) as server:
        with pytest.raises(AppError, match="La prédiction a échoué"):
            server.client.submit(client=client_data, api_name="/predict").result(timeout=20)
        row = read_last(server.db_path)
    assert row["status"] == "error"
    assert "RuntimeError" in row["reason"] and secret not in row["reason"]
    assert row["input_json"] is None and row["score"] is None


@pytest.mark.parametrize("operation", ["predict", "schema"])
def test_T12_storage_failure_preserves_success(
    scorer, client_data, tmp_path, monkeypatch, caplog, operation,
):
    # Un dossier à la place du fichier est une erreur portable Windows/Linux.
    monkeypatch.setenv("API_CALLS_DB_PATH", str(tmp_path))
    monkeypatch.setattr(main, "DEMO_CSV", tmp_path / "absent.csv")
    demo = main.create_demo(scorer.model)
    try:
        functions = {block.fn.__name__: block.fn for block in demo.fns.values()}
        with caplog.at_level(logging.ERROR, logger="app.storage"):
            if operation == "predict":
                result = functions["predict"](client_data)
                assert result["score"] == 0.25
            else:
                assert functions["get_schema"]()["n_features"] == 310
        assert "Stockage SQLite impossible" in caplog.text
        assert "AMT_INCOME_TOTAL" not in caplog.text
    finally:
        demo.close(verbose=False)


@pytest.mark.docker
def test_T13_history_survives_container_replacement():
    image = os.environ.get("SCORING_DOCKER_IMAGE")
    if not image:
        pytest.skip("T13 non exécuté : définir SCORING_DOCKER_IMAGE avec une image construite.")
    result = verify_persistence(image, os.environ.get("DOCKER_BINARY", "docker"))
    assert result["rows_before"] == result["rows_after_replacement"] == 2
    print(json.dumps(result))
