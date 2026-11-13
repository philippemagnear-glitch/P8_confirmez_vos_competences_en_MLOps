"""Appels HTTP Gradio depuis un client tiers et tests des callbacks de l'interface."""

import json
import logging
from unittest.mock import Mock

import gradio as gr
import pandas as pd
import pytest
from gradio_client.exceptions import AppError

from app import main
from app.prediction import CreditScoringModel
from verifier_api_deployee import load_reference, verify_api, verify_model_artifact


def callback(demo, name):
    return next(block.fn for block in demo.fns.values() if block.fn.__name__ == name)


def test_T02_demo_csv_prepares_complete_json(scorer, client_data, tmp_path, monkeypatch):
    path = tmp_path / "sample.csv"
    rows = [{"SK_ID_CURR": str(100000 + index), **client_data} for index in range(51)]
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(main, "DEMO_CSV", path)
    clients = main.load_demo_clients(scorer.model)
    assert len(clients) == 50
    assert "100050" not in clients
    demo = main.create_demo(scorer.model)
    text, old_result = callback(demo, "select_client")("100000")
    assert old_result is None
    payload = json.loads(text)
    assert set(payload) == set(client_data)
    assert payload["OWN_CAR_AGE"] is None
    assert payload["CNT_CHILDREN"] == 0
    result = callback(demo, "predict_json")(text)
    assert result["score"] == 0.25
    with pytest.raises(gr.Error, match="Sélectionner un client"):
        callback(demo, "select_client")("inconnu")
    assert callback(demo, "<lambda>")() == ("", None)


@pytest.mark.parametrize("text", ['{"AMT_INCOME_TOTAL":', None])
def test_T03_ui_malformed_json(scorer, tmp_path, monkeypatch, text):
    monkeypatch.setattr(main, "DEMO_CSV", tmp_path / "absent.csv")
    demo = main.create_demo(scorer.model)
    with pytest.raises(gr.Error, match="JSON valide"):
        callback(demo, "predict_json")(text)
    scorer.pipeline.predict_proba.assert_not_called()


def test_T08_real_v4_shared_reference_through_predict(serve_api):
    _, reference = load_reference()
    verify_model_artifact(reference)
    model = CreditScoringModel()
    with serve_api(model) as server:
        # La même vérification HTTP sera exécutée sur l'image Docker en CI.
        verify_api(server.url, timeout=30)


def test_T09_missing_key_reaches_third_party(api, client_data):
    api_info = api.client.view_api(print_info=False, return_format="dict")
    description = api_info["named_endpoints"]["/predict"]["description"]
    assert "AppError" in description
    assert "La prédiction a échoué" in description
    assert "event_id" in description and "SSE" in description
    del client_data["OWN_CAR_AGE"]
    with pytest.raises(AppError, match="Variables obligatoires absentes.*OWN_CAR_AGE"):
        api.client.submit(client=client_data, api_name="/predict").result(timeout=20)
    api.scorer.pipeline.predict_proba.assert_not_called()


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError])
def test_T09_internal_error_is_generic_and_private(
    api, client_data, caplog, capsys, exception_type,
):
    secret = "DONNEES_CLIENT_CONFIDENTIELLES_180000"
    api.scorer.pipeline.failure = exception_type(secret + json.dumps(client_data))
    with caplog.at_level(logging.ERROR, logger="app.main"):
        with pytest.raises(AppError) as raised:
            api.client.submit(client=client_data, api_name="/predict").result(timeout=20)
    assert "La prédiction a échoué" in str(raised.value)
    assert secret not in str(raised.value)
    assert "AMT_INCOME_TOTAL" not in str(raised.value)
    assert "Échec de la prédiction" in caplog.text
    assert exception_type.__name__ in caplog.text
    assert secret not in caplog.text
    assert "AMT_INCOME_TOTAL" not in caplog.text
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert "AMT_INCOME_TOTAL" not in captured.out + captured.err


def test_T10_http_schema_and_valid_json_without_demo_csv(api, client_data, schema_specs):
    schema = api.client.submit(api_name="/schema").result(timeout=20)
    columns = schema["columns"]
    assert schema["model_name"] == "credit_scoring_lightgbm"
    assert schema["model_version"] == "4"
    assert schema["n_features"] == len(columns) == 310
    assert len({column["name"] for column in columns}) == 310
    assert [column["name"] for column in columns] == [spec["name"] for spec in schema_specs]
    assert [column["type"] for column in columns] == [spec["type"] for spec in schema_specs]
    assert all(column["required"] is True for column in columns)
    by_name = {column["name"]: column for column in columns}
    assert by_name["OWN_CAR_AGE"]["nullable"] is True
    assert by_name["OWN_CAR_AGE_ANOMALOUS"]["type"] == "boolean"
    assert by_name["OWN_CAR_AGE_ANOMALOUS"]["nullable"] is False
    result = api.client.submit(client=client_data, api_name="/predict").result(timeout=20)
    assert result == {
        "score": 0.25, "prediction": 0, "threshold": 0.5,
        "model_name": "credit_scoring_lightgbm", "model_version": "4",
    }
    frame = api.scorer.pipeline.predict_proba.call_args.args[0]
    assert pd.isna(frame.at[0, "OWN_CAR_AGE"])
    assert frame.at[0, "CNT_CHILDREN"] == 0
    api.scorer.pipeline.predict_proba.reset_mock()
    client_data["OWN_CAR_AGE_ANOMALOUS"] = None
    with pytest.raises(AppError, match="OWN_CAR_AGE_ANOMALOUS.*booléen"):
        api.client.submit(client=client_data, api_name="/predict").result(timeout=20)
    api.scorer.pipeline.predict_proba.assert_not_called()


def test_T10_csv_failure_does_not_block_api(scorer, tmp_path, monkeypatch, caplog):
    path = tmp_path / "invalid.csv"
    path.write_text("other\n1\n", encoding="utf-8")
    monkeypatch.setattr(main, "DEMO_CSV", path)
    with caplog.at_level(logging.WARNING, logger="app.main"):
        demo = main.create_demo(scorer.model)
    assert "Échantillon CSV indisponible" in caplog.text
    assert callback(demo, "get_schema")()["n_features"] == 310


def test_T10_default_startup_loads_model_once(scorer, tmp_path, monkeypatch, client_data):
    loader = Mock(return_value=scorer.model)
    monkeypatch.setattr(main, "CreditScoringModel", loader)
    monkeypatch.setattr(main, "DEMO_CSV", tmp_path / "absent.csv")
    demo = main.create_demo()
    predict = callback(demo, "predict")
    predict(client_data)
    predict(client_data)
    loader.assert_called_once()
    assert scorer.pipeline.predict_proba.call_count == 2
