"""Vérifie le contrôle de santé public avec le protocole Gradio."""

from unittest.mock import Mock

import pytest

from app import main


def test_health_responds_without_prediction(scorer, serve_api):
    with serve_api(scorer.model) as server:
        result = server.client.submit(api_name="/health").result(timeout=20)
        endpoints = server.client.view_api(print_info=False, return_format="dict")["named_endpoints"]
        assert {"/predict", "/schema", "/health", "/measurements"} <= endpoints.keys()
        for name in ("health", "save_latency"):
            function = next(block for block in server.demo.fns.values() if block.fn.__name__ == name)
            assert function.queue is False
    assert result == {"status": "ok", "model_loaded": True}
    scorer.session.run.assert_not_called()


def test_model_loading_failure_prevents_startup(monkeypatch):
    loader = Mock(side_effect=RuntimeError("Modèle indisponible"))
    monkeypatch.setattr(main, "CreditScoringModel", loader)
    with pytest.raises(RuntimeError, match="Modèle indisponible"):
        main.create_demo()
    loader.assert_called_once_with()
