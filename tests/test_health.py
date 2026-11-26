"""Vérifie la santé HTTP sans CSV de démonstration ni serveur réseau."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import main


def test_health_responds_without_prediction(scorer, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DEMO_CSV", tmp_path / "absent.csv")
    with TestClient(main.create_app(scorer.model)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}
    scorer.session.run.assert_not_called()


def test_model_loading_failure_prevents_startup(monkeypatch):
    loader = Mock(side_effect=RuntimeError("Modèle indisponible"))
    monkeypatch.setattr(main, "CreditScoringModel", loader)
    with pytest.raises(RuntimeError, match="Modèle indisponible"):
        main.create_app()
    loader.assert_called_once_with()
