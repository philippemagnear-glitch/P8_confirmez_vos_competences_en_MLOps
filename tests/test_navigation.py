"""Vérifie les adresses des boutons sans lancer d'analyse de dérive."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import main


@pytest.mark.parametrize("url", ["http://localhost:8501", "https://dashboard.example.org"])
def test_gradio_dashboard_button_opens_configured_url_in_new_tab(
    scorer, tmp_path, monkeypatch, url,
):
    monkeypatch.setenv("MONITORING_URL", url)
    monkeypatch.setattr(main, "DEMO_CSV", tmp_path / "absent.csv")
    demo = main.create_demo(scorer.model)
    buttons = [item for item in demo.config["components"] if item["type"] == "button"]
    props = next(item["props"] for item in buttons if item["props"]["value"] == "Ouvrir le dashboard")
    assert props["link"] == url
    assert props["link_target"] == "_blank"


@pytest.mark.parametrize("url", ["http://localhost:7860", "https://scoring.example.org"])
def test_streamlit_prediction_button_uses_configured_url(monkeypatch, url):
    monkeypatch.setenv("SCORING_URL", url)
    dashboard = Path(__file__).resolve().parents[1] / "dashboard.py"
    app = AppTest.from_file(str(dashboard)).run(timeout=30)
    assert not app.exception
    button = app.get("link_button")[0]
    assert button.proto.label == "Ouvrir les prédictions"
    assert button.proto.url == url
