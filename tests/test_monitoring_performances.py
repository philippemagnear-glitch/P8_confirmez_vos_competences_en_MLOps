"""Vérifie les fenêtres de performance sur un historique synthétique isolé.

Aucun dataset, modèle ou calcul Evidently n'est nécessaire à cette vue.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.monitoring.analysis import analyser_performances, mesures_performances


@pytest.fixture
def performance_database(tmp_path):
    database = tmp_path / "performances.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.execute("""CREATE TABLE api_calls (
                timestamp_utc TEXT, operation TEXT, status TEXT, model_version TEXT,
                latency_ms REAL, inference_ms REAL, cpu_time_ms REAL
            )""")
            for index in range(200):
                date = (pd.Timestamp("2026-11-21T09:00:00Z") + pd.Timedelta(seconds=index)).isoformat()
                status = "rejected" if index >= 190 else "success"
                connection.execute(
                    "INSERT INTO api_calls VALUES (?,?,?,?,?,?,?)",
                    (date, "predict", status, "4", 200 if index >= 100 else 100, 10, 20),
                )
            # Un appel schema ne doit pas déplacer les fenêtres predict.
            connection.execute("INSERT INTO api_calls VALUES ('2026-11-22','schema','success',NULL,NULL,NULL,NULL)")
    return database


def test_performance_windows_have_no_overlap_and_keep_all_calls(performance_database):
    report = analyser_performances(performance_database, 100)
    current, previous = report["actuel"], report["precedent"]
    assert report["comparaison"]
    assert current["Appels"] == previous["Appels"] == 100
    assert current["Mesures latency_ms"] == 90
    assert current["Latence médiane (ms)"] == 200
    assert previous["Latence médiane (ms)"] == 100
    assert current["Rejets (%)"] == 10
    assert previous["Rejets (%)"] == 0
    assert report["historique"]["Appels"].tolist() == [100, 100]
    assert report["historique"]["Latence médiane (ms)"].tolist() == [100, 200]


def test_missing_or_invalid_measurements_are_excluded_and_zero_is_kept():
    calls = pd.DataFrame({
        "status": ["success"] * 100, "model_version": ["4"] * 100,
        "timestamp_utc": ["2026-11-21T09:00:00Z"] * 100,
        "latency_ms": [0] * 50 + [None] * 50,
        "inference_ms": [10] * 49 + [float("inf"), -1] + [None] * 49,
        "cpu_time_ms": [None] * 100,
    })
    result = mesures_performances(calls)
    assert result["Mesures latency_ms"] == 50
    assert result["Latence médiane (ms)"] == result["Latence P95 (ms)"] == 0
    assert result["Mesures inference_ms"] == 49
    assert pd.isna(result["Inférence médiane (ms)"])
    assert pd.isna(result["Temps CPU médian (ms)"])


def test_partial_window_disables_comparison(performance_database):
    report = analyser_performances(performance_database, 150)
    assert not report["comparaison"]
    assert report["precedent"]["Appels"] == 50
    assert report["historique"]["Appels"].tolist() == [50, 150]


def test_empty_and_uninstrumented_database_remain_readable(performance_database):
    with closing(sqlite3.connect(performance_database)) as connection:
        with connection:
            for name in ["latency_ms", "inference_ms", "cpu_time_ms"]:
                connection.execute(f"ALTER TABLE api_calls DROP COLUMN {name}")
    report = analyser_performances(performance_database)
    assert report["actuel"]["Mesures latency_ms"] == 0
    assert pd.isna(report["actuel"]["Latence médiane (ms)"])
    with closing(sqlite3.connect(performance_database)) as connection:
        with connection:
            connection.execute("DELETE FROM api_calls")
    report = analyser_performances(performance_database)
    assert report["actuel"]["Appels"] == 0
    assert report["historique"].empty
    assert not report["comparaison"]


def test_performance_view_does_not_need_train_or_evidently(performance_database, monkeypatch):
    from app.monitoring import analysis as monitoring

    def forbidden(*args, **kwargs):
        pytest.fail("La vue performances ne doit pas calculer la référence ou la dérive.")

    monkeypatch.setattr(monitoring, "DB_PATH", performance_database)
    monkeypatch.setattr(monitoring, "charger_reference", forbidden)
    monkeypatch.setattr(monitoring, "analyser_appels", forbidden)
    dashboard = Path(__file__).resolve().parents[1] / "app" / "monitoring" / "dashboard.py"
    app = AppTest.from_file(str(dashboard)).run(timeout=30)
    next(button for button in app.button if button.label == "Actualiser les performances").click().run(timeout=30)
    assert not app.exception
    assert any(metric.label == "Latence médiane (ms)" and metric.value == "200.00" for metric in app.metric)
