"""Données autonomes, pipeline simulé et serveur Gradio temporaire."""

import copy
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"

import pandas as pd
import pytest
from gradio_client import Client
from mlflow.models import Model, ModelSignature
from mlflow.types import ColSpec, Schema

from app import main
from app.prediction import CreditScoringModel


FIXTURES = Path(__file__).parent / "fixtures"
PROVENANCE = {
    "model_name": "credit_scoring_lightgbm",
    "model_version": "4",
    "model_id": "modele-simule",
    "run_id": "run-simule",
    "threshold": 0.5,
}


class FakePipeline:
    """Simule uniquement le calcul ; la vraie validation reste utilisée."""

    def __init__(self, names, score=0.25, classes=(0, 1)):
        self.feature_names_in_ = names
        self.classes_ = list(classes)
        self.score = score
        self.failure = None
        self.predict_proba = Mock(side_effect=self._predict_proba)

    def _predict_proba(self, frame):
        if self.failure is not None:
            raise self.failure
        row = [self.score if label == 1 else 1 - self.score for label in self.classes_]
        return pd.DataFrame([row]).to_numpy()


def build_scorer(path, specs, score=0.25, classes=(0, 1)):
    path.mkdir(parents=True, exist_ok=True)
    (path / "provenance.json").write_text(json.dumps(PROVENANCE), encoding="utf-8")
    metadata = Model(
        signature=ModelSignature(inputs=Schema([ColSpec(**spec) for spec in specs])),
        run_id=PROVENANCE["run_id"],
        model_id=PROVENANCE["model_id"],
    )
    pipeline = FakePipeline([spec["name"] for spec in specs], score, classes)
    loader = Mock(return_value=pipeline)
    # Aucun artefact réel n'est nécessaire aux tests simulés, même en CI.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("app.prediction.Model.load", Mock(return_value=metadata))
        patch.setattr("app.prediction.mlflow.sklearn.load_model", loader)
        model = CreditScoringModel(path)
    return SimpleNamespace(model=model, pipeline=pipeline, loader=loader)


@pytest.fixture(scope="session")
def schema_specs():
    return json.loads((FIXTURES / "schema_v4.json").read_text(encoding="utf-8"))


@pytest.fixture
def client_data():
    # Ce JSON synthétique n'est pas la référence numérique de T08.
    return json.loads((FIXTURES / "client_valid.json").read_text(encoding="utf-8"))


@pytest.fixture
def model_factory(tmp_path, schema_specs):
    def factory(score=0.25, classes=(0, 1), specs=None):
        return build_scorer(
            tmp_path / "model", copy.deepcopy(schema_specs if specs is None else specs),
            score, classes,
        )
    return factory


@pytest.fixture
def scorer(model_factory):
    return model_factory()


@pytest.fixture(scope="module")
def serve_api(tmp_path_factory):
    @contextmanager
    def serve(model):
        # Le programme appelant ne dépend jamais du CSV de démonstration.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(main, "DEMO_CSV", tmp_path_factory.mktemp("api") / "absent.csv")
            demo = main.create_demo(model)
        with socket.socket() as temporary_socket:
            temporary_socket.bind(("127.0.0.1", 0))
            port = temporary_socket.getsockname()[1]
        client = None
        try:
            _, url, _ = demo.queue().launch(
                server_name="127.0.0.1", server_port=port,
                prevent_thread_lock=True, share=False, quiet=True,
                run_history=False, footer_links=["api"],
            )
            client = Client(url, verbose=False, httpx_kwargs={"trust_env": False})
            yield SimpleNamespace(client=client, demo=demo)
        finally:
            if client is not None:
                client.close()
            demo.close(verbose=False)
    return serve


@pytest.fixture(scope="module")
def api_server(serve_api, tmp_path_factory, schema_specs):
    scorer = build_scorer(tmp_path_factory.mktemp("model"), schema_specs)
    with serve_api(scorer.model) as server:
        server.scorer = scorer
        yield server


@pytest.fixture
def api(api_server):
    pipeline = api_server.scorer.pipeline
    pipeline.predict_proba.reset_mock()
    pipeline.failure = None
    pipeline.score = 0.25
    yield api_server
    pipeline.failure = None


def pytest_terminal_summary(terminalreporter):
    """Affiche le nombre de scénarios réussis, échoués ou ignorés par ID."""
    terminalreporter.section("Résultats par ID du plan")
    for number in range(1, 11):
        test_id = f"T{number:02}"
        counts = {}
        for outcome in ("passed", "failed", "error", "skipped"):
            reports = terminalreporter.stats.get(outcome, [])
            counts[outcome] = sum(test_id in report.nodeid for report in reports)
        terminalreporter.write_line(
            f"{test_id}: {counts['passed']} réussi(s), {counts['failed']} échec(s), "
            f"{counts['error']} erreur(s), {counts['skipped']} ignoré(s)"
        )

