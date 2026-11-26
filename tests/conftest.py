"""Données de référence, session ONNX simulée et serveur Gradio temporaire."""

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

import numpy as np
import pytest
from gradio_client import Client
from mlflow.models import Model, ModelSignature
from mlflow.types import ColSpec, Schema

from app import main
from app.prediction import ARTIFACTS_PATH, CreditScoringModel


FIXTURES = Path(__file__).parent / "fixtures"
PROVENANCE = {
    "model_name": "credit_scoring_lightgbm",
    "model_version": "4",
    "model_id": "modele-simule",
    "run_id": "run-simule",
    "threshold": 0.5,
}


class FakeSession:
    """Simule uniquement ONNX Runtime ; la vraie validation reste utilisée."""

    def __init__(self, score=0.25, classes=(0, 1)):
        self.classes_ = list(classes)
        self.score = score
        self.failure = None
        self.run = Mock(side_effect=self._run)

    def _run(self, output_names, inputs):
        if self.failure is not None:
            raise self.failure
        row = [self.score if label == 1 else 1 - self.score for label in self.classes_]
        return [None, np.asarray([row])]


def build_scorer(path, specs, score=0.25, classes=(0, 1)):
    path.mkdir(parents=True, exist_ok=True)
    (path / "provenance.json").write_text(json.dumps(PROVENANCE), encoding="utf-8")
    metadata = Model(
        signature=ModelSignature(inputs=Schema([ColSpec(**spec) for spec in specs])),
        run_id=PROVENANCE["run_id"],
        model_id=PROVENANCE["model_id"],
    )
    pipeline = SimpleNamespace(
        feature_names_in_=[spec["name"] for spec in specs], classes_=list(classes),
    )
    session = FakeSession(score, classes)
    loader = Mock(return_value=pipeline)
    session_loader = Mock(return_value=session)
    # Signature réelle, métadonnées simulées et session ONNX remplacée pour les tests unitaires.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("app.prediction.Model.load", Mock(return_value=metadata))
        patch.setattr("app.prediction.mlflow.sklearn.load_model", loader)
        patch.setattr(CreditScoringModel, "create_onnx_session", session_loader)
        model = CreditScoringModel(path)
    return SimpleNamespace(
        model=model, session=session, loader=loader, session_loader=session_loader,
    )


@pytest.fixture(scope="session")
def schema_specs():
    """Lit la signature réelle une fois, sans désérialiser model.pkl."""
    metadata = Model.load(str(ARTIFACTS_PATH / "MLmodel"))
    if metadata.signature is None or metadata.signature.inputs is None:
        pytest.fail("L'artefact sélectionné doit contenir une signature d'entrée MLflow.")
    return [
        {"name": column.name, "type": column.type.name, "required": column.required}
        for column in metadata.signature.inputs.inputs
    ]


@pytest.fixture
def client_data():
    # Même ligne synthétique pour les tests unitaires, T08 et le conteneur.
    return json.loads((FIXTURES / "client_valid.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def isolate_call_storage(tmp_path, monkeypatch):
    """Les tests n'écrivent jamais dans l'historique utilisé manuellement."""
    monkeypatch.setenv("API_CALLS_DB_PATH", str(tmp_path / "api_calls.sqlite3"))


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
            db_path = tmp_path_factory.mktemp("storage") / "api_calls.sqlite3"
            patch.setenv("API_CALLS_DB_PATH", str(db_path))
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
            yield SimpleNamespace(client=client, demo=demo, url=url, db_path=db_path)
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
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
    session = api_server.scorer.session
    session.run.reset_mock()
    session.failure = None
    session.score = 0.25
    yield api_server
    session.failure = None


def pytest_terminal_summary(terminalreporter):
    """Affiche le nombre de scénarios réussis, échoués ou ignorés par ID."""
    terminalreporter.section("Résultats par ID du plan")
    for number in range(1, 14):
        test_id = f"T{number:02}"
        counts = {}
        for outcome in ("passed", "failed", "error", "skipped"):
            reports = terminalreporter.stats.get(outcome, [])
            counts[outcome] = sum(test_id in report.nodeid for report in reports)
        terminalreporter.write_line(
            f"{test_id}: {counts['passed']} réussi(s), {counts['failed']} échec(s), "
            f"{counts['error']} erreur(s), {counts['skipped']} ignoré(s)"
        )

