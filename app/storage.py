"""Enregistre l'historique des appels à /predict et /schema dans SQLite.

Chaque appel conserve un identifiant, sa date UTC, l'opération demandée,
son statut et sa durée. Une prédiction réussie conserve également les
données du client fictif et le résultat du modèle. Un rejet ou une panne
conserve uniquement son motif en complément, sans les données envoyées.

La variable d'environnement API_CALLS_DB_PATH indique où enregistrer
la base. Dans Docker, un volume permet de conserver cette base lorsque
le conteneur est supprimé puis remplacé.

Chaque écriture ouvre puis ferme sa propre connexion SQLite.
Les colonnes de mesure sont ajoutées sans effacer l'historique : les anciennes
lignes restent sans mesure d'inférence ou CPU. La latence complète est mesurée
par le programme appelant, qui la corrèle avec request_id.
Si l'enregistrement échoue, le serveur journalise le problème sans
modifier la réponse de l'API.
"""

import json
import logging
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from time import perf_counter
from uuid import uuid4


LOGGER = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "api_calls.sqlite3"


class ApiCallStore:
    """Écrit une ligne par opération, sans modifier sa réponse en cas d'échec."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path if path is not None else os.environ.get(
            "API_CALLS_DB_PATH", DEFAULT_DB_PATH
        ))

    def record(self, *, operation, status, started_at, started, inputs=None,
               result=None, reason=None, request_id=None, measurements=None):
        duration_ms = max(0.0, (perf_counter() - started) * 1000)
        request_id = request_id or str(uuid4())
        measurements = measurements or {}
        try:
            # Seules les prédictions réussies conservent leurs entrées et sorties.
            prediction = operation == "predict" and status == "success"
            payload = json.dumps(inputs, allow_nan=False) if prediction else None
            output = result if prediction else {}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=1.0)) as connection:
                with connection:
                    # Sérialise aussi la migration si deux premiers appels arrivent ensemble.
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute("""
                        CREATE TABLE IF NOT EXISTS api_calls (
                            request_id TEXT PRIMARY KEY,
                            timestamp_utc TEXT NOT NULL,
                            operation TEXT NOT NULL CHECK(operation IN ('predict', 'schema')),
                            status TEXT NOT NULL CHECK(status IN ('success', 'rejected', 'error')),
                            duration_ms REAL NOT NULL CHECK(duration_ms >= 0),
                            input_json TEXT,
                            score REAL,
                            prediction INTEGER,
                            threshold REAL,
                            model_version TEXT,
                            reason TEXT
                        )
                    """)
                    existing = {row[1] for row in connection.execute("PRAGMA table_info(api_calls)")}
                    for name, data_type in (
                        ("inference_ms", "REAL"), ("cpu_time_ms", "REAL"),
                        ("cpu_percent", "REAL"), ("execution_device", "TEXT"),
                    ):
                        if name not in existing:
                            connection.execute(f"ALTER TABLE api_calls ADD COLUMN {name} {data_type}")
                    connection.execute("""
                        INSERT INTO api_calls (
                            request_id, timestamp_utc, operation, status, duration_ms,
                            input_json, score, prediction, threshold, model_version, reason,
                            inference_ms, cpu_time_ms, cpu_percent, execution_device
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        request_id, started_at, operation, status, duration_ms, payload,
                        output.get("score"), output.get("prediction"),
                        output.get("threshold"), output.get("model_version"), reason,
                        measurements.get("inference_ms"), measurements.get("cpu_time_ms"),
                        measurements.get("cpu_percent"), measurements.get("execution_device"),
                    ))
        except (OSError, sqlite3.Error, TypeError, ValueError) as error:
            LOGGER.error("Stockage SQLite impossible pour %s (%s).",
                         request_id, type(error).__name__)
