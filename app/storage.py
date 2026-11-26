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
import math
import os
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4


LOGGER = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "api_calls.sqlite3"
CREATE_API_CALLS_TABLE = """
    CREATE TABLE IF NOT EXISTS api_calls (
        request_id TEXT PRIMARY KEY, timestamp_utc TEXT NOT NULL,
        operation TEXT NOT NULL CHECK(operation IN ('predict', 'schema')),
        status TEXT NOT NULL CHECK(status IN ('success', 'rejected', 'error')),
        model_version TEXT, execution_device TEXT,
        input_json TEXT, score REAL, prediction INTEGER, threshold REAL, reason TEXT,
        cpu_time_ms REAL, preparation_ms REAL, inference_ms REAL,
        other_elapsed_ms REAL, latency_ms REAL
    )
"""


class ApiCallStore:
    """Écrit une ligne par opération, sans modifier sa réponse en cas d'échec."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path if path is not None else os.environ.get(
            "API_CALLS_DB_PATH", DEFAULT_DB_PATH
        ))

    @contextmanager
    def _connection(self):
        """Prépare le schéma sans perdre l'historique, dans une transaction courte."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5.0)) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(CREATE_API_CALLS_TABLE)
                existing = {row[1] for row in connection.execute("PRAGMA table_info(api_calls)")}
                for name, data_type in (
                    ("inference_ms", "REAL"), ("cpu_time_ms", "REAL"),
                    ("execution_device", "TEXT"),
                    ("latency_ms", "REAL"),
                    ("preparation_ms", "REAL"),
                    ("other_elapsed_ms", "REAL"),
                ):
                    if name not in existing:
                        connection.execute(f"ALTER TABLE api_calls ADD COLUMN {name} {data_type}")
                obsolete = {"cpu_percent", "duration_ms", "server_ms", "exchange_wait_ms",
                            "application_processing_ms", "outside_application_ms", "storage_ms"} & existing
                if obsolete:
                    connection.execute(
                        "UPDATE api_calls SET other_elapsed_ms=latency_ms-preparation_ms-inference_ms"
                    )
                    for name in sorted(obsolete):
                        connection.execute(f"ALTER TABLE api_calls DROP COLUMN {name}")
                yield connection

    def record_latency(self, *, request_id: str, latency_ms: float):
        """Complète un succès existant sans écraser une mesure déjà enregistrée."""
        if isinstance(latency_ms, bool) or not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("La latence doit être un nombre fini positif ou nul.")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT latency_ms FROM api_calls "
                "WHERE request_id=? AND operation='predict' AND status='success'",
                (request_id,),
            ).fetchone()
            if row is None:
                raise KeyError("Prédiction réussie introuvable.")
            if row[0] is not None:
                if row[0] == latency_ms:
                    return
                raise RuntimeError("Une mesure existe déjà pour cet appel.")
            connection.execute(
                "UPDATE api_calls SET latency_ms=?,other_elapsed_ms=?-preparation_ms-inference_ms WHERE request_id=?",
                (latency_ms, latency_ms, request_id),
            )

    def record_api_call(self, *, operation, status, started_at, inputs=None,
                        result=None, reason=None, request_id=None, measurements=None):
        """Enregistre l'appel en une transaction, sans mesurer la durée du stockage."""
        request_id = request_id or str(uuid4())
        measurements = measurements or {}
        try:
            # Seules les prédictions réussies conservent leurs entrées et sorties.
            prediction = operation == "predict" and status == "success"
            payload = json.dumps(inputs, allow_nan=False) if prediction else None
            output = result if prediction else {}
            with self._connection() as connection:
                connection.execute("""
                        INSERT INTO api_calls (
                            request_id, timestamp_utc, operation, status,
                            input_json, score, prediction, threshold, model_version, reason,
                            inference_ms, cpu_time_ms, execution_device, preparation_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        request_id, started_at, operation, status, payload,
                        output.get("score"), output.get("prediction"),
                        output.get("threshold"), output.get("model_version"), reason,
                        measurements.get("inference_ms"), measurements.get("cpu_time_ms"),
                        measurements.get("execution_device"), measurements.get("preparation_ms"),
                    ))
        except (OSError, sqlite3.Error, TypeError, ValueError) as error:
            LOGGER.error("Stockage SQLite impossible pour %s (%s).",
                         request_id, type(error).__name__)
