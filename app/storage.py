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

import hashlib
import json
import logging
import math
import os
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
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

    @contextmanager
    def _connection(self):
        """Prépare le schéma sans perdre l'historique, dans une transaction courte."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5.0)) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS api_calls (
                        request_id TEXT PRIMARY KEY, timestamp_utc TEXT NOT NULL,
                        operation TEXT NOT NULL CHECK(operation IN ('predict', 'schema')),
                        status TEXT NOT NULL CHECK(status IN ('success', 'rejected', 'error')),
                        duration_ms REAL NOT NULL CHECK(duration_ms >= 0),
                        input_json TEXT, score REAL, prediction INTEGER, threshold REAL,
                        model_version TEXT, reason TEXT
                    )
                """)
                existing = {row[1] for row in connection.execute("PRAGMA table_info(api_calls)")}
                for name, data_type in (
                    ("inference_ms", "REAL"), ("cpu_time_ms", "REAL"),
                    ("cpu_percent", "REAL"), ("execution_device", "TEXT"),
                    ("latency_ms", "REAL"), ("run_id", "TEXT"),
                    ("input_sha256", "TEXT"), ("client_timestamp_utc", "TEXT"),
                    ("client_sequence", "INTEGER"),
                ):
                    if name not in existing:
                        connection.execute(f"ALTER TABLE api_calls ADD COLUMN {name} {data_type}")
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS measurement_runs (
                        run_id TEXT PRIMARY KEY, name TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL, metadata_json TEXT NOT NULL
                    )
                """)
                yield connection

    def create_run(self, name: str, metadata: dict) -> str:
        """Identifie un lot ; sa configuration est conservée dans la même base."""
        run_id = str(uuid4())
        payload = json.dumps(metadata, allow_nan=False)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO measurement_runs VALUES (?, ?, ?, ?)",
                (run_id, name, datetime.now(timezone.utc).isoformat(), payload),
            )
        return run_id

    def record_latency(self, *, request_id: str, run_id: str, latency_ms: float,
                       client_timestamp_utc: str, client_sequence: int):
        """Complète un succès existant ; un second envoi identique est sans effet."""
        if isinstance(latency_ms, bool) or not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("La latence doit être un nombre fini positif ou nul.")
        with self._connection() as connection:
            if connection.execute(
                "SELECT 1 FROM measurement_runs WHERE run_id=?", (run_id,)
            ).fetchone() is None:
                raise KeyError("Lot inconnu.")
            row = connection.execute(
                "SELECT input_json,latency_ms,run_id,client_timestamp_utc,client_sequence "
                "FROM api_calls WHERE request_id=? AND operation='predict' AND status='success'",
                (request_id,),
            ).fetchone()
            if row is None:
                raise KeyError("Prédiction réussie introuvable.")
            values = (latency_ms, run_id, client_timestamp_utc, client_sequence)
            if row[1] is not None:
                if tuple(row[1:]) == values:
                    return
                raise RuntimeError("Une mesure existe déjà pour cet appel.")
            fingerprint = hashlib.sha256(json.dumps(
                json.loads(row[0]), sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode()).hexdigest()
            connection.execute(
                "UPDATE api_calls SET latency_ms=?,run_id=?,client_timestamp_utc=?,"
                "client_sequence=?,input_sha256=? WHERE request_id=?",
                (*values, fingerprint, request_id),
            )

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
            with self._connection() as connection:
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
