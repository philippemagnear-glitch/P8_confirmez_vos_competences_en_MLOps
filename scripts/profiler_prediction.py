"""Profile le traitement réel sans Gradio, puis efface uniquement ses traces SQLite."""

import argparse
import cProfile
import random
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from app.prediction import CreditScoringModel
from app.scoring_service import process_api_call
from app.storage import ApiCallStore
from scripts.simuler_appels_api import DEFAULT_CSV, select_clients


REPORTS = Path(__file__).resolve().parents[1] / "tests" / "reports" / "profiling"


def run_predictions(model, store, clients, request_ids):
    """Utilise le même traitement que l'API, avec des identifiants de nettoyage."""
    for client in clients:
        request_id = str(uuid4())
        request_ids.append(request_id)
        process_api_call(model, store, "predict", client, request_id=request_id)


def remove_profiled_calls(store, request_ids):
    """Supprime par identifiant exact, sans toucher aux appels API concurrents."""
    if not request_ids or not store.path.is_file():
        return
    try:
        uri = store.path.resolve().as_uri() + "?mode=rw"
        with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
            with connection:
                connection.executemany(
                    "DELETE FROM api_calls WHERE request_id=?",
                    [(request_id,) for request_id in request_ids],
                )
    except sqlite3.Error as error:
        raise RuntimeError(
            "Nettoyage SQLite impossible ; lignes de profiling à supprimer : "
            + ", ".join(request_ids)
        ) from error
    print("Traces SQLite de cette exécution supprimées.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1, help="Nombre de clients distincts.")
    parser.add_argument("--seed", type=int, default=42, help="Tirage reproductible des clients.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count doit être supérieur à zéro.")
    model = CreditScoringModel()
    clients = select_clients(args.csv, model.schema(), args.count, random.Random(args.seed))
    store = ApiCallStore()
    REPORTS.mkdir(parents=True, exist_ok=True)
    profile_path = REPORTS / f"prediction_directe_{uuid4()}.prof"
    profiler = cProfile.Profile()
    request_ids = []
    try:
        try:
            # Chargement du modèle et du CSV exclus ; préparation et stockage inclus.
            profiler.runcall(run_predictions, model, store, clients, request_ids)
        finally:
            profiler.dump_stats(str(profile_path))
            print(f"Profil enregistré : {profile_path}")
        uri = store.path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
            print("Mesures sous cProfile (ms), à ne pas comparer aux latences API :")
            for request_id in request_ids:
                row = connection.execute(
                    "SELECT cpu_time_ms, preparation_ms, inference_ms, storage_ms "
                    "FROM api_calls WHERE request_id=?", (request_id,),
                ).fetchone()
                if row is None or any(value is None for value in row):
                    raise RuntimeError(f"Mesures SQLite absentes ou incomplètes : {request_id}")
                print(
                    f"CPU {row[0]:.2f} | préparation {row[1]:.2f} | "
                    f"inférence {row[2]:.2f} | stockage {row[3]:.2f}"
                )
    finally:
        # Hors capture, même si le calcul ou l'export échoue.
        remove_profiled_calls(store, request_ids)


if __name__ == "__main__":
    main()
