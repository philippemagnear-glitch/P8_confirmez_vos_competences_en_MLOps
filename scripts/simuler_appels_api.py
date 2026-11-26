"""Envoie un lot de clients du CSV processed à une API de scoring déjà démarrée.

Le script demande le schéma à /schema, choisit 100 lignes au hasard par défaut,
puis les transmet une par une à /predict avec une courte pause aléatoire.
Il conserve les types du modèle, les valeurs manquantes en null et les zéros.
L'API enregistre les appels et reçoit la latence mesurée par ce script pour
compléter SQLite. Le CSV reste en lecture seule. Le script constitue un historique ;
il ne mesure pas la charge maximale et ne crée pas de dérive artificielle.
--run-name nomme le lot dans SQLite ; request_id relie la latence à sa prédiction.
Aucun fichier de mesures n'est exporté.
"""

import argparse
import platform
import json
import random
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import pandas as pd
from gradio_client import Client
from gradio_client.exceptions import AppError


DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "data" / "processed" / "application_test_final_encoded.csv"
)
PANDAS_TYPES = {
    "integer": "Int32", "long": "Int64", "float": "Float64",
    "double": "Float64", "boolean": "boolean",
}


def select_clients(csv_path, schema, count, rng):
    """Sélectionne des lignes distinctes et produit des objets JSON typés."""
    columns = schema["columns"]
    names = [column["name"] for column in columns]
    dtypes = {column["name"]: PANDAS_TYPES[column["type"]] for column in columns}
    frame = pd.read_csv(csv_path, usecols=names, dtype=dtypes)
    if count > len(frame):
        raise ValueError(f"Le CSV contient {len(frame)} clients, mais {count} sont demandés.")
    selected = frame.iloc[rng.sample(range(len(frame)), count)][names]
    clients = []
    for row in selected.itertuples(index=False, name=None):
        client = {}
        for column, value in zip(columns, row, strict=True):
            if pd.isna(value):
                value = None
            elif column["type"] == "boolean":
                value = bool(value)
            elif column["type"] in {"integer", "long"}:
                value = int(value)
            else:
                value = float(value)
            client[column["name"]] = value
        clients.append(client)
    return clients


def post_json(url, path, payload):
    """Transmet les mesures à l'API ; une erreur arrête le lot sans rejouer le score."""
    request = Request(
        url.rstrip('/') + path, data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, help="Fixer le tirage pour pouvoir le reproduire.")
    parser.add_argument("--run-name", default="simulation", help="Nom du lot dans SQLite.")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count doit être supérieur à zéro.")
    if not args.csv.is_file():
        parser.error(f"CSV introuvable : {args.csv}")

    client = None
    successes = errors = attempted = 0
    try:
        rng = random.Random(args.seed)
        client = Client(args.url, verbose=False)
        schema = client.submit(api_name="/schema").result(timeout=60)
        clients = select_clients(args.csv, schema, args.count, rng)
        run = post_json(args.url, "/measurement-runs", {
            "name": args.run_name,
            "metadata": {"count": args.count, "seed": args.seed,
                         "source_csv": args.csv.name, "client_python": platform.python_version(),
                         "client_system": platform.platform()},
        })
        print(f"Lot SQLite : {run['run_id']}")
        print(f"Envoi de {len(clients)} clients à {args.url}, modèle {schema['model_version']}.")
        for index, payload in enumerate(clients, start=1):
            attempted += 1
            timestamp = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            try:
                result = client.submit(client=payload, api_name="/predict").result(timeout=60)
                latency_ms = (time.perf_counter() - started) * 1000
            except AppError as error:
                errors += 1
                print(f"[{index}/{len(clients)}] Rejet ou erreur serveur : {error}")
            else:
                successes += 1
                post_json(args.url, "/measurements", {
                    "request_id": result["request_id"], "run_id": run["run_id"],
                    "latency_ms": latency_ms, "client_timestamp_utc": timestamp,
                    "client_sequence": index,
                })
                print(f"[{index}/{len(clients)}] Score={result['score']:.6f}, classe={result['prediction']}")
            if index < len(clients):
                time.sleep(rng.uniform(0.2, 1.0))
    except KeyboardInterrupt:
        print("\nSimulation interrompue. Les appels déjà enregistrés sont conservés.")
        return 130
    except Exception as error:
        # Aucun nouvel essai automatique : un appel sans réponse peut déjà être enregistré.
        print(f"Simulation arrêtée : {error}. Aucun nouvel essai automatique.", file=sys.stderr)
        return 1
    finally:
        print(f"Bilan : {attempted} tentative(s), {successes} réussite(s), {errors} rejet(s) ou erreur(s) API.")
        if client is not None:
            client.close()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
