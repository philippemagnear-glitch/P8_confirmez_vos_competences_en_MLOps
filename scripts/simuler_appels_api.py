"""Envoie un lot de clients du CSV processed à une API de scoring déjà démarrée.

Le script demande le schéma à /schema, choisit 100 lignes au hasard par défaut,
puis les transmet une par une à /predict avec une courte pause aléatoire.
Il conserve les types du modèle, les valeurs manquantes en null et les zéros.
L'API enregistre elle-même les appels dans sa base SQLite : ce script ne touche
ni à cette base ni au CSV. Il sert à constituer progressivement un historique,
pas à mesurer la charge maximale du serveur ni à créer une dérive artificielle.
--metrics-output exporte la latence complète et les résultats sans les entrées ;
request_id permet de rapprocher ces mesures du traitement enregistré par l'API.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, help="Fixer le tirage pour pouvoir le reproduire.")
    parser.add_argument(
        "--metrics-output", type=Path,
        help="Exporter les latences et identifiants des appels dans un CSV, sans leurs entrées.",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count doit être supérieur à zéro.")
    if not args.csv.is_file():
        parser.error(f"CSV introuvable : {args.csv}")

    client = None
    successes = errors = attempted = 0
    metrics_rows = []
    try:
        rng = random.Random(args.seed)
        client = Client(args.url, verbose=False)
        schema = client.submit(api_name="/schema").result(timeout=60)
        clients = select_clients(args.csv, schema, args.count, rng)
        print(f"Envoi de {len(clients)} clients à {args.url}, modèle {schema['model_version']}.")
        for index, payload in enumerate(clients, start=1):
            attempted += 1
            row = {
                "index": index,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "input_sha256": hashlib.sha256(json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
                ).encode()).hexdigest(),
                "status": "transport_error", "request_id": None,
            }
            started = time.perf_counter()
            try:
                result = client.submit(client=payload, api_name="/predict").result(timeout=60)
                row["latency_ms"] = (time.perf_counter() - started) * 1000
            except AppError as error:
                row["latency_ms"] = (time.perf_counter() - started) * 1000
                errors += 1
                row["status"] = "api_error"
                print(f"[{index}/{len(clients)}] Rejet ou erreur serveur : {error}")
            else:
                successes += 1
                row.update(status="success", request_id=result.get("request_id"),
                           score=result["score"], prediction=result["prediction"],
                           threshold=result["threshold"], model_version=result["model_version"])
                print(f"[{index}/{len(clients)}] Score={result['score']:.6f}, classe={result['prediction']}")
            finally:
                # La pause entre clients et le chargement CSV ne font pas partie de la latence.
                row.setdefault("latency_ms", (time.perf_counter() - started) * 1000)
                metrics_rows.append(row)
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
        if args.metrics_output is not None:
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(metrics_rows).to_csv(args.metrics_output, index=False)
            print(f"Mesures exportées : {args.metrics_output}")
        print(f"Bilan : {attempted} tentative(s), {successes} réussite(s), {errors} rejet(s) ou erreur(s) API.")
        if client is not None:
            client.close()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
