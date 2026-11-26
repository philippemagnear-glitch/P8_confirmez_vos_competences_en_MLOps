"""Envoie un lot de clients du CSV processed à une API de scoring déjà démarrée.

Le script demande le schéma à /schema, choisit 100 lignes au hasard par défaut,
puis les transmet une par une à /predict avec une courte pause aléatoire.
Il conserve les types du modèle, les valeurs manquantes en null et les zéros.
L'API enregistre les appels et reçoit la latence mesurée par ce script pour
compléter SQLite. Le CSV reste en lecture seule. Le script constitue un historique ;
il ne mesure pas la charge maximale et ne crée pas de dérive artificielle.
request_id relie la latence à sa prédiction.
Les prédictions et leur réception partagent une session HTTP et un contexte TLS
pendant le lot. La vérification des certificats reste activée.
Aucun fichier de mesures n'est exporté.
"""

import argparse
import json
import random
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from urllib.request import Request, urlopen
from unittest.mock import patch

import pandas as pd
from gradio_client import Client
from gradio_client import client as client_module
from gradio_client.exceptions import AppError


DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "data" / "processed" / "application_test_final_encoded.csv"
)
PANDAS_TYPES = {
    "integer": "Int32", "long": "Int64", "float": "Float64",
    "double": "Float64", "boolean": "boolean",
}


@contextmanager
def shared_gradio_client(url):
    """Réutilise une session HTTP et un contexte TLS pour le lot entier.

    Gradio recrée normalement des clients pour ses POST et ses flux SSE. Les
    adaptateurs ci-dessous ne concernent que cette session de ce script ; les
    fonctions originales sont restaurées à la sortie, sans modifier la bibliothèque.
    """
    httpx = client_module.httpx
    ssl_context = httpx.create_ssl_context()
    original_client, original_post = httpx.Client, httpx.post
    with original_client(verify=ssl_context) as shared:
        client = Client(url, verbose=False, ssl_verify=ssl_context)

        class StreamSession:
            def __init__(self, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, *exception):
                # Le flux ferme sa réponse, mais la session appartient au lot.
                return False

            def stream(self, *args, **kwargs):
                kwargs.setdefault("timeout", self.timeout)
                return shared.stream(*args, **kwargs)

        def stream_client(*args, **kwargs):
            if not args and kwargs.get("verify") is ssl_context and set(kwargs) <= {"verify", "timeout"}:
                return StreamSession(kwargs.get("timeout", 5))
            return original_client(*args, **kwargs)

        def post(request_url, *args, **kwargs):
            if str(request_url) == client.sse_data_url and kwargs.get("verify") is ssl_context:
                kwargs.pop("verify")
                return shared.post(request_url, *args, **kwargs)
            return original_post(request_url, *args, **kwargs)

        try:
            with patch.object(httpx, "Client", stream_client), patch.object(httpx, "post", post):
                yield client
        finally:
            client.close()


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
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count doit être supérieur à zéro.")
    if not args.csv.is_file():
        parser.error(f"CSV introuvable : {args.csv}")

    client = None
    successes = errors = attempted = 0
    resources = ExitStack()
    try:
        rng = random.Random(args.seed)
        client = resources.enter_context(shared_gradio_client(args.url))
        schema = client.submit(api_name="/schema").result(timeout=60)
        clients = select_clients(args.csv, schema, args.count, rng)
        print(f"Envoi de {len(clients)} clients à {args.url}, modèle {schema['model_version']}.")
        for index, payload in enumerate(clients, start=1):
            attempted += 1
            started = time.perf_counter()
            try:
                job = client.submit(client=payload, api_name="/predict")
                result = job.result(timeout=60)
                latency_ms = (time.perf_counter() - started) * 1000
            except AppError as error:
                errors += 1
                print(f"[{index}/{len(clients)}] Rejet ou erreur serveur : {error}")
            else:
                successes += 1
                post_json(args.url, "/measurements", {
                    "request_id": result["request_id"], "latency_ms": latency_ms,
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
        resources.close()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
