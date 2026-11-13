"""Vérifie T13 : remplacer un conteneur doit conserver son historique SQLite.

Ce contrôle utilise le vrai modèle dans une image déjà construite. Il crée
un volume et deux conteneurs successifs, envoie le client fictif de T08 au
premier, puis relit exactement la même ligne dans le second. Les ressources
créées pour ce test sont supprimées en fin de contrôle, même en cas d'échec.
Utilisable avec pytest localement et sans dépendance Python dans le job deploy.
"""

import argparse
import json
import subprocess
import sys
import time
from uuid import uuid4

from verifier_api_deployee import load_reference, verify_api, wait_ready


def verify_persistence(image: str, docker: str = "docker") -> dict:
    """Compare la ligne enregistrée avant et après remplacement du conteneur."""
    suffix = uuid4().hex[:12]
    volume = f"scoring-t13-{suffix}"
    containers = [f"{volume}-first", f"{volume}-second"]

    def command(*args):
        return subprocess.run([docker, *args], check=True, capture_output=True,
                              text=True, timeout=150).stdout.strip()

    def start(name):
        command("run", "--detach", "--name", name,
                "--publish", "127.0.0.1::7860",
                "--mount", f"type=volume,source={volume},target=/app/data",
                "--env", "API_CALLS_DB_PATH=/app/data/api_calls.sqlite3", image)
        address = command("port", name, "7860/tcp")
        if not address.startswith("127.0.0.1:"):
            raise AssertionError(f"Publication inattendue du port : {address}")
        url = f"http://{address}"
        wait_ready(url, time.monotonic() + 120)
        return url

    def read_calls(name):
        code = (
            "import sqlite3,json; "
            "db=sqlite3.connect('file:/app/data/api_calls.sqlite3?mode=ro',uri=True); "
            "db.row_factory=sqlite3.Row; "
            "print(json.dumps([dict(row) for row in db.execute("
            "\"SELECT * FROM api_calls ORDER BY timestamp_utc, request_id\")])); "
            "db.close()"
        )
        return json.loads(command("exec", name, "python", "-c", code))

    try:
        command("volume", "create", volume)
        url = start(containers[0])
        result = verify_api(url)
        before = read_calls(containers[0])
        if len(before) != 2 or {row["operation"] for row in before} != {"schema", "predict"}:
            raise AssertionError("Les appels /schema et /predict doivent être enregistrés.")
        row = next(row for row in before if row["operation"] == "predict")
        client, _ = load_reference()
        if row["status"] != "success" or json.loads(row["input_json"]) != client:
            raise AssertionError("L'entrée synthétique complète n'a pas été enregistrée.")
        for field in ("score", "prediction", "threshold", "model_version"):
            if row[field] != result[field]:
                raise AssertionError(f"Résultat SQLite incorrect : {field}")
        command("rm", "--force", containers[0])
        second_url = start(containers[1])
        after = read_calls(containers[1])
        if after != before:
            raise AssertionError("L'historique a changé après remplacement du conteneur.")
        verify_api(second_url)
        return {"request_id": row["request_id"], "timestamp_utc": row["timestamp_utc"],
                "status": row["status"], "duration_ms": row["duration_ms"],
                "score": row["score"], "prediction": row["prediction"],
                "threshold": row["threshold"], "model_version": row["model_version"],
                "input_columns": len(client),
                "OWN_CAR_AGE": json.loads(row["input_json"])["OWN_CAR_AGE"],
                "rows_before": len(before), "rows_after_replacement": len(after)}
    except Exception:
        for name in containers:
            logs = subprocess.run([docker, "logs", "--tail", "50", name],
                                  capture_output=True, text=True, timeout=20)
            print(logs.stdout + logs.stderr, file=sys.stderr)
        raise
    finally:
        for name in containers:
            subprocess.run([docker, "rm", "--force", name],
                           capture_output=True, timeout=30)
        subprocess.run([docker, "volume", "rm", volume],
                       capture_output=True, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--docker", default="docker")
    args = parser.parse_args()
    try:
        result = verify_persistence(args.image, args.docker)
    except (OSError, subprocess.SubprocessError, ValueError, AssertionError, RuntimeError) as error:
        print(f"T13 : échec de la persistance Docker : {error}", file=sys.stderr)
        return 1
    print("T13 : historique conservé après remplacement du conteneur.")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
