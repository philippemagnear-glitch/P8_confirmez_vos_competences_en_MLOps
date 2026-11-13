"""Vérifie qu'une API de scoring démarrée fournit la prédiction attendue.

Ce script joue le rôle d'un programme extérieur qui utilise l'API : il attend
qu'elle soit prête, demande son schéma, puis lui envoie le client fictif complet
de tests/fixtures/client_valid.json. Il vérifie le modèle disponible et compare
le score, la classe, le seuil et la version reçus à la référence T08 approuvée.

Le test d'intégration T08 réutilise ces fonctions. Dans GitHub Actions, le job
deploy exécute ce script après avoir démarré l'image produite par build, afin
de vérifier que l'application fonctionne réellement dans le conteneur Docker.
Ce contrôle temporaire ne constitue pas un hébergement permanent de l'API.

Le script ne démarre ni n'arrête le serveur et ne modifie pas la référence.
En ligne de commande, il termine avec un code d'échec si le contrôle échoue,
ce qui fait échouer le job GitHub Actions. Il utilise uniquement la bibliothèque
standard de Python, sans installer pytest sur la machine qui lance ce contrôle.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


FIXTURES = Path(__file__).resolve().parent / "fixtures"
ARTIFACTS = (
    Path(__file__).resolve().parents[1]
    / "app" / "artifacts" / "credit_scoring_lightgbm" / "4"
)


def load_reference(fixtures: Path = FIXTURES) -> tuple[dict, dict]:
    """Charge la référence figée ; toute absence ou modification est une erreur."""
    reference_path = fixtures / "t08_reference.json"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Référence de scoring absente : {reference_path}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    client_path = fixtures / reference["client_file"]
    if not client_path.is_file():
        raise FileNotFoundError(f"Ligne JSON de référence absente : {client_path}")
    client = json.loads(client_path.read_text(encoding="utf-8"))
    if not isinstance(client, dict) or len(client) != 310:
        raise ValueError("La ligne de référence doit contenir les 310 clés du modèle.")
    canonical = json.dumps(
        client, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != reference["client_sha256"]:
        raise ValueError("Le JSON ne correspond plus à la ligne de référence figée.")
    return client, reference


def verify_model_artifact(reference: dict, artifacts: Path = ARTIFACTS) -> None:
    """Vérifie que le pipeline v4 attendu est disponible dans le dépôt."""
    for filename in ("model.pkl", "MLmodel", "provenance.json"):
        if not (artifacts / filename).is_file():
            raise FileNotFoundError(f"Artefact de modèle v4 manquant : {artifacts / filename}")
    digest = hashlib.sha256((artifacts / "model.pkl").read_bytes()).hexdigest()
    if digest != reference["model_sha256"]:
        raise ValueError("Le pipeline disponible diffère de l'artefact v4 de référence.")
    provenance = json.loads((artifacts / "provenance.json").read_text(encoding="utf-8"))
    for field in ("model_name", "model_version", "model_id", "run_id"):
        if provenance[field] != reference[field]:
            raise ValueError(f"Provenance du modèle incompatible : {field}")


def remaining_time(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Délai dépassé en attendant l'API de scoring.")
    return remaining


def wait_ready(base_url: str, deadline: float) -> None:
    """Attend la publication des informations API, avec une limite de temps."""
    while True:
        timeout = min(3.0, remaining_time(deadline))
        try:
            with urlopen(f"{base_url}/gradio_api/info", timeout=timeout) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(min(0.5, remaining_time(deadline)))


def call_endpoint(base_url: str, name: str, arguments: list, deadline: float) -> dict:
    """Appelle une fonction Gradio et lit son résultat dans le flux SSE."""
    request = Request(
        f"{base_url}/gradio_api/call/{name}",
        data=json.dumps({"data": arguments}, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=remaining_time(deadline)) as response:
        event_id = json.load(response)["event_id"]
    result_url = f"{base_url}/gradio_api/call/{name}/{event_id}"
    with urlopen(result_url, timeout=remaining_time(deadline)) as response:
        event = None
        for raw_line in response:
            remaining_time(deadline)
            line = raw_line.decode("utf-8").strip()
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if event == "error":
                    raise RuntimeError(f"L'endpoint /{name} a signalé une erreur : {data}")
                if event == "complete":
                    if not isinstance(data, list) or len(data) != 1:
                        raise ValueError(f"Réponse Gradio inattendue pour /{name}.")
                    return data[0]
    raise RuntimeError(f"Aucun résultat complet reçu pour /{name}.")


def verify_prediction(result: dict, reference: dict) -> None:
    """Applique les mêmes assertions au vrai modèle local et au conteneur."""
    if not isinstance(result, dict):
        raise ValueError("La réponse de prédiction doit être un objet JSON.")
    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Le score retourné n'est pas numérique.")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Le score doit être fini et compris entre 0 et 1.")
    for field in ("threshold", "model_name", "model_version"):
        if result.get(field) != reference[field]:
            raise ValueError(f"Résultat incompatible avec la référence : {field}")
    prediction = result.get("prediction")
    if type(prediction) is not int or prediction not in (0, 1):
        raise ValueError("La classe prédite doit être l'entier 0 ou 1.")
    if prediction != int(score >= result["threshold"]):
        raise ValueError("La classe retournée ne correspond pas au seuil de décision.")
    if prediction != reference["prediction"]:
        raise ValueError("La classe retournée diffère de la référence.")
    if not math.isclose(score, reference["score"], rel_tol=0, abs_tol=reference["absolute_tolerance"]):
        raise ValueError("Le score diffère de la référence au-delà de la tolérance autorisée.")


def verify_api(base_url: str, timeout: float = 120) -> dict:
    """Attend l'API, vérifie /schema, puis /predict avec l'unique JSON de référence."""
    client, reference = load_reference()
    verify_model_artifact(reference)
    base_url = base_url.rstrip("/")
    deadline = time.monotonic() + timeout
    wait_ready(base_url, deadline)
    schema = call_endpoint(base_url, "schema", [], deadline)
    columns = schema.get("columns", [])
    names = [column["name"] for column in columns]
    if schema.get("n_features") != 310 or len(names) != 310 or set(names) != set(client):
        raise ValueError("/schema doit exposer exactement les 310 variables de référence.")
    if not all(column.get("required") is True for column in columns):
        raise ValueError("/schema doit déclarer toutes les clés obligatoires.")
    for field in ("model_name", "model_version"):
        if schema.get(field) != reference[field]:
            raise ValueError(f"/schema expose un modèle incompatible : {field}")
    result = call_endpoint(base_url, "predict", [client], deadline)
    verify_prediction(result, reference)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        result = verify_api(args.url, args.timeout)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"Échec de la vérification API : {error}", file=sys.stderr)
        return 1
    print("/schema puis /predict vérifiés avec le vrai modèle et la référence figée.")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
