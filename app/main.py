"""Interface Gradio et API : lancer depuis la racine avec python -m app.main."""

import json
import logging
import math
import os
import sqlite3
from pathlib import Path
from uuid import UUID

# Désactive la télémétrie Gradio avant son import.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import pandas as pd

from app.prediction import CreditScoringModel, InputValidationError
from app.scoring_service import process_api_call
from app.storage import ApiCallStore


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_CSV = PROJECT_ROOT / "data" / "processed" / "application_test_final_encoded.csv"
SAMPLE_SIZE = 50


def load_demo_clients(model: CreditScoringModel) -> dict[str, dict]:
    """Lit au plus 50 clients, en conservant les types et l'ordre du modèle."""
    if not DEMO_CSV.is_file():
        return {}
    pandas_types = {
        "integer": "Int32",
        "long": "Int64",
        "float": "Float64",
        "double": "Float64",
        "boolean": "boolean",
    }
    dtype = {
        column.name: pandas_types[column.type.name] for column in model.columns
    }
    dtype["SK_ID_CURR"] = "string"
    selected = ["SK_ID_CURR", *model.feature_names]
    frame = pd.read_csv(DEMO_CSV, usecols=selected, dtype=dtype, nrows=SAMPLE_SIZE)
    clients = json.loads(frame[model.feature_names].to_json(orient="records"))
    return dict(zip(frame["SK_ID_CURR"].tolist(), clients, strict=True))


def create_demo(model: CreditScoringModel | None = None) -> gr.Blocks:
    """Charge une fois le modèle et construit l'interface et ses points d'accès."""
    if model is None:
        model = CreditScoringModel()
    call_store = ApiCallStore()
    try:
        demo_clients = load_demo_clients(model)
    except (OSError, ValueError) as error:
        LOGGER.warning("Échantillon CSV indisponible : %s", error)
        demo_clients = {}

    def handle_api_call(operation, client=None):
        """Adapte les erreurs du traitement commun aux messages Gradio."""
        try:
            return process_api_call(model, call_store, operation, client)
        except InputValidationError as error:
            raise gr.Error(str(error), print_exception=False) from None
        except Exception:
            message = (
                "La prédiction a échoué. Consulter les logs du serveur."
                if operation == "predict" else
                "La lecture du schéma a échoué. Consulter les logs du serveur."
            )
            raise gr.Error(message, print_exception=False) from None

    def predict(client: dict) -> dict:
        """Prédit pour un client préparé ; le score est la probabilité de TARGET=1."""
        return handle_api_call("predict", client)

    def get_schema() -> dict:
        """Retourne les variables attendues, leurs types et leur caractère obligatoire."""
        return handle_api_call("schema")

    def health() -> dict:
        """Confirme que Gradio répond après chargement du modèle, sans prédiction."""
        return {"status": "ok", "model_loaded": True}

    def save_latency(measurement: dict) -> dict:
        """Enregistre request_id et latency_ms pour une prédiction réussie existante."""
        if not isinstance(measurement, dict):
            raise gr.Error("La mesure doit être un objet JSON.", print_exception=False)
        request_id = measurement.get("request_id")
        try:
            if not isinstance(request_id, str):
                raise ValueError
            request_id = str(UUID(request_id))
        except ValueError:
            raise gr.Error("request_id doit être un UUID valide.", print_exception=False) from None
        latency_ms = measurement.get("latency_ms")
        try:
            valid_latency = (
                type(latency_ms) in (int, float)
                and math.isfinite(latency_ms) and latency_ms >= 0
            )
        except OverflowError:
            valid_latency = False
        if not valid_latency:
            raise gr.Error(
                "latency_ms doit être un nombre fini positif ou nul.", print_exception=False,
            )
        try:
            call_store.record_latency(request_id=request_id, latency_ms=latency_ms)
        except (KeyError, RuntimeError) as error:
            raise gr.Error(error.args[0], print_exception=False) from None
        except (OSError, sqlite3.Error) as error:
            LOGGER.error("Mesure non enregistrée (%s).", type(error).__name__)
            raise gr.Error("Stockage SQLite indisponible.", print_exception=False) from None
        return {"status": "saved", "request_id": request_id}

    def select_client(client_id: str) -> tuple[str, None]:
        if client_id not in demo_clients:
            raise gr.Error("Sélectionner un client de l'échantillon.", print_exception=False)
        return json.dumps(demo_clients[client_id], indent=2, allow_nan=False), None

    def predict_json(client_json: str) -> dict:
        try:
            client = json.loads(client_json)
        except (json.JSONDecodeError, TypeError):
            raise gr.Error(
                "Le contenu saisi n'est pas un JSON valide.", print_exception=False
            ) from None
        return predict(client)

    with gr.Blocks(title="Scoring crédit", analytics_enabled=False) as demo:
        monitoring_url = os.environ.get("MONITORING_URL", "http://localhost:8501")
        gr.Button("Ouvrir le dashboard", link=monitoring_url, link_target="_blank")
        gr.Markdown(
            "# Scoring crédit\n"
            "Obtenez la probabilité de défaut d'un client à partir de ses données "
            "déjà préparées. Le score est compris entre 0 et 1 ; "
            "la classe 1 correspond à un défaut prédit."
        )
        if demo_clients:
            gr.Markdown("Choisissez un client parmi les 50 premières lignes du CSV.")
        else:
            gr.Markdown(
                "Échantillon CSV absent ou incompatible. "
                "Vous pouvez saisir un objet JSON ou appeler l'API."
            )
        client_selector = gr.Dropdown(
            choices=list(demo_clients),
            label="Client de démonstration",
            interactive=bool(demo_clients),
        )
        load_button = gr.Button("Charger les données", interactive=bool(demo_clients))
        client_json = gr.Code(
            label="Données du client (JSON)", language="json", lines=15
        )
        predict_button = gr.Button("Calculer le score", variant="primary")
        result = gr.JSON(label="Résultat")
        with gr.Accordion("Contrat des données d'entrée", open=False):
            gr.Markdown(
                "Toutes les clés du schéma ci-dessous sont obligatoires. "
                "Les colonnes inconnues sont refusées. Les valeurs numériques "
                "manquantes s'écrivent `null` et sont traitées par l'imputation "
                "du modèle. La valeur `0` est conservée. "
                "L'identifiant client et TARGET ne sont pas des variables d'entrée."
            )
            gr.JSON(value=model.schema(), label="Contrat d'entrée de l'API")

        load_button.click(
            select_client,
            inputs=client_selector,
            outputs=[client_json, result],
            api_visibility="private",
        )
        client_selector.change(
            lambda: ("", None),
            inputs=None,
            outputs=[client_json, result],
            api_visibility="private",
        )
        predict_button.click(
            predict_json,
            inputs=client_json,
            outputs=result,
            api_visibility="private",
            concurrency_id="prediction",
            concurrency_limit=1,
        )
        gr.api(
            predict,
            api_name="predict",
            api_description=(
                "Reçoit un objet JSON contenant les 310 clés obligatoires et retourne "
                "score, prediction, threshold, model_name et model_version. "
                "request_id permet de retrouver les mesures de cet appel dans SQLite. "
                "Les valeurs numériques null sont imputées ; zéro est conservé. "
                "Données invalides : gradio_client lève AppError avec un message "
                "nommant le champ à corriger. Panne interne du calcul : AppError "
                "avec le message générique « La prédiction a échoué. Consulter les "
                "logs du serveur. », sans données client ni détails internes. "
                "Aucun nouveau score n'est retourné en cas d'erreur. "
                "En HTTP brut, le POST /gradio_api/call/predict fournit un event_id ; "
                "le GET associé transmet un événement SSE complete (résultat) ou "
                "error (erreur). Le statut du POST ne suffit pas à conclure au succès."
            ),
            concurrency_id="prediction",
            concurrency_limit=1,
        )
        gr.api(get_schema, api_name="schema")
        gr.api(health, api_name="health", queue=False)
        gr.api(
            save_latency, api_name="measurements", queue=False,
            api_description=(
                "Reçoit un objet JSON measurement contenant request_id (UUID) et "
                "latency_ms (nombre fini positif ou nul). Complète une prédiction "
                "réussie déjà enregistrée. Renvoyer la même mesure est accepté ; "
                "une valeur différente pour cet appel est refusée. "
                "Une mesure invalide, un appel absent ou une erreur SQLite produit AppError."
            ),
        )

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_demo().queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False, footer_links=["api"], run_history=False,
    )
