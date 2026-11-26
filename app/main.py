"""Interface Gradio et API : lancer depuis la racine avec python -m app.main."""

import json
import logging
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter, process_time
from typing import Annotated
from uuid import UUID, uuid4

# Désactive la télémétrie Gradio avant son import.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import pandas as pd
from fastapi import Body, FastAPI, HTTPException

from app.prediction import CreditScoringModel, InputValidationError
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

    def run_operation(operation, action, client=None, measurements=None):
        """Mesure le traitement et trace son issue, même si le calcul échoue."""
        started = perf_counter()
        cpu_started = process_time()
        request_id = str(uuid4())
        measurements = measurements if measurements is not None else {}
        started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        status, result, reason = "error", None, None
        try:
            result = action()
            if operation == "predict":
                result = {**result, "request_id": request_id}
            status = "success"
            return result
        except InputValidationError as error:
            status, reason = "rejected", str(error)
            raise gr.Error(str(error), print_exception=False) from None
        except Exception as error:
            # Le texte de l'exception peut contenir des données client.
            message = (
                "La prédiction a échoué. Consulter les logs du serveur."
                if operation == "predict" else
                "La lecture du schéma a échoué. Consulter les logs du serveur."
            )
            reason = f"{message} ({type(error).__name__})"
            LOGGER.error("Échec de la %s (%s).",
                         "prédiction" if operation == "predict" else "lecture du schéma",
                         type(error).__name__)
            raise gr.Error(message, print_exception=False) from None
        finally:
            elapsed_ms = (perf_counter() - started) * 1000
            cpu_ms = (process_time() - cpu_started) * 1000
            measurements.update(
                cpu_time_ms=cpu_ms,
                cpu_percent=100 * cpu_ms / elapsed_ms if elapsed_ms else 0.0,
                execution_device=model.execution_device if operation == "predict" else None,
            )
            call_store.record(operation=operation, status=status, started_at=started_at,
                              started=started, inputs=client, result=result, reason=reason,
                              request_id=request_id, measurements=measurements)

    def predict(client: dict) -> dict:
        """Prédit pour un client préparé ; le score est la probabilité de TARGET=1."""
        measurements = {}
        return run_operation(
            "predict", lambda: model.predict(client, measurements=measurements),
            client, measurements,
        )

    def get_schema() -> dict:
        """Retourne les variables attendues, leurs types et leur caractère obligatoire."""
        return run_operation("schema", model.schema)

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

    return demo


def create_app(model: CreditScoringModel | None = None) -> FastAPI:
    """Expose la santé et Gradio après chargement réussi du modèle.

    /health confirme que le serveur répond et que le modèle a été chargé.
    Il ne teste ni une prédiction, ni SQLite, ni le dashboard.
    """
    if model is None:
        model = CreditScoringModel()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    demo = create_demo(model).queue()
    store = ApiCallStore()

    @app.post("/measurement-runs")
    def create_run(
        name: Annotated[str, Body(min_length=1, max_length=120)],
        metadata: Annotated[dict, Body()],
    ):
        """Identifie un lot de mesures, sans créer d'appel de scoring supplémentaire."""
        try:
            return {"run_id": store.create_run(name, metadata)}
        except (OSError, sqlite3.Error) as error:
            LOGGER.error("Création du lot impossible (%s).", type(error).__name__)
            raise HTTPException(503, "Stockage SQLite indisponible.") from None
        except (TypeError, ValueError):
            raise HTTPException(422, "Métadonnées JSON invalides.") from None

    @app.post("/measurements")
    def save_latency(
        request_id: Annotated[UUID, Body()],
        run_id: Annotated[UUID, Body()],
        latency_ms: Annotated[float, Body(strict=True, ge=0, allow_inf_nan=False)],
        client_timestamp_utc: Annotated[datetime, Body()],
        client_sequence: Annotated[int, Body(strict=True, ge=1)],
    ):
        """Ajoute la latence côté client à une prédiction déjà enregistrée."""
        if client_timestamp_utc.utcoffset() is None:
            raise HTTPException(422, "La date doit préciser son fuseau horaire.")
        try:
            store.record_latency(
                request_id=str(request_id), run_id=str(run_id), latency_ms=latency_ms,
                client_timestamp_utc=client_timestamp_utc.astimezone(timezone.utc).isoformat(),
                client_sequence=client_sequence,
            )
        except KeyError as error:
            raise HTTPException(404, error.args[0]) from None
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from None
        except (OSError, sqlite3.Error) as error:
            LOGGER.error("Mesure non enregistrée (%s).", type(error).__name__)
            raise HTTPException(503, "Stockage SQLite indisponible.") from None
        return {"status": "saved", "request_id": str(request_id)}

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True}

    # Déclarer /health avant le montage à la racine, qui reçoit les autres routes.
    return gr.mount_gradio_app(
        app, demo, path="/", footer_links=["api"], run_history=False,
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        create_app(),
        host=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
