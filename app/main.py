"""Interface Gradio et API : lancer depuis la racine avec python -m app.main."""

import json
import logging
import os
from pathlib import Path

# Désactive la télémétrie Gradio avant son import.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import pandas as pd

from app.prediction import CreditScoringModel


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


def create_demo() -> gr.Blocks:
    """Charge une fois le modèle et construit l'interface et ses points d'accès."""
    model = CreditScoringModel()
    try:
        demo_clients = load_demo_clients(model)
    except (OSError, ValueError) as error:
        LOGGER.warning("Échantillon CSV indisponible : %s", error)
        demo_clients = {}

    def predict(client: dict) -> dict:
        """Prédit pour un client préparé ; le score est la probabilité de TARGET=1."""
        try:
            return model.predict(client)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        except Exception as error:
            LOGGER.exception("Échec de la prédiction.")
            raise gr.Error(
                "La prédiction a échoué. Consulter les logs du serveur."
            ) from error

    def get_schema() -> dict:
        """Retourne les variables attendues, leurs types et leur caractère obligatoire."""
        return model.schema()

    def select_client(client_id: str) -> tuple[str, None]:
        if client_id not in demo_clients:
            raise gr.Error("Sélectionner un client de l'échantillon.")
        return json.dumps(demo_clients[client_id], indent=2, allow_nan=False), None

    def predict_json(client_json: str) -> dict:
        try:
            client = json.loads(client_json)
        except (json.JSONDecodeError, TypeError) as error:
            raise gr.Error("Le contenu saisi n'est pas un JSON valide.") from error
        return predict(client)

    with gr.Blocks(title="Scoring crédit", analytics_enabled=False) as demo:
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
                "Utilisez les noms du schéma ci-dessous. Les colonnes inconnues sont "
                "refusées. Les valeurs numériques manquantes s'écrivent `null` ; "
                "elles sont traitées par l'imputation du modèle. "
                "L'identifiant client et TARGET ne sont pas des variables d'entrée."
            )
            gr.JSON(value=model.schema(), label="Signature MLflow")

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
            concurrency_id="prediction",
            concurrency_limit=1,
        )
        gr.api(get_schema, api_name="schema")

    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_demo().queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
        footer_links=["api"],
        run_history=False,
    )
