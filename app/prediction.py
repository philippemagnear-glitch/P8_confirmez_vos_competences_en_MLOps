"""Chargement autonome du modèle et validation des données d'un client."""

import json
import logging
import math
import os
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter

import mlflow.sklearn
import numpy as np
from mlflow.models import Model


ARTIFACTS_PATH = (
    Path(__file__).resolve().parent / "artifacts" / "credit_scoring_lightgbm" / "4"
)


class InputValidationError(ValueError):
    """Distingue une requête incorrecte d'une panne interne du calcul."""


# Définitions Home Credit et agrégations des notebooks 01, 02 et 03.
NON_NEGATIVE_FIELDS = {
    "OWN_CAR_AGE", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    "BUREAU_MAX_DAYS_OVERDUE", "BUREAU_PROLONGATION_SUM",
    "PREV_APPLICATION_COUNT", "PREV_INSTAL_MAX_DAYS_LATE",
    "PREV_APPROVED_CNT_PAYMENT_MEAN", "PREV_POS_REMAINING_INSTALLMENTS_SUM",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "BUREAU_CREDIT_SUM", "BUREAU_OVERDUE_SUM",
    "PREV_APPROVED_ANNUITY_MEAN", "PREV_APPROVED_CREDIT_SUM",
    "BUREAU_PROLONGATION_PER_CREDIT",
    "PREV_APPROVED_CREDIT_TO_APPLICATION_RATIO_MEAN",
}
PAST_DATE_FIELDS = {
    "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE", "BUREAU_DAYS_CREDIT_MAX", "PREV_DAYS_DECISION_MAX",
}
# Ces rapports sont des moyennes d'indicateurs ou des parts d'un nombre total.
PROPORTION_FIELDS = {
    "BUREAU_ACTIVE_CREDIT_RATIO", "BUREAU_BB_HISTORY_RATIO",
    "PREV_APPROVED_RATIO", "PREV_REFUSED_RATIO", "PREV_CASH_LOAN_RATIO",
    "PREV_CONSUMER_LOAN_RATIO", "PREV_REVOLVING_LOAN_RATIO",
    "PREV_INSURED_APPROVAL_RATIO", "PREV_HIGH_YIELD_APPROVAL_RATIO",
    "PREV_POS_HISTORY_RATIO", "PREV_INSTAL_HISTORY_RATIO",
    "PREV_INSTAL_LATE_PAYMENT_RATIO_MEAN", "PREV_INSTAL_UNDERPAYMENT_RATIO_MEAN",
    "PREV_CC_HISTORY_RATIO",
}


def validate_value(name: str, value: Real) -> None:
    """Applique seulement les bornes établies pour les variables préparées.

    Les soldes et les rapports de montants ne sont pas assimilés à des
    proportions. Le revenu nul reste accepté faute de règle contraire établie.
    """
    is_count = (
        name.startswith("AMT_REQ_CREDIT_BUREAU_")
        or (name.startswith("BUREAU_") and name.endswith("_COUNT"))
    )
    if (name in NON_NEGATIVE_FIELDS or is_count) and value < 0:
        raise InputValidationError(f"{name} : la valeur doit être supérieure ou égale à 0.")
    if name == "DAYS_BIRTH" and value >= 0:
        raise InputValidationError(f"{name} : la date de naissance relative doit être < 0.")
    if name == "HOUR_APPR_PROCESS_START" and not 0 <= value <= 23:
        raise InputValidationError(f"{name} : l'heure doit être comprise entre 0 et 23.")
    if name in PAST_DATE_FIELDS and value > 0:
        raise InputValidationError(
            f"{name} : une date relative passée doit être <= 0 ; "
            "les sentinelles brutes doivent être préparées en null."
        )
    if name in PROPORTION_FIELDS and not 0 <= value <= 1:
        raise InputValidationError(f"{name} : la proportion doit être comprise entre 0 et 1.")


class CreditScoringModel:
    """Exécute avec ONNX Runtime le pipeline LightGBM exporté depuis MLflow."""

    def __init__(self, artifacts_path: Path = ARTIFACTS_PATH):
        self.backend = "onnx"
        self.execution_device = "cpu"
        artifacts_path = Path(artifacts_path)
        with (artifacts_path / "provenance.json").open(encoding="utf-8") as file:
            self.provenance = json.load(file)

        metadata = Model.load(str(artifacts_path / "MLmodel"))
        if metadata.signature is None or metadata.signature.inputs is None:
            raise ValueError("L'artefact MLflow ne contient pas de signature d'entrée.")
        self.columns = list(metadata.signature.inputs.inputs)
        supported_types = {"integer", "long", "float", "double", "boolean"}
        if not self.columns or any(
            not column.name or column.type.name not in supported_types
            for column in self.columns
        ):
            raise ValueError("La signature doit contenir des colonnes numériques nommées.")
        self.feature_names = [column.name for column in self.columns]
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("La signature contient des noms de colonnes dupliqués.")
        self.expected_features = frozenset(self.feature_names)
        self.input_fields = tuple(
            (column.name, column.type.name)
            for column in self.columns
        )
        if (
            metadata.run_id != self.provenance["run_id"]
            or metadata.model_id != self.provenance["model_id"]
        ):
            raise ValueError("Le modèle et sa provenance ne correspondent pas.")
        self.threshold = float(self.provenance["threshold"])
        if not 0 <= self.threshold <= 1:
            raise ValueError("Le seuil de décision doit être compris entre 0 et 1.")

        # Le chargement local ne consulte ni le registre ni la base MLflow.
        self.pipeline = mlflow.sklearn.load_model(str(artifacts_path))
        if list(self.pipeline.feature_names_in_) != self.feature_names:
            raise ValueError("Les variables du pipeline diffèrent de sa signature MLflow.")
        classes = list(self.pipeline.classes_)
        if len(classes) != 2 or 1 not in classes:
            raise ValueError("Le modèle doit prédire la classe de défaut 1.")
        self.default_class_index = classes.index(1)
        try:
            threads = int(os.environ.get("MODEL_NUM_THREADS", "1"))
        except ValueError:
            raise ValueError("MODEL_NUM_THREADS doit être un entier strictement positif.") from None
        if threads < 1:
            raise ValueError("MODEL_NUM_THREADS doit être un entier strictement positif.")
        self.onnx_session = self.create_onnx_session(threads)
        logging.getLogger(__name__).info("ONNX Runtime CPU : %s thread(s).", threads)

    def create_onnx_session(self, threads):
        """Convertit le pipeline une fois en mémoire et prépare son exécution CPU.

        Le pipeline original reste disponible pour le monitoring.
        L'artefact MLflow sauvegardé n'est jamais modifié.
        """
        import onnxruntime as ort
        from lightgbm import LGBMClassifier
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
            convert_lightgbm,
        )
        from skl2onnx import convert_sklearn, update_registered_converter
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
        )

        estimator = self.pipeline.steps[-1][1]
        if not isinstance(estimator, LGBMClassifier):
            raise ValueError("Le moteur ONNX attend un pipeline terminé par LGBMClassifier.")
        update_registered_converter(
            LGBMClassifier, "LightGbmLGBMClassifier",
            calculate_linear_classifier_output_shapes, convert_lightgbm,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
        graph = convert_sklearn(
            self.pipeline,
            initial_types=[("input", FloatTensorType([None, len(self.feature_names)]))],
            options={id(estimator): {"zipmap": False}},
            target_opset={"": 15, "ai.onnx.ml": 2},
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(
            graph.SerializeToString(), options, providers=["CPUExecutionProvider"],
        )

    def schema(self) -> dict:
        """Expose les variables MLflow avec une présence obligatoire côté API."""
        return {
            "model_name": self.provenance["model_name"],
            "model_version": self.provenance["model_version"],
            "n_features": len(self.columns),
            "columns": [
                {
                    "name": column.name,
                    "type": column.type.name,
                    "required": True,
                    "nullable": column.type.name != "boolean",
                }
                for column in self.columns
            ],
        }

    def prepare_input(self, client: dict) -> np.ndarray:
        """Valide un client et ordonne ses variables selon la signature MLflow.

        Les nombres manquants deviennent des NaN pour l'imputation sauvegardée.
        Toutes les clés sont obligatoires, même si la signature MLflow les
        déclare facultatives. Seul None indique une valeur manquante : 0 est
        conservé comme valeur numérique.
        Les chaînes numériques et les nombres non finis sont refusés.
        Le tableau float32 est construit directement dans l'ordre du modèle.
        """
        if not isinstance(client, dict):
            raise InputValidationError(
                "L'entrée doit être un objet JSON représentant un seul client."
            )
        if any(not isinstance(name, str) for name in client):
            raise InputValidationError("Les noms des variables doivent être des chaînes.")
        if client.keys() != self.expected_features:
            unknown = sorted(client.keys() - self.expected_features)
            missing = [name for name in self.feature_names if name not in client]
            details = []
            if missing:
                details.append(
                    "Variables obligatoires absentes : " + ", ".join(missing)
                )
            if unknown:
                details.append("Variables inconnues : " + ", ".join(unknown))
            raise InputValidationError("\n".join(details))

        # Tableau propre à cet appel : aucune donnée mutable partagée entre clients.
        values = np.empty((1, len(self.input_fields)), dtype=np.float32)
        for index, (name, data_type) in enumerate(self.input_fields):
            value = client[name]
            if value is None:
                if data_type == "boolean":
                    raise InputValidationError(f"{name} : un booléen est attendu.")
                values[0, index] = np.nan
                continue
            if data_type == "boolean":
                if not isinstance(value, bool):
                    raise InputValidationError(f"{name} : un booléen est attendu.")
            elif data_type in {"integer", "long"}:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise InputValidationError(f"{name} : un entier JSON est attendu.")
                bits = 32 if data_type == "integer" else 64
                if not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1):
                    raise InputValidationError(
                        f"{name} : entier hors des limites du schéma."
                    )
            else:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise InputValidationError(f"{name} : un nombre JSON est attendu.")
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise InputValidationError(
                        f"{name} : utiliser un nombre fini ou null."
                    )
            if data_type != "boolean":
                validate_value(name, value)
            values[0, index] = value
        return values

    def predict(self, client: dict, *, measurements: dict | None = None) -> dict:
        """Enchaîne préparation, inférence et construction de la réponse."""
        started = perf_counter()
        try:
            frame = self.prepare_input(client)
        finally:
            if measurements is not None:
                measurements["preparation_ms"] = (perf_counter() - started) * 1000
        probabilities = self.run_inference(frame, measurements=measurements)
        return self.format_prediction(probabilities)

    def run_inference(
        self, frame: np.ndarray, *, measurements: dict | None = None,
    ):
        """Exécute l'imputation et la classification dans la session ONNX partagée."""
        started = perf_counter()
        try:
            probabilities = self.onnx_session.run(None, {"input": frame})[1]
            if probabilities.shape != (len(frame), 2):
                raise RuntimeError("Le moteur ONNX a retourné une sortie incompatible.")
            return probabilities
        finally:
            if measurements is not None:
                # Préparation du tableau exclue ; imputation et classification incluses.
                measurements["inference_ms"] = (perf_counter() - started) * 1000

    def format_prediction(self, probabilities) -> dict:
        """Vérifie le score et construit la classe et les métadonnées de réponse."""
        score = float(probabilities[0, self.default_class_index])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError("Le modèle a retourné une probabilité invalide.")
        return {
            "score": score,
            "prediction": int(score >= self.threshold),
            "threshold": self.threshold,
            "model_name": self.provenance["model_name"],
            "model_version": self.provenance["model_version"],
        }
