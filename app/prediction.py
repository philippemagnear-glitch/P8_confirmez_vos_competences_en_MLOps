"""Chargement autonome du modèle et validation des données d'un client."""

import json
import math
from numbers import Integral, Real
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from mlflow.models import Model


ARTIFACTS_PATH = (
    Path(__file__).resolve().parent / "artifacts" / "credit_scoring_lightgbm" / "4"
)


class CreditScoringModel:
    """Associe le pipeline entraîné à sa signature et à sa provenance."""

    def __init__(self, artifacts_path: Path = ARTIFACTS_PATH):
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

    def schema(self) -> dict:
        """Expose le contrat d'entrée sans recopier les noms de variables."""
        return {
            "model_name": self.provenance["model_name"],
            "model_version": self.provenance["model_version"],
            "n_features": len(self.columns),
            "columns": [
                {
                    "name": column.name,
                    "type": column.type.name,
                    "required": column.required,
                    "nullable": column.type.name != "boolean",
                }
                for column in self.columns
            ],
        }

    def prepare_input(self, client: dict) -> pd.DataFrame:
        """Valide un client et ordonne ses variables selon la signature MLflow.

        Les nombres manquants deviennent des NaN pour l'imputation sauvegardée.
        Une colonne facultative omise est traitée comme une valeur manquante.
        Les chaînes numériques et les nombres non finis sont refusés.
        """
        if not isinstance(client, dict):
            raise ValueError("L'entrée doit être un objet JSON représentant un seul client.")
        if any(not isinstance(name, str) for name in client):
            raise ValueError("Les noms des variables doivent être des chaînes.")
        unknown = sorted(set(client).difference(self.feature_names))
        missing = [
            column.name
            for column in self.columns
            if column.required and column.name not in client
        ]
        if missing or unknown:
            details = []
            if missing:
                details.append(
                    "Variables obligatoires absentes : " + ", ".join(missing)
                )
            if unknown:
                details.append("Variables inconnues : " + ", ".join(unknown))
            raise ValueError("\n".join(details))

        values = {}
        dtypes = {}
        for column in self.columns:
            value = client.get(column.name)
            data_type = column.type.name
            if value is None:
                if data_type == "boolean":
                    raise ValueError(f"{column.name} : un booléen est attendu.")
                values[column.name] = float("nan")
                dtypes[column.name] = "float64"
                continue
            if data_type == "boolean":
                if not isinstance(value, bool):
                    raise ValueError(f"{column.name} : un booléen est attendu.")
                dtypes[column.name] = "bool"
            elif data_type in {"integer", "long"}:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f"{column.name} : un entier JSON est attendu.")
                bits = 32 if data_type == "integer" else 64
                if not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1):
                    raise ValueError(
                        f"{column.name} : entier hors des limites du schéma."
                    )
                dtypes[column.name] = f"int{bits}"
            else:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(f"{column.name} : un nombre JSON est attendu.")
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
                    raise ValueError(f"{column.name} : utiliser un nombre fini ou null.")
                dtypes[column.name] = "float64"
            values[column.name] = value

        return pd.DataFrame([values], columns=self.feature_names).astype(dtypes)

    def predict(self, client: dict) -> dict:
        """Retourne la probabilité de défaut et la classe au seuil retenu."""
        frame = self.prepare_input(client)
        probabilities = self.pipeline.predict_proba(frame)
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
