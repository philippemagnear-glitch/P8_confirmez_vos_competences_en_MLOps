"""Analyse du suivi de scoring, utilisable sans Streamlit.

Lit SQLite en lecture seule, prépare la référence avec le pipeline entraîné,
exécute Evidently et retourne ses résultats avec les indicateurs opérationnels.
Le module ne lance rien à l'import et ne gère ni interface ni cache.
"""

import os

os.environ["DO_NOT_TRACK"] = "1"

import hashlib
import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import MissingValueCount, ValueDrift

from app.prediction import ARTIFACTS_PATH, CreditScoringModel

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data/processed/application_train_final_encoded.csv"
DB_PATH = Path(os.environ.get("API_CALLS_DB_PATH", str(ROOT / "data/api_calls.sqlite3")))
MIN_OBSERVATIONS = 50
LOGGER = logging.getLogger(__name__)


def empreinte_reference() -> str:
    """Identifie le contenu des sources, pour ne pas réutiliser une référence périmée."""
    sources = [TRAIN_PATH] + sorted(p for p in ARTIFACTS_PATH.rglob("*") if p.is_file())
    if not (ARTIFACTS_PATH / "MLmodel").is_file():
        raise FileNotFoundError("Artefact local absent.")
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def charger_reference(empreinte: str) -> tuple:
    """Calcule une fois la référence ; le DataFrame partagé reste en lecture seule."""
    model = CreditScoringModel()
    batches = []
    for batch in pd.read_csv(TRAIN_PATH, usecols=model.feature_names, chunksize=10000):
        batch = batch.loc[:, model.feature_names]
        # La copie regroupe les blocs créés par la conversion des 310 colonnes.
        frame = batch.astype("float64").copy()
        frame["score"] = model.pipeline.predict_proba(batch)[:, model.default_class_index]
        batches.append(frame)
    if not batches:
        raise ValueError("Le train est vide.")
    reference = pd.concat(batches, ignore_index=True)
    if not reference["score"].between(0, 1).all():
        raise ValueError("Score de référence invalide.")
    if empreinte_reference() != empreinte:
        raise ValueError("Les sources ont changé pendant le calcul.")
    categories = [
        c.name for c in model.columns
        if c.type.name == "boolean" or (
            reference[c.name].notna().any()
            and reference[c.name].dropna().isin([0, 1]).all()
        )
    ]
    # Les noms après prétraitement évitent de décaler les gains si l'imputation
    # a supprimé une colonne vide. Cet encart facultatif ne bloque pas le suivi.
    importance = pd.Series(dtype="float64")
    try:
        names = list(model.pipeline[:-1].get_feature_names_out(model.feature_names))
        gains = model.pipeline.steps[-1][1].booster_.feature_importance(importance_type="gain")
        values = pd.Series(gains, index=names, dtype="float64")
        if (
            not values.index.is_unique or not set(names).issubset(model.feature_names)
            or values.isna().any() or values.isin([float("inf"), float("-inf")]).any()
            or values.lt(0).any() or values.sum() <= 0
        ):
            raise ValueError("Importances non exploitables.")
        importance = values / values.sum() * 100
    except (AttributeError, ValueError, TypeError):
        LOGGER.warning("Importance par gain indisponible : vérifier le modèle et les noms après prétraitement.")
    return reference, str(model.provenance["model_version"]), model.threshold, categories, importance


def lire_fenetres(database: Path, nombre: int) -> tuple:
    """Lit deux fenêtres sans chevauchement dans un même instantané SQLite."""
    if not database.is_file():
        raise FileNotFoundError("Base SQLite absente.")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)) as connection:
        calls = pd.read_sql_query(
            """SELECT status, model_version, input_json, score, prediction, threshold, duration_ms
               FROM api_calls WHERE operation = 'predict'
               ORDER BY timestamp_utc DESC, rowid DESC LIMIT ?""",
            connection, params=(2 * nombre,),
        )
    if not calls["status"].isin(["success", "rejected", "error"]).all():
        raise ValueError("Statut enregistré inconnu.")
    return calls.iloc[:nombre].copy(), calls.iloc[nombre:].copy()


def mesures_service(calls: pd.DataFrame) -> pd.Series:
    """Mesure les erreurs sur tous les appels et les durées sur les réussites."""
    durations = pd.to_numeric(calls.loc[calls["status"].eq("success"), "duration_ms"], errors="raise")
    if durations.isna().any() or durations.isin([float("inf"), float("-inf")]).any() or durations.lt(0).any():
        raise ValueError("Durée enregistrée invalide.")
    rows = []
    for status, label in [("rejected", "Rejets (%)"), ("error", "Pannes (%)")]:
        value = 100 * calls["status"].eq(status).mean() if len(calls) >= MIN_OBSERVATIONS else None
        rows.append((label, value))
    for quantile, label in [(0.5, "Médiane (ms)"), (0.95, "P95 (ms)")]:
        value = durations.quantile(quantile) if len(durations) >= MIN_OBSERVATIONS else None
        rows.append((label, value))
    return pd.Series(dict(rows), dtype="float64")


def mesures_performances(calls: pd.DataFrame) -> dict:
    """Agrège une fenêtre ; chaque durée utilise ses seules réussites mesurées."""
    success = calls.loc[calls["status"].eq("success")]
    result = {"Appels": len(calls), "Réussites": len(success)}
    for status, label in [("rejected", "Rejets (%)"), ("error", "Pannes (%)")]:
        result[label] = 100 * calls["status"].eq(status).mean() if len(calls) >= MIN_OBSERVATIONS else float("nan")
    for column, label in [
        ("latency_ms", "Latence médiane (ms)"),
        ("inference_ms", "Inférence médiane (ms)"),
        ("cpu_time_ms", "Temps CPU médian (ms)"),
    ]:
        values = pd.to_numeric(success[column], errors="coerce")
        values = values.loc[values.notna() & values.ge(0) & values.lt(float("inf"))]
        result[f"Mesures {column}"] = len(values)
        result[label] = values.median() if len(values) >= MIN_OBSERVATIONS else float("nan")
        if column == "latency_ms":
            result["Latence P95 (ms)"] = values.quantile(.95) if len(values) >= MIN_OBSERVATIONS else float("nan")
            result["P95 provisoire"] = 50 <= len(values) < 100
    if len(calls):
        dates = pd.to_datetime(calls["timestamp_utc"], utc=True, errors="raise").dt.tz_convert("Europe/Paris")
        result["Début"] = dates.min().strftime("%d/%m/%Y %H:%M:%S")
        result["Fin"] = dates.max().strftime("%d/%m/%Y %H:%M:%S")
    else:
        result.update({"Début": "—", "Fin": "—"})
    result["Versions du modèle"] = ", ".join(sorted(success["model_version"].dropna().astype(str).unique())) or "—"
    return result


def analyser_performances(database: Path, nombre: int = 100) -> dict:
    """Lit un instantané en lecture seule, indépendamment d'Evidently et du CSV.

    Les fenêtres sont ancrées sur les appels les plus récents. Les mesures
    absentes, négatives ou non finies sont exclues, jamais remplacées par zéro.
    Aucun groupe ne chevauche un autre ; le plus ancien peut être incomplet.
    """
    if not 50 <= nombre <= 500:
        raise ValueError("Choisir entre 50 et 500 appels par fenêtre.")
    if not database.is_file():
        raise FileNotFoundError("Base SQLite absente.")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(api_calls)")}
        # Un historique antérieur à l'instrumentation reste lisible.
        metrics = ", ".join(
            name if name in columns else f"NULL AS {name}"
            for name in ["latency_ms", "inference_ms", "cpu_time_ms"]
        )
        calls = pd.read_sql_query(
            f"SELECT timestamp_utc,status,model_version,{metrics} FROM api_calls "
            "WHERE operation='predict' ORDER BY timestamp_utc DESC,rowid DESC", connection,
        )
    if not calls["status"].isin(["success", "rejected", "error"]).all():
        raise ValueError("Statut enregistré inconnu.")
    current = mesures_performances(calls.iloc[:nombre])
    previous = mesures_performances(calls.iloc[nombre:2 * nombre])
    history = [mesures_performances(calls.iloc[start:start + nombre])
               for start in range(0, len(calls), nombre)]
    history = pd.DataFrame(history[::-1])
    history.index = pd.RangeIndex(1, len(history) + 1, name="Groupe")
    return {
        "actuel": current, "precedent": previous, "historique": history,
        "comparaison": current["Appels"] == previous["Appels"] == nombre,
        "parametres": (str(database.resolve()), nombre),
        "date": pd.Timestamp.now(tz="Europe/Paris").strftime("%d/%m/%Y à %H:%M:%S"),
    }


def rapport_drift(reference: pd.DataFrame, current: pd.DataFrame, categories: list,
                  distance: float, ks_global: float) -> pd.DataFrame:
    """Restitue les résultats Evidently et les null dans un tableau unique."""
    numerical = [name for name in reference if name not in categories]
    ks_threshold = ks_global / len(numerical)  # Bonferroni, score compris.
    summary = pd.DataFrame({
        "Renseignées train": reference.count(), "Renseignées API": current.count(),
    })
    summary[["Null train (%)", "Null API (%)"]] = float("nan")
    summary["Manquantes référence"] = len(reference) - summary["Renseignées train"]
    summary["Manquantes appels"] = len(current) - summary["Renseignées API"]
    enough = summary[["Renseignées train", "Renseignées API"]].ge(MIN_OBSERVATIONS).all(axis=1)
    missing_metrics = [MissingValueCount(column=name, tests=[], share_tests=[]) for name in reference]
    metrics = list(missing_metrics) if len(current) else []
    for name in summary.index[enough]:
        if name in categories:
            metrics.append(ValueDrift(column=name, method="jensenshannon", threshold=distance))
        else:
            metrics.extend([
                ValueDrift(column=name, method="wasserstein", threshold=distance),
                ValueDrift(column=name, method="ks", threshold=ks_threshold),
            ])
    scores, detections = {}, {}
    report = None
    if metrics:
        definition = DataDefinition(numerical_columns=numerical, categorical_columns=categories)
        report = Report(metrics, include_tests=True).run(
            current_data=Dataset.from_pandas(current, data_definition=definition),
            reference_data=Dataset.from_pandas(reference, data_definition=definition),
        )
        scores = {
            (m["config"]["column"], m["config"]["method"]): float(m["value"])
            for m in report.dict()["metrics"] if "method" in m["config"]
        }
        detections = {
            (test.metric_config.params["column"], test.metric_config.params["method"]): test.status.value
            for test in report.tests_results if "method" in test.metric_config.params
        }
        # Réutilise les comptes natifs des deux lots, sans tests sur les null.
        for metric in missing_metrics:
            actual = report.context.get_metric_result(metric)
            baseline = report.context.get_reference_metric_result(metric.metric_id)
            name = metric.column
            summary.loc[name, ["Manquantes référence", "Manquantes appels"]] = [baseline.count.value, actual.count.value]
            summary.loc[name, ["Null train (%)", "Null API (%)"]] = [baseline.share.value * 100, actual.share.value * 100]
    summary["Méthode"] = ["jensenshannon" if name in categories else "wasserstein" for name in reference]
    summary["Distance"] = [scores.get((name, method)) for name, method in summary["Méthode"].items()]
    summary["Seuil distance"] = distance
    summary["P-value KS"] = pd.Series({name: scores.get((name, "ks")) for name in reference}, dtype="float64")
    summary["Seuil KS"] = [None if name in categories else ks_threshold for name in reference]
    summary["Détection KS"] = [detections.get((name, "ks")) == "FAIL" for name in reference]
    labels = {"FAIL": "Dérive détectée", "SUCCESS": "Pas de dérive détectée"}
    summary["Résultat"] = [
        "Données insuffisantes" if not enough[name] else labels.get(detections.get((name, method)), "Calcul indisponible")
        for name, method in summary["Méthode"].items()
    ]
    summary.index.name = "Variable"
    return summary


def analyser_appels(
    database: Path, nombre: int, distance: float, ks_global: float,
    reference_chargee: tuple | None = None,
) -> dict:
    """Assemble un résultat agrégé sans écriture dans l'API ou dans SQLite."""
    if nombre < MIN_OBSERVATIONS or not 0 < ks_global < 1 or not 0 < distance < float("inf"):
        raise ValueError("Paramètres d'analyse invalides.")
    calls, previous = lire_fenetres(database, nombre)
    if reference_chargee is None:
        reference_chargee = charger_reference(empreinte_reference())
    reference, version, threshold, categories, importance = reference_chargee
    selected = calls.loc[
        calls["status"].eq("success") & calls["model_version"].eq(version)
        & calls["threshold"].eq(threshold)
    ]
    features = reference.columns.drop("score").tolist()
    inputs = selected["input_json"].map(json.loads).tolist()
    if any(not isinstance(item, dict) or not set(features).issubset(item) for item in inputs):
        raise ValueError("Entrée enregistrée incomplète.")
    current = pd.DataFrame(inputs, columns=features).astype("float64")
    current["score"] = pd.to_numeric(selected["score"], errors="raise").to_numpy()
    classes = selected["prediction"].reset_index(drop=True)
    if not current["score"].between(0, 1).all() or not classes.isin([0, 1]).all():
        raise ValueError("Score ou classe enregistré invalide.")
    if not classes.eq(current["score"].ge(threshold).astype(int)).all():
        raise ValueError("Classe incohérente avec le score et le seuil.")
    for frame in (reference, current):
        if frame.isin([float("inf"), float("-inf")]).any().any():
            raise ValueError("Valeur infinie dans les données.")
    summary = rapport_drift(reference, current, categories, distance, ks_global)
    decisions = None
    if min(len(reference), len(current)) >= MIN_OBSERVATIONS:
        decisions = pd.DataFrame({
            "Clients": [len(reference), len(current)],
            "Score moyen": [reference["score"].mean(), current["score"].mean()],
            "Défauts prédits (%)": [
                reference["score"].ge(threshold).mean() * 100, classes.eq(1).mean() * 100,
            ],
        }, index=["Train", "API"])
    operations = pd.DataFrame({"Précédent": mesures_service(previous), "Actuel": mesures_service(calls)})
    operations["Écart"] = (
        operations["Actuel"] - operations["Précédent"]
        if len(calls) == len(previous) == nombre else float("nan")
    )
    return {
        "parametres": (str(database.resolve()), nombre, distance, ks_global),
        "date": pd.Timestamp.now(tz="Europe/Paris").strftime("%d/%m/%Y à %H:%M:%S (heure de Paris)"),
        "version": version, "seuil": threshold, "appels": len(calls),
        "precedents": len(previous), "clients": len(current),
        "synthese": summary,
        "decisions": decisions,
        "operations": operations, "importance": importance,
    }


