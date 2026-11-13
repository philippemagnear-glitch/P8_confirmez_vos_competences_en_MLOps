"""Protège la référence commune contre une omission ou un faux résultat positif."""

import json

import pytest

from verifier_api_deployee import load_reference, verify_model_artifact, verify_prediction


def test_T08_missing_reference_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="Référence de scoring absente"):
        load_reference(tmp_path)


def test_T08_missing_client_fails(tmp_path):
    _, reference = load_reference()
    (tmp_path / "t08_reference.json").write_text(json.dumps(reference), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Ligne JSON de référence absente"):
        load_reference(tmp_path)


def test_T08_modified_reference_client_fails(tmp_path):
    client, reference = load_reference()
    client["AMT_INCOME_TOTAL"] += 1
    (tmp_path / "t08_reference.json").write_text(json.dumps(reference), encoding="utf-8")
    (tmp_path / reference["client_file"]).write_text(json.dumps(client), encoding="utf-8")
    with pytest.raises(ValueError, match="référence figée"):
        load_reference(tmp_path)


def test_T08_missing_model_fails(tmp_path):
    _, reference = load_reference()
    with pytest.raises(FileNotFoundError, match="Artefact de modèle v4 manquant.*model.pkl"):
        verify_model_artifact(reference, tmp_path)


@pytest.mark.parametrize("field,value,message", [
    ("score", float("nan"), "fini"),
    ("score", float("inf"), "fini"),
    ("score", -0.1, "fini"),
    ("score", 1.1, "fini"),
    ("score", True, "numérique"),
    ("score", 0.5, "référence"),
    ("prediction", 0, "seuil"),
    ("prediction", True, "entier"),
    ("threshold", 0.4, "threshold"),
    ("model_version", "5", "model_version"),
    ("model_name", "autre_modele", "model_name"),
])
def test_T08_shared_checker_rejects_invalid_result(field, value, message):
    _, reference = load_reference()
    result = {name: reference[name] for name in (
        "score", "prediction", "threshold", "model_version", "model_name"
    )}
    result[field] = value
    with pytest.raises(ValueError, match=message):
        verify_prediction(result, reference)


def test_T08_shared_tolerance_is_one_millionth():
    _, reference = load_reference()
    assert reference["absolute_tolerance"] == 0.000001
    result = {name: reference[name] for name in (
        "score", "prediction", "threshold", "model_version", "model_name"
    )}
    result["score"] += 0.0000005
    verify_prediction(result, reference)
    result["score"] += 0.000002
    with pytest.raises(ValueError, match="tolérance"):
        verify_prediction(result, reference)
