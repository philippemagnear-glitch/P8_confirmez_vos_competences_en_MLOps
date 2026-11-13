"""Tests unitaires T01 à T07 et cohérence du contrat T10."""

import copy
import math

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.prediction import InputValidationError


@pytest.mark.parametrize("field", ["OWN_CAR_AGE", "CNT_CHILDREN", "BUREAU_CREDIT_SUM"])
def test_T01_missing_key_names_column_before_model(scorer, client_data, field):
    del client_data[field]
    with pytest.raises(InputValidationError, match=field):
        scorer.model.predict(client_data)
    scorer.pipeline.predict_proba.assert_not_called()


def test_T02_explicit_null_and_zero_reach_pipeline(scorer, client_data):
    assert client_data["OWN_CAR_AGE"] is None
    assert client_data["CNT_CHILDREN"] == 0
    scorer.model.predict(client_data)
    frame = scorer.pipeline.predict_proba.call_args.args[0]
    assert pd.isna(frame.at[0, "OWN_CAR_AGE"])
    assert frame.at[0, "CNT_CHILDREN"] == 0
    assert not pd.isna(frame.at[0, "CNT_CHILDREN"])


@pytest.mark.parametrize("field", ["AMT_INCOME_TOTAL", "CNT_CHILDREN"])
def test_T02_numeric_null_is_accepted(scorer, client_data, field):
    client_data[field] = None
    scorer.model.predict(client_data)
    frame = scorer.pipeline.predict_proba.call_args.args[0]
    assert pd.isna(frame.at[0, field])


def test_T02_boolean_null_is_rejected(scorer, client_data):
    client_data["OWN_CAR_AGE_ANOMALOUS"] = None
    with pytest.raises(InputValidationError, match="OWN_CAR_AGE_ANOMALOUS.*booléen"):
        scorer.model.predict(client_data)
    scorer.pipeline.predict_proba.assert_not_called()


@pytest.mark.parametrize("payload", [[], "client", None, {1: 0}])
def test_T03_invalid_structure(scorer, payload):
    with pytest.raises(InputValidationError):
        scorer.model.predict(payload)
    scorer.pipeline.predict_proba.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("TARGET", 0),
    ("AMT_INCOME_TOTAL", "180000.0"),
    ("AMT_INCOME_TOTAL", True),
    ("AMT_INCOME_TOTAL", float("nan")),
    ("AMT_INCOME_TOTAL", float("inf")),
    ("AMT_INCOME_TOTAL", float("-inf")),
    ("AMT_INCOME_TOTAL", 10 ** 400),
    ("CNT_CHILDREN", 1.5),
    ("CNT_CHILDREN", True),
    ("CNT_CHILDREN", 2 ** 63),
    ("CNT_CHILDREN", -(2 ** 63) - 1),
    ("OWN_CAR_AGE_ANOMALOUS", 0),
])
def test_T03_invalid_field_before_calculation(scorer, client_data, field, value):
    client_data[field] = value
    with pytest.raises(InputValidationError, match=field):
        scorer.model.predict(client_data)
    scorer.pipeline.predict_proba.assert_not_called()


@pytest.mark.parametrize("value", [2 ** 31, -(2 ** 31) - 1])
def test_T03_integer32_overflow(model_factory, schema_specs, client_data, value):
    specs = copy.deepcopy(schema_specs)
    next(spec for spec in specs if spec["name"] == "CNT_CHILDREN")["type"] = "integer"
    scorer = model_factory(specs=specs)
    client_data["CNT_CHILDREN"] = value
    with pytest.raises(InputValidationError, match="CNT_CHILDREN.*limites"):
        scorer.model.predict(client_data)
    scorer.pipeline.predict_proba.assert_not_called()


# Cas indépendants des ensembles de règles définis dans le code applicatif.
NON_NEGATIVE_CASES = [
    "OWN_CAR_AGE", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    "BUREAU_CREDIT_COUNT", "BUREAU_ACTIVE_CREDIT_COUNT",
    "BUREAU_OVERDUE_CREDIT_COUNT", "BUREAU_MAX_DAYS_OVERDUE",
    "BUREAU_PROLONGATION_SUM", "BUREAU_BB_HISTORY_CREDIT_COUNT",
    "BUREAU_CURRENCY_CURRENCY_1_COUNT",
    "BUREAU_CREDIT_TYPE_CONSUMER_CREDIT_COUNT",
    "PREV_APPLICATION_COUNT", "PREV_INSTAL_MAX_DAYS_LATE",
    "PREV_APPROVED_CNT_PAYMENT_MEAN", "PREV_POS_REMAINING_INSTALLMENTS_SUM",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "BUREAU_CREDIT_SUM", "BUREAU_OVERDUE_SUM",
    "PREV_APPROVED_ANNUITY_MEAN", "PREV_APPROVED_CREDIT_SUM",
    "BUREAU_PROLONGATION_PER_CREDIT",
    "PREV_APPROVED_CREDIT_TO_APPLICATION_RATIO_MEAN",
]


@pytest.mark.parametrize("field", NON_NEGATIVE_CASES)
def test_T04_non_negative_fields(scorer, client_data, schema_specs, field):
    spec = next(spec for spec in schema_specs if spec["name"] == field)
    is_integer = spec["type"] in {"integer", "long"}
    for value in (0 if is_integer else 0.0, 7 if is_integer else 7.0):
        client_data[field] = value
        scorer.model.predict(client_data)
        frame = scorer.pipeline.predict_proba.call_args.args[0]
        assert frame.at[0, field] == value
    scorer.pipeline.predict_proba.reset_mock()
    client_data[field] = -1 if is_integer else -1.0
    with pytest.raises(InputValidationError, match=f"{field}.*supérieure ou égale à 0"):
        scorer.model.predict(client_data)
    scorer.pipeline.predict_proba.assert_not_called()


@pytest.mark.parametrize("value,accepted", [(-12994, True), (-1, True), (0, False), (12994, False)])
def test_T04_birth_date(scorer, client_data, value, accepted):
    client_data["DAYS_BIRTH"] = value
    if accepted:
        scorer.model.predict(client_data)
    else:
        with pytest.raises(InputValidationError, match="DAYS_BIRTH.*< 0"):
            scorer.model.predict(client_data)
        scorer.pipeline.predict_proba.assert_not_called()


@pytest.mark.parametrize("value,accepted", [(0, True), (23, True), (-1, False), (24, False)])
def test_T04_application_hour(scorer, client_data, value, accepted):
    client_data["HOUR_APPR_PROCESS_START"] = value
    if accepted:
        scorer.model.predict(client_data)
    else:
        with pytest.raises(InputValidationError, match="HOUR_APPR_PROCESS_START.*0 et 23"):
            scorer.model.predict(client_data)
        scorer.pipeline.predict_proba.assert_not_called()


@pytest.mark.parametrize("field", [
    "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE", "BUREAU_DAYS_CREDIT_MAX", "PREV_DAYS_DECISION_MAX",
])
def test_T04_past_relative_dates(scorer, client_data, schema_specs, field):
    spec = next(spec for spec in schema_specs if spec["name"] == field)
    integer = spec["type"] == "long"
    for value in (-2055 if integer else -2055.0, 0 if integer else 0.0, None):
        client_data[field] = value
        scorer.model.predict(client_data)
    scorer.pipeline.predict_proba.reset_mock()
    client_data[field] = 1 if integer else 1.0
    with pytest.raises(InputValidationError, match=f"{field}.*<= 0"):
        scorer.model.predict(client_data)
    scorer.pipeline.predict_proba.assert_not_called()


def test_T04_employed_raw_sentinel_is_not_processed(scorer, client_data):
    client_data["DAYS_EMPLOYED"] = 365243.0
    with pytest.raises(InputValidationError, match="DAYS_EMPLOYED.*sentinelles"):
        scorer.model.predict(client_data)
    scorer.pipeline.predict_proba.assert_not_called()


@pytest.mark.parametrize("field", [
    "BUREAU_ACTIVE_CREDIT_RATIO", "BUREAU_BB_HISTORY_RATIO",
    "PREV_APPROVED_RATIO", "PREV_REFUSED_RATIO", "PREV_CASH_LOAN_RATIO",
    "PREV_CONSUMER_LOAN_RATIO", "PREV_REVOLVING_LOAN_RATIO",
    "PREV_INSURED_APPROVAL_RATIO", "PREV_HIGH_YIELD_APPROVAL_RATIO",
    "PREV_POS_HISTORY_RATIO", "PREV_INSTAL_HISTORY_RATIO",
    "PREV_INSTAL_LATE_PAYMENT_RATIO_MEAN", "PREV_INSTAL_UNDERPAYMENT_RATIO_MEAN",
    "PREV_CC_HISTORY_RATIO",
])
def test_T04_proportion_boundaries(scorer, client_data, field):
    for value in (0.0, 0.5, 1.0, None):
        client_data[field] = value
        scorer.model.predict(client_data)
    for value in (-0.1, 1.1):
        scorer.pipeline.predict_proba.reset_mock()
        client_data[field] = value
        with pytest.raises(InputValidationError, match=f"{field}.*0 et 1"):
            scorer.model.predict(client_data)
        scorer.pipeline.predict_proba.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("AMT_INCOME_TOTAL", 0.0),
    ("PREV_APPROVED_CREDIT_TO_APPLICATION_RATIO_MEAN", 1.0594774772),
    ("PREV_CC_BALANCE_LATEST_SUM", -100.0),
    ("BUREAU_DEBT_SUM", -100.0),
    ("PREV_CC_UTILIZATION_MEAN", -0.1),
    ("PREV_CC_UTILIZATION_MEAN", 1.2),
    ("BUREAU_DEBT_TO_CREDIT_RATIO", -0.1),
    ("PREV_INSTAL_PAYMENT_RATIO_MEAN", 1.2),
])
def test_T04_no_unestablished_bound(scorer, client_data, field, value):
    client_data[field] = value
    scorer.model.predict(client_data)
    frame = scorer.pipeline.predict_proba.call_args.args[0]
    assert frame.at[0, field] == value


def test_T05_key_order_does_not_change_model_input(scorer, client_data):
    scorer.model.predict(client_data)
    original = scorer.pipeline.predict_proba.call_args.args[0].copy()
    shuffled = dict(reversed(list(client_data.items())))
    scorer.model.predict(shuffled)
    reordered = scorer.pipeline.predict_proba.call_args.args[0]
    assert list(reordered.columns) == scorer.model.feature_names
    assert_frame_equal(original, reordered)


@pytest.mark.parametrize("classes", [(0, 1), (1, 0)])
@pytest.mark.parametrize("score,expected", [(0.49, 0), (0.5, 1), (0.51, 1)])
def test_T06_default_class_and_threshold(model_factory, client_data, classes, score, expected):
    scorer = model_factory(score=score, classes=classes)
    result = scorer.model.predict(client_data)
    assert result == {
        "score": score, "prediction": expected, "threshold": 0.5,
        "model_name": "credit_scoring_lightgbm", "model_version": "4",
    }


@pytest.mark.parametrize("score", [-0.1, 1.2, float("nan"), float("inf")])
def test_T07_invalid_model_score(model_factory, client_data, score):
    scorer = model_factory(score=score)
    with pytest.raises(RuntimeError, match="probabilité invalide"):
        scorer.model.predict(client_data)


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_T07_probability_boundaries(model_factory, client_data, score):
    result = model_factory(score=score).model.predict(client_data)
    assert math.isfinite(result["score"])
    assert result["score"] == score


def test_T10_schema_and_shared_model(scorer, client_data, schema_specs):
    schema = scorer.model.schema()
    assert schema["n_features"] == 310
    assert len({column["name"] for column in schema["columns"]}) == 310
    assert all(column["required"] is True for column in schema["columns"])
    assert [column["type"] for column in schema["columns"]] == [
        spec["type"] for spec in schema_specs
    ]
    scorer.model.predict(client_data)
    scorer.model.predict(client_data)
    scorer.loader.assert_called_once()
    assert scorer.pipeline.predict_proba.call_count == 2
