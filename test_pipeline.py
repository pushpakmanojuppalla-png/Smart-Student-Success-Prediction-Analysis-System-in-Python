import pandas as pd
import pytest

from src import config
from src.data_generator import generate_students, score_to_level
from src.predict import DEMO_STUDENT, StudentPredictor
from src.preprocessing import build_preprocessor, split_data, validate_columns
from src.recommend import generate_recommendations
from src.train import run_training


@pytest.fixture(scope="module")
def df():
    return generate_students(n=400, seed=1)


@pytest.fixture(scope="module")
def trained(tmp_path_factory, df):
    out = tmp_path_factory.mktemp("artifacts")
    metrics = run_training(df, models_dir=out / "models",
                           reports_dir=out / "reports", fast=True)
    return out, metrics


def test_generated_data_shape_and_columns(df):
    assert len(df) == 400
    validate_columns(df)
    assert df[config.TARGET_SCORE].between(0, 100).all()
    assert set(df[config.TARGET_LEVEL].unique()) <= set(config.LEVELS)


def test_all_three_levels_present(df):
    assert set(df[config.TARGET_LEVEL].unique()) == set(config.LEVELS)


def test_score_to_level_thresholds():
    assert score_to_level(40) == "At-Risk"
    assert score_to_level(60) == "Average"
    assert score_to_level(90) == "High"


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        validate_columns(pd.DataFrame({"age": [20]}))


def test_preprocessor_handles_missing_values(df):
    out = build_preprocessor().fit_transform(df[config.FEATURES])
    assert not pd.isna(out.toarray() if hasattr(out, "toarray") else out).any()


def test_split_is_stratified(df):
    train, test = split_data(df)
    assert len(train) + len(test) == len(df)
    gap = abs((train[config.TARGET_LEVEL] == "At-Risk").mean()
              - (test[config.TARGET_LEVEL] == "At-Risk").mean())
    assert gap < 0.05


def test_models_beat_baselines(trained):
    _, metrics = trained
    assert metrics["classification"]["accuracy"] > 0.7
    assert metrics["regression"]["r2"] > 0.6


def test_single_prediction(trained):
    out, _ = trained
    res = StudentPredictor(out / "models").predict(DEMO_STUDENT)
    assert 0 <= res["predicted_score"] <= 100
    assert res["performance_level"] in config.LEVELS
    assert abs(sum(res["probabilities"].values()) - 1) < 0.01
    assert res["recommendations"]


def test_batch_prediction(trained, df):
    out, _ = trained
    preds = StudentPredictor(out / "models").predict_df(df.head(10))
    assert len(preds) == 10
    assert {"predicted_score", "predicted_level"} <= set(preds.columns)


def test_prediction_with_missing_inputs(trained):
    out, _ = trained
    res = StudentPredictor(out / "models").predict({"age": 21, "attendance_pct": 90})
    assert res["performance_level"] in config.LEVELS


def test_recommendations_for_struggling_student():
    recs = generate_recommendations(DEMO_STUDENT, "At-Risk")
    assert any("early-intervention" in r for r in recs)
    assert any("Attendance" in r for r in recs)


def test_missing_models_message(tmp_path):
    with pytest.raises(FileNotFoundError):
        StudentPredictor(tmp_path)
