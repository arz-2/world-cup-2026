import numpy as np
import pandas as pd
import pytest

from world_cup_2026.models.poisson_lambda import (
    PoissonLambdaPredictor,
    build_poisson_training_frame,
    FEATURE_COLS,
)
from world_cup_2026.models.poisson import PoissonMatchModel


def _make_minimal_artifacts():
    """Build a tiny synthetic feature frame to test the reshaper and predictor."""
    from world_cup_2026.core.schemas import BaselineFeatureArtifacts

    n = 40
    rng = np.random.default_rng(0)

    data = {
        "match_id": np.arange(n),
        "match_date": pd.date_range("2010-01-01", periods=n, freq="7D"),
        "home_score": rng.integers(0, 5, n).astype(float),
        "away_score": rng.integers(0, 4, n).astype(float),
        "result_code": rng.integers(0, 3, n),
        "neutral_flag": np.zeros(n),
        "host_advantage_flag": np.zeros(n),
        "rest_days_diff": rng.normal(0, 3, n),
        "h2h_matches": np.ones(n),
        "h2h_home_points_avg": np.ones(n),
        "h2h_goal_diff_avg": np.zeros(n),
        "h2h_days_since_last": np.full(n, 365.0),
        "match_altitude": np.zeros(n),
        "home_travel_km": np.zeros(n),
        "away_travel_km": rng.uniform(0, 5000, n),
        "travel_diff_km": rng.normal(0, 1000, n),
        "elo_diff_pre": rng.normal(0, 100, n),
        "tournament_group": ["Friendly"] * n,
    }
    # Add all required home/away rolling feature columns with random values
    rolling_bases = [
        "goals_for_last_10", "goals_for_ewm_10", "goals_for_career_avg",
        "xg_for_last_5", "xg_for_ewm_5", "xg_for_career_avg",
        "goal_diff_ewm_10", "goal_diff_career_avg",
        "goals_against_last_10", "goals_against_ewm_10", "goals_against_career_avg",
        "clean_sheet_last_10", "clean_sheet_ewm_5",
        "xg_against_last_5", "xg_against_ewm_5", "xg_against_career_avg",
    ]
    for side in ("home", "away"):
        for col in rolling_bases:
            data[f"{side}_{col}"] = rng.uniform(0, 2, n)

    df = pd.DataFrame(data)

    return BaselineFeatureArtifacts(
        feature_frame=df,
        feature_columns=[],
        categorical_columns=[],
        numeric_columns=[],
    )


def test_build_poisson_training_frame_doubles_rows():
    artifacts = _make_minimal_artifacts()
    poisson_df = build_poisson_training_frame(artifacts)
    assert len(poisson_df) == len(artifacts.feature_frame) * 2


def test_build_poisson_training_frame_has_required_columns():
    artifacts = _make_minimal_artifacts()
    poisson_df = build_poisson_training_frame(artifacts)
    for col in FEATURE_COLS:
        assert col in poisson_df.columns, f"Missing column: {col}"


def test_poisson_lambda_predictor_fit_predict():
    artifacts = _make_minimal_artifacts()
    poisson_df = build_poisson_training_frame(artifacts)

    predictor = PoissonLambdaPredictor()
    predictor.fit(poisson_df, poisson_df["goals"])

    preds = predictor.predict_lambda(poisson_df)
    assert len(preds) == len(poisson_df)
    assert (preds > 0).all(), "All predicted lambdas must be positive"


def test_predict_lambdas_returns_two_positive_floats():
    artifacts = _make_minimal_artifacts()
    poisson_df = build_poisson_training_frame(artifacts)

    predictor = PoissonLambdaPredictor()
    predictor.fit(poisson_df, poisson_df["goals"])

    home_row = poisson_df[poisson_df["is_home"] == 1].head(1)
    away_row = poisson_df[poisson_df["is_home"] == 0].head(1)
    λ_home, λ_away = predictor.predict_lambdas(home_row, away_row)

    assert isinstance(λ_home, float)
    assert isinstance(λ_away, float)
    assert λ_home > 0
    assert λ_away > 0


def test_lambda_into_poisson_model_sums_to_one():
    artifacts = _make_minimal_artifacts()
    poisson_df = build_poisson_training_frame(artifacts)

    predictor = PoissonLambdaPredictor()
    predictor.fit(poisson_df, poisson_df["goals"])

    home_row = poisson_df[poisson_df["is_home"] == 1].head(1)
    away_row = poisson_df[poisson_df["is_home"] == 0].head(1)
    λ_home, λ_away = predictor.predict_lambdas(home_row, away_row)

    model = PoissonMatchModel()
    prediction = model.predict_result(λ_home, λ_away)
    total = prediction.team_a_win + prediction.draw + prediction.team_b_win

    assert round(total, 6) == 1.0
