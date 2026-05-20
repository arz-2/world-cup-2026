"""
Poisson lambda predictor: predicts expected goals (λ) for each team in a match.

The predictor reshapes each historical match into two team-perspective rows
(one for the home team's goals, one for the away team's goals), then fits a
PoissonRegressor on those rows. At prediction time it returns (λ_home, λ_away),
which feed directly into PoissonMatchModel.predict_result().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from world_cup_2026.core.schemas import BaselineFeatureArtifacts
from world_cup_2026.features.registry import build_baseline_training_frame
from world_cup_2026.models.poisson import PoissonMatchModel, fit_rho

# Features from the scoring team's attack history
# xg_for_* excluded: derived from score * 0.9 + elo_adj (target leakage)
_ATTACK_COLS = [
    "goals_for_last_10",
    "goals_for_ewm_10",
    "goals_for_career_avg",
    "goal_diff_ewm_10",
    "goal_diff_career_avg",
]

# Features from the opponent's defensive history
# xg_against_* excluded: derived from opponent score (target leakage)
_DEFENSE_COLS = [
    "goals_against_last_10",
    "goals_against_ewm_10",
    "goals_against_career_avg",
    "clean_sheet_last_10",
    "clean_sheet_ewm_5",
]

# Match-level context (same for both rows)
_CONTEXT_NUM_COLS = [
    "elo_advantage",   # scoring_team_elo - opponent_elo
    "is_home",
    "neutral_flag",
    "match_altitude",
    "travel_km_diff",  # opponent travel - scoring team travel (positive = opponent more fatigued)
    "rest_days_diff",  # scoring team rest - opponent rest
    "h2h_goal_diff_avg",
]
_CONTEXT_CAT_COLS = ["tournament_group"]

FEATURE_COLS = (
    [f"team_{c}" for c in _ATTACK_COLS]
    + [f"opp_{c}" for c in _DEFENSE_COLS]
    + _CONTEXT_NUM_COLS
    + _CONTEXT_CAT_COLS
)


def build_poisson_training_frame(artifacts: BaselineFeatureArtifacts) -> pd.DataFrame:
    """Reshape each match into two team-perspective rows with a goals target."""
    df = artifacts.feature_frame[artifacts.feature_frame["home_score"].notna()].copy()

    home_rows = _build_perspective(df, side="home")
    away_rows = _build_perspective(df, side="away")
    return pd.concat([home_rows, away_rows], ignore_index=True)


def _build_perspective(df: pd.DataFrame, side: str) -> pd.DataFrame:
    is_home = side == "home"
    opp = "away" if is_home else "home"

    rows = pd.DataFrame(index=df.index)
    rows["match_date"] = df["match_date"]
    rows["match_id"] = df["match_id"]
    rows["goals"] = df[f"{side}_score"].astype(float)

    for col in _ATTACK_COLS:
        rows[f"team_{col}"] = df[f"{side}_{col}"]
    for col in _DEFENSE_COLS:
        rows[f"opp_{col}"] = df[f"{opp}_{col}"]

    # elo_advantage: positive = scoring team is stronger
    elo_diff = df.get("elo_diff_pre", pd.Series(0.0, index=df.index))
    rows["elo_advantage"] = elo_diff if is_home else -elo_diff

    rows["is_home"] = int(is_home)
    rows["neutral_flag"] = df["neutral_flag"]
    rows["match_altitude"] = df.get("match_altitude", 0.0)

    # travel_km_diff: positive = opponent traveled more (scoring team fresher)
    home_travel = df.get("home_travel_km", 0.0)
    away_travel = df.get("away_travel_km", 0.0)
    if is_home:
        rows["travel_km_diff"] = away_travel - home_travel
    else:
        rows["travel_km_diff"] = home_travel - away_travel

    rows["rest_days_diff"] = df["rest_days_diff"] if is_home else -df["rest_days_diff"]
    rows["h2h_goal_diff_avg"] = df["h2h_goal_diff_avg"] if is_home else -df["h2h_goal_diff_avg"]
    rows["tournament_group"] = df["tournament_group"]

    return rows


@dataclass(frozen=True)
class PoissonFoldMetrics:
    fold_index: int
    validation_start: str
    validation_end: str
    n_validation_matches: int
    goals_mae: float        # mean absolute error on goals predicted
    log_loss: float         # log loss on derived win/draw/loss probabilities


class PoissonLambdaPredictor:
    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        num_cols = [c for c in FEATURE_COLS if c != "tournament_group"]
        cat_cols = ["tournament_group"]

        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ])

        self._pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(
                objective="count:poisson",
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=4,
                verbosity=0,
            )),
        ])
        self._pipeline.fit(X[FEATURE_COLS], y)

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        assert self._pipeline is not None, "Call fit() first."
        return self._pipeline.predict(X[FEATURE_COLS])

    def predict_lambdas(self, home_row: pd.DataFrame, away_row: pd.DataFrame) -> tuple[float, float]:
        """Return (λ_home, λ_away) for a single match."""
        λ_home = float(self.predict_lambda(home_row)[0])
        λ_away = float(self.predict_lambda(away_row)[0])
        return max(λ_home, 0.01), max(λ_away, 0.01)


def run_poisson_experiment(
    processed_dir: str | Path = "processed",
    output_dir: str | Path = "artifacts",
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts = build_baseline_training_frame(
        processed_dir=processed_dir,
        include_elo=True,
        include_fifa=True,
    )
    poisson_frame = build_poisson_training_frame(artifacts)
    match_frame = artifacts.feature_frame[artifacts.feature_frame["home_score"].notna()].copy()

    years = sorted(poisson_frame["match_date"].dt.year.unique())
    validation_span = 4
    minimum_train = 10_000   # two rows per match, so ~5k matches
    minimum_val_matches = 500

    fold_metrics: list[PoissonFoldMetrics] = []
    poisson_model = PoissonMatchModel()

    for start_index in range(1, len(years) - validation_span + 1, validation_span):
        train_end_year = years[start_index - 1]
        val_years = years[start_index: start_index + validation_span]

        train_mask = poisson_frame["match_date"].dt.year <= train_end_year
        val_mask = poisson_frame["match_date"].dt.year.isin(val_years)
        match_val_mask = match_frame["match_date"].dt.year.isin(val_years)

        train_df = poisson_frame.loc[train_mask]
        val_df = poisson_frame.loc[val_mask]
        val_matches = match_frame.loc[match_val_mask]

        if len(train_df) < minimum_train or len(val_matches) < minimum_val_matches:
            continue

        fold_index = len(fold_metrics) + 1

        predictor = PoissonLambdaPredictor()
        predictor.fit(train_df, train_df["goals"])

        # Fit rho on training data: predict λ for training matches, then optimise DC correction
        train_home = train_df[train_df["is_home"] == 1].set_index("match_id")
        train_away = train_df[train_df["is_home"] == 0].set_index("match_id")
        train_ids = train_home.index.intersection(train_away.index)
        λ_train_home = np.maximum(predictor.predict_lambda(train_home.loc[train_ids]), 0.01)
        λ_train_away = np.maximum(predictor.predict_lambda(train_away.loc[train_ids]), 0.01)
        rho = fit_rho(
            train_home.loc[train_ids, "goals"].to_numpy(),
            train_away.loc[train_ids, "goals"].to_numpy(),
            λ_train_home,
            λ_train_away,
        )
        poisson_model = PoissonMatchModel(rho=rho)

        # Predict λ for each side of validation matches
        home_val = val_df[val_df["is_home"] == 1].set_index("match_id")
        away_val = val_df[val_df["is_home"] == 0].set_index("match_id")

        common_ids = home_val.index.intersection(away_val.index)
        home_val = home_val.loc[common_ids]
        away_val = away_val.loc[common_ids]

        λ_home = np.maximum(predictor.predict_lambda(home_val), 0.01)
        λ_away = np.maximum(predictor.predict_lambda(away_val), 0.01)

        # MAE on goals
        actual_home = home_val["goals"].to_numpy()
        actual_away = away_val["goals"].to_numpy()
        goals_mae = float(np.mean(
            np.abs(np.concatenate([λ_home - actual_home, λ_away - actual_away]))
        ))

        # Log loss on derived match probabilities (using DC-corrected model)
        val_matches_indexed = val_matches.set_index("match_id").loc[common_ids]
        y_true = val_matches_indexed["result_code"].to_numpy()

        probs = [
            poisson_model.predict_result(lh, la).as_dict()
            for lh, la in zip(λ_home, λ_away)
        ]
        prob_matrix = np.array([
            [p["team_a_win"], p["draw"], p["team_b_win"]]
            for p in probs
        ])

        fold_log_loss = float(log_loss(y_true, prob_matrix, labels=[0, 1, 2]))

        fold_metrics.append(PoissonFoldMetrics(
            fold_index=fold_index,
            validation_start=val_matches_indexed["match_date"].min().date().isoformat(),
            validation_end=val_matches_indexed["match_date"].max().date().isoformat(),
            n_validation_matches=len(common_ids),
            goals_mae=goals_mae,
            log_loss=fold_log_loss,
        ))

        print(
            f"Fold {fold_index:>2} | val {val_years[0]}-{val_years[-1]} "
            f"| goals MAE {goals_mae:.3f} | log loss {fold_log_loss:.4f} | rho {rho:.3f}"
        )

    summary = {
        "experiment": "poisson_lambda",
        "n_folds": len(fold_metrics),
        "mean_goals_mae": float(np.mean([m.goals_mae for m in fold_metrics])),
        "mean_log_loss": float(np.mean([m.log_loss for m in fold_metrics])),
        "folds": [asdict(m) for m in fold_metrics],
    }

    out_path = output_path / "poisson_lambda_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved → {out_path}")
    print(f"Mean goals MAE : {summary['mean_goals_mae']:.3f}")
    print(f"Mean log loss  : {summary['mean_log_loss']:.4f}  (XGBoost baseline: 0.9027)")
    return summary
