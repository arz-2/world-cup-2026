from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from world_cup_2026.core.schemas import BaselineFeatureArtifacts
from world_cup_2026.features.registry import build_baseline_training_frame


@dataclass(frozen=True)
class FoldMetrics:
    fold_index: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    n_train: int
    n_validation: int
    log_loss: float
    accuracy: float
    brier_score: float


def run_baseline_experiment(
    processed_dir: str | Path = "processed",
    output_dir: str | Path = "artifacts",
    include_elo: bool = False,
    include_fifa: bool = False,
) -> dict:
    artifacts = build_baseline_training_frame(
        processed_dir=processed_dir, 
        include_elo=include_elo,
        include_fifa=include_fifa
    )
    data = artifacts.feature_frame.copy()
    folds = _build_time_folds(data["match_date"])

    metrics: list[FoldMetrics] = []
    validation_predictions: list[pd.DataFrame] = []

    for fold_index, (train_mask, validation_mask) in enumerate(folds, start=1):
        train_full = data.loc[train_mask]
        validation_frame = data.loc[validation_mask]
        if train_full.empty or validation_frame.empty:
            continue

        # Temporal split for early stopping (last 10% of training data)
        split_idx = int(len(train_full) * 0.9)
        train_frame = train_full.iloc[:split_idx]
        val_es_frame = train_full.iloc[split_idx:]

        # Preprocessing
        preprocessor = _build_preprocessor(artifacts)
        X_train = preprocessor.fit_transform(train_frame[artifacts.feature_columns])
        X_val_es = preprocessor.transform(val_es_frame[artifacts.feature_columns])
        X_test = preprocessor.transform(validation_frame[artifacts.feature_columns])

        y_train = train_frame["result_code"]
        y_val_es = val_es_frame["result_code"]
        y_test = validation_frame["result_code"]

        # Training with early stopping
        classifier = _build_classifier()
        classifier.fit(
            X_train,
            y_train,
            eval_set=[(X_val_es, y_val_es)],
            verbose=False,
        )
        
        # Calibration (improves probability estimates for simulation)
        # Using FrozenEstimator (available in sklearn 1.8+) to calibrate a pre-fitted model
        calibrated = CalibratedClassifierCV(FrozenEstimator(classifier), method="sigmoid")
        calibrated.fit(X_val_es, y_val_es)
        
        predicted_probabilities = calibrated.predict_proba(X_test)
        predicted_labels = predicted_probabilities.argmax(axis=1)

        fold_metrics = FoldMetrics(
            fold_index=fold_index,
            train_start=train_frame["match_date"].min().date().isoformat(),
            train_end=train_frame["match_date"].max().date().isoformat(),
            validation_start=validation_frame["match_date"].min().date().isoformat(),
            validation_end=validation_frame["match_date"].max().date().isoformat(),
            n_train=len(train_frame),
            n_validation=len(validation_frame),
            log_loss=float(log_loss(y_test, predicted_probabilities, labels=[0, 1, 2])),
            accuracy=float(accuracy_score(y_test, predicted_labels)),
            brier_score=float(_multiclass_brier_score(y_test.to_numpy(), predicted_probabilities)),
        )
        metrics.append(fold_metrics)

        prediction_frame = validation_frame[["match_date", "home_team", "away_team", "result_code"]].copy()
        prediction_frame["fold_index"] = fold_index
        prediction_frame["pred_home_win"] = predicted_probabilities[:, 0]
        prediction_frame["pred_draw"] = predicted_probabilities[:, 1]
        prediction_frame["pred_away_win"] = predicted_probabilities[:, 2]
        prediction_frame["predicted_label"] = predicted_labels
        validation_predictions.append(prediction_frame)

    experiment_name = "baseline"
    if include_elo:
        experiment_name += "_elo"
    if include_fifa:
        experiment_name += "_fifa"
    experiment_name += "_xgboost"

    summary = {
        "experiment": experiment_name,
        "n_folds": len(metrics),
        "mean_log_loss": float(np.mean([metric.log_loss for metric in metrics])),
        "mean_accuracy": float(np.mean([metric.accuracy for metric in metrics])),
        "mean_brier_score": float(np.mean([metric.brier_score for metric in metrics])),
        "folds": [asdict(metric) for metric in metrics],
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / f"{experiment_name}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if validation_predictions:
        pd.concat(validation_predictions, ignore_index=True).to_csv(
            output_path / f"{experiment_name}_validation_predictions.csv",
            index=False,
        )
    return summary


def _build_preprocessor(artifacts: BaselineFeatureArtifacts) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), artifacts.numeric_columns),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                artifacts.categorical_columns,
            ),
        ]
    )


def _build_classifier() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=1000,
        learning_rate=0.0776,
        max_depth=8,
        subsample=0.61,
        colsample_bytree=0.59,
        min_child_weight=5,
        gamma=4.94,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=4,
    )


def _build_time_folds(match_dates: pd.Series) -> list[tuple[pd.Series, pd.Series]]:
    years = sorted(match_dates.dt.year.unique())
    validation_span_years = 4
    minimum_train_matches = 5_000
    minimum_validation_matches = 500
    folds: list[tuple[pd.Series, pd.Series]] = []

    for start_index in range(1, len(years) - validation_span_years + 1, validation_span_years):
        train_end_year = years[start_index - 1]
        validation_years = years[start_index : start_index + validation_span_years]
        train_mask = match_dates.dt.year <= train_end_year
        validation_mask = match_dates.dt.year.isin(validation_years)
        if int(train_mask.sum()) < minimum_train_matches:
            continue
        if int(validation_mask.sum()) < minimum_validation_matches:
            continue
        folds.append((train_mask, validation_mask))

    return folds


def _multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
