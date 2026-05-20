"""
Diagnostic pass: calibration curve + XGBoost feature importance on the 2021-2024 fold.

Run with:
    uv run python -m world_cup_2026.scripts.diagnostics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from world_cup_2026.features.registry import build_baseline_training_frame

PROCESSED_DIR = Path("processed")
OUTPUT_DIR = Path("artifacts")
# Target the fold whose validation window covers 2021-2024
TARGET_FOLD_START = 2021


def run_diagnostics() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = build_baseline_training_frame(
        processed_dir=PROCESSED_DIR,
        include_elo=True,
        include_fifa=True,
    )
    data = artifacts.feature_frame.copy()

    # Find the fold whose validation window starts at or near TARGET_FOLD_START
    years = sorted(data["match_date"].dt.year.unique())
    validation_span = 4
    target_fold = None

    for start_index in range(1, len(years) - validation_span + 1, validation_span):
        train_end_year = years[start_index - 1]
        validation_years = years[start_index : start_index + validation_span]
        if validation_years[0] >= TARGET_FOLD_START:
            target_fold = (train_end_year, validation_years)
            break

    if target_fold is None:
        # Fall back to last valid fold
        start_index = len(years) - validation_span
        target_fold = (years[start_index - 1], years[start_index:])

    train_end_year, validation_years = target_fold
    train_mask = data["match_date"].dt.year <= train_end_year
    val_mask = data["match_date"].dt.year.isin(validation_years)

    train_full = data.loc[train_mask]
    val_frame = data.loc[val_mask]

    split_idx = int(len(train_full) * 0.9)
    train_frame = train_full.iloc[:split_idx]
    val_es_frame = train_full.iloc[split_idx:]

    # Build preprocessor and fit
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), artifacts.numeric_columns),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                artifacts.categorical_columns,
            ),
        ]
    )

    X_train = preprocessor.fit_transform(train_frame[artifacts.feature_columns])
    X_val_es = preprocessor.transform(val_es_frame[artifacts.feature_columns])
    X_test = preprocessor.transform(val_frame[artifacts.feature_columns])

    y_train = train_frame["result_code"].to_numpy()
    y_val_es = val_es_frame["result_code"].to_numpy()
    y_test = val_frame["result_code"].to_numpy()

    clf = XGBClassifier(
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
    clf.fit(X_train, y_train, eval_set=[(X_val_es, y_val_es)], verbose=False)

    calibrated = CalibratedClassifierCV(FrozenEstimator(clf), method="sigmoid")
    calibrated.fit(X_val_es, y_val_es)

    probs = calibrated.predict_proba(X_test)

    # --- Calibration curves (one per class) ---
    class_names = ["home_win", "draw", "away_win"]
    calibration_results: dict[str, dict] = {}
    for class_idx, class_name in enumerate(class_names):
        y_binary = (y_test == class_idx).astype(int)
        fraction_pos, mean_pred = calibration_curve(y_binary, probs[:, class_idx], n_bins=10)
        calibration_results[class_name] = {
            "mean_predicted_prob": mean_pred.tolist(),
            "fraction_positive": fraction_pos.tolist(),
        }

    cal_path = OUTPUT_DIR / "diagnostics_calibration.json"
    cal_path.write_text(json.dumps(calibration_results, indent=2))
    print(f"Calibration saved → {cal_path}")

    _print_calibration_summary(calibration_results, class_names)

    # --- Feature importance (gain-based, from raw XGBoost) ---
    feature_names = _get_feature_names(preprocessor, artifacts)
    importances = clf.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": importances,
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    imp_path = OUTPUT_DIR / "diagnostics_feature_importance.csv"
    importance_df.to_csv(imp_path, index=False)
    print(f"\nFeature importance saved → {imp_path}")

    _print_importance_summary(importance_df)


def _get_feature_names(preprocessor: ColumnTransformer, artifacts) -> list[str]:
    numeric_names = artifacts.numeric_columns
    cat_encoder = preprocessor.named_transformers_["categorical"]["encoder"]
    cat_names = cat_encoder.get_feature_names_out(artifacts.categorical_columns).tolist()
    return numeric_names + cat_names


def _print_calibration_summary(results: dict, class_names: list[str]) -> None:
    print("\n=== Calibration Summary (mean |predicted - actual|) ===")
    for name in class_names:
        pred = np.array(results[name]["mean_predicted_prob"])
        actual = np.array(results[name]["fraction_positive"])
        mae = float(np.mean(np.abs(pred - actual)))
        print(f"  {name:<12}: mean absolute error = {mae:.4f}")
    print("  (closer to 0 = better calibrated; >0.05 = notable miscalibration)")


def _print_importance_summary(df: pd.DataFrame) -> None:
    total = df["importance_gain"].sum()
    df = df.copy()
    df["cumulative_pct"] = (df["importance_gain"].cumsum() / total * 100).round(1)
    df["pct"] = (df["importance_gain"] / total * 100).round(2)

    top20 = df.head(20)
    noise_floor = df[df["importance_gain"] < 0.001]

    print("\n=== Top 20 Features by Gain ===")
    for _, row in top20.iterrows():
        bar = "█" * int(row["pct"] / 2)
        print(f"  {row['feature']:<45} {row['pct']:>5.2f}%  {bar}")

    print(f"\n=== Near-Zero Features (gain < 0.001): {len(noise_floor)} features ===")
    if not noise_floor.empty:
        for _, row in noise_floor.head(30).iterrows():
            print(f"  {row['feature']}")
    else:
        print("  None found — no obvious candidates to prune.")

    print(f"\nTop 10 features account for {df.head(10)['pct'].sum():.1f}% of total gain.")
    print(f"Bottom half ({len(df) // 2} features) account for {df.tail(len(df) // 2)['pct'].sum():.1f}% of total gain.")


if __name__ == "__main__":
    run_diagnostics()
