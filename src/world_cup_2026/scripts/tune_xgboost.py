import optuna
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from world_cup_2026.features.baseline import build_baseline_training_frame
from world_cup_2026.training.baseline_xgboost import _build_preprocessor, _build_time_folds, _multiclass_brier_score

def objective(trial):
    # Load data
    artifacts = build_baseline_training_frame(include_elo=True, include_fifa=True)
    data = artifacts.feature_frame
    
    # Use only the most recent fold for faster tuning
    folds = _build_time_folds(data["match_date"])
    train_mask, validation_mask = folds[-1]
    
    train_full = data.loc[train_mask]
    validation_frame = data.loc[validation_mask]
    
    split_idx = int(len(train_full) * 0.9)
    train_frame = train_full.iloc[:split_idx]
    val_es_frame = train_full.iloc[split_idx:]
    
    preprocessor = _build_preprocessor(artifacts)
    X_train = preprocessor.fit_transform(train_frame[artifacts.feature_columns])
    X_val_es = preprocessor.transform(val_es_frame[artifacts.feature_columns])
    X_test = preprocessor.transform(validation_frame[artifacts.feature_columns])
    
    y_train = train_frame["result_code"]
    y_val_es = val_es_frame["result_code"]
    y_test = validation_frame["result_code"]
    
    # Hyperparameters
    param = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }
    
    model = XGBClassifier(
        **param,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val_es, y_val_es)], verbose=False)
    
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(X_val_es, y_val_es)
    
    probs = calibrated.predict_proba(X_test)
    loss = log_loss(y_test, probs, labels=[0, 1, 2])
    
    return loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    
    print("Best trials:")
    print(study.best_trial.params)
