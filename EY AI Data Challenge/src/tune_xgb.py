import optuna
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold, cross_val_score
import numpy as np


def objective_cv(trial, X, y, groups, n_splits=5):
    """Optuna objective using GroupKFold cross-validation and R2 as metric (maximize R2 -> minimize 1 - R2)."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = xgb.XGBRegressor(**params)
    cv = GroupKFold(n_splits=n_splits)

    scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring='r2', n_jobs=1)
    # Minimize 1 - mean(R2)
    return 1.0 - float(np.mean(scores))


def objective_holdout(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1500),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1
    }

    model = xgb.XGBRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )

    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)

    return rmse


def tune_model(X_train, y_train, X_val=None, y_val=None, groups=None, n_trials=50, use_group_cv=True, n_splits=5):
    """Tune hyperparameters.

    If `groups` is provided and `use_group_cv=True`, perform GroupKFold CV optimizing for mean R2.
    Otherwise, fall back to holdout-based tuning minimizing RMSE on (X_val, y_val).
    """
    study = optuna.create_study(direction="minimize")
    if groups is not None and use_group_cv:
        print("Using GroupKFold CV for tuning (optimizing R2)")
        study.optimize(lambda trial: objective_cv(trial, X_train, y_train, groups, n_splits=n_splits), n_trials=n_trials)
    else:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must be provided for holdout tuning")
        study.optimize(lambda trial: objective_holdout(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

    print("Best objective:", study.best_value)
    print("Best Params:", study.best_params)

    return study.best_params