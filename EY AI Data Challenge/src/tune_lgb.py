import optuna
from sklearn.model_selection import GroupKFold, cross_val_score
import numpy as np


def objective_cv(trial, X, y, groups, n_splits=5):
    import lightgbm as lgb
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "objective": "regression",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = lgb.LGBMRegressor(**params)
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring='r2', n_jobs=1)
    return 1.0 - float(np.mean(scores))


def tune_model(X_train, y_train, X_val=None, y_val=None, groups=None, n_trials=50, use_group_cv=True, n_splits=5):
    study = optuna.create_study(direction="minimize")
    if groups is not None and use_group_cv:
        print("Using GroupKFold CV for LightGBM tuning (optimizing R2)")
        study.optimize(lambda trial: objective_cv(trial, X_train, y_train, groups, n_splits=n_splits), n_trials=n_trials)
    else:
        # fallback to simple holdout using default params
        study.optimize(lambda trial: objective_cv(trial, X_train, y_train, np.repeat(0, len(y_train)), n_splits=n_splits), n_trials=n_trials)

    print("Best objective:", study.best_value)
    print("Best Params:", study.best_params)

    return study.best_params