import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_lgb(
    X_train,
    y_train,
    X_val,
    y_val,
    output_path: str,
    params: dict = None,
):
    """Train a LightGBM regressor and save it to `output_path`.

    If LightGBM is not installed, raises ImportError.
    """
    try:
        import lightgbm as lgb
    except Exception as e:
        raise ImportError("lightgbm is required to train LightGBM models. Install with `pip install lightgbm`.") from e

    if params is None:
        params = {
            "n_estimators": 800,
            "max_depth": -1,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "regression",
            "random_state": 42,
            "n_jobs": -1,
        }

    # Sanitize params so XGBoost-style objective strings don't break LightGBM
    params = params.copy()
    obj = params.get("objective")
    if isinstance(obj, str) and obj.startswith("reg:"):
        params["objective"] = "regression"

    model = lgb.LGBMRegressor(**params)

    # Use callbacks for early stopping/logging to support different lightgbm versions
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
        )
    except Exception:
        # Fallback to a simple fit without callbacks if the API differs
        model.fit(X_train, y_train)

    preds = model.predict(X_val)

    # compute RMSE without relying on sklearn's 'squared' kwarg for compatibility
    rmse = mean_squared_error(y_val, preds) ** 0.5
    mae = mean_absolute_error(y_val, preds)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    joblib.dump(model, output_path)
    return model