import xgboost as xgb
import joblib
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

def train_xgb(
    X_train,
    y_train,
    X_val,
    y_val,
    output_path: str,
    params: dict = None,
):
    # Train an XGBoost regressor and save it to `output_path`
    if params is None:
        params = {
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }

    model = xgb.XGBRegressor(**params)

    # Support different xgboost versions: early_stopping may be passed directly or via callbacks
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=50
        )
    except TypeError:
        # newer xgboost versions use callbacks for early stopping
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[xgb.callback.EarlyStopping(rounds=50)],
                verbose=50
            )
        except Exception:
            # fallback to simple fit without early stopping
            model.fit(X_train, y_train, verbose=50)


    preds = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    joblib.dump(model, output_path)
    return model