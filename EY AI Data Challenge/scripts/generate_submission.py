from pathlib import Path
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import (load_data, cleaning, add_time_features, create_station_id)
from src.features import add_lag_features, add_rolling_features, finalize_features
from src.train_xgb import train_xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "outputs/models"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

TARGETS = [
    'Total Alkalinity',
    'Electrical Conductance',
    'Dissolved Reactive Phosphorus'
]

# Config for quick training (kept small for speed)
QUICK_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}


def safe_name(name: str) -> str:
    return re.sub(r"\W+", "_", name)


def build_submission_features():
    # load submission template and validation features
    sub = pd.read_csv(DATA_DIR / "submission_template.csv")
    landsat_val = pd.read_csv(DATA_DIR / "landsat_features_validation.csv")
    terra_val = pd.read_csv(DATA_DIR / "terraclimate_features_validation.csv")

    # combine (column-wise), drop duplicate columns
    df = pd.concat([sub, landsat_val.drop(columns=["Latitude", "Longitude", "Sample Date"], errors='ignore'), terra_val.drop(columns=["Latitude", "Longitude", "Sample Date"], errors='ignore')], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    df = cleaning(df)
    df = add_time_features(df)
    # add cyclic encodings
    from src.features import add_cyclic_time_features
    df = add_cyclic_time_features(df)
    df = create_station_id(df)
    return df


def populate_history_features(train_df: pd.DataFrame, submit_df: pd.DataFrame, target: str, lags=(1,3,7,14,30), windows=(3,7,14,30), quantiles=(0.25, 0.5, 0.75)):
    """Populate lag and rolling features (mean, std, median, quantiles) for submission rows using history from train_df.

    For each station_id in submit_df, we look up historical target values in train_df and compute lag/rolling features.
    If insufficient history exists, values will be NaN and later filled with global/stat-based values.
    """
    # Ensure training df has station_id and is sorted by date
    train = train_df.copy()
    train = train.sort_values("Sample Date")

    # Pre-compute lists of historical values per station
    history = (
        train.groupby('station_id')[target]
        .apply(list)
        .to_dict()
    )

    out = submit_df.copy()

    for lag in lags:
        col = f"{target}_lag_{lag}"
        out[col] = np.nan

    for w in windows:
        out[f"{target}_roll_mean_{w}"] = np.nan
        out[f"{target}_roll_std_{w}"] = np.nan
        out[f"{target}_roll_med_{w}"] = np.nan
        for q in quantiles:
            q_name = int(q * 100)
            out[f"{target}_roll_q{q_name}_{w}"] = np.nan

    for idx, row in out.iterrows():
        sid = row['station_id']
        hist = history.get(sid, [])
        # lags: lag 1 is last obs, lag 3 is 3rd last, etc.
        for lag in lags:
            if len(hist) >= lag:
                out.at[idx, f"{target}_lag_{lag}"] = hist[-lag]
        for w in windows:
            if len(hist) >= w:
                window_vals = np.array(hist[-w:], dtype=float)
                out.at[idx, f"{target}_roll_mean_{w}"] = float(np.mean(window_vals))
                out.at[idx, f"{target}_roll_std_{w}"] = float(np.std(window_vals, ddof=0))
                out.at[idx, f"{target}_roll_med_{w}"] = float(np.median(window_vals))
                for q in quantiles:
                    q_name = int(q * 100)
                    out.at[idx, f"{target}_roll_q{q_name}_{w}"] = float(np.quantile(window_vals, q))

    # fill remaining NaNs with global median/zero to avoid missing features
    cols_to_fill = [f"{target}_lag_{l}" for l in lags] + [f"{target}_roll_mean_{w}" for w in windows] + [f"{target}_roll_std_{w}" for w in windows] + [f"{target}_roll_med_{w}" for w in windows] + [f"{target}_roll_q{int(q*100)}_{w}" for w in windows for q in quantiles]
    for col in cols_to_fill:
        if col in out.columns:
            if out[col].isna().all():
                out[col] = 0.0
            else:
                out[col] = out[col].fillna(out[col].median())

    return out


import argparse
from src.tune_xgb import tune_model as tune_xgb_model
from src.tune_lgb import tune_model as tune_lgb_model
from src.evaluate import evaluate
from src.train_lgb import train_lgb


def main():
    parser = argparse.ArgumentParser(description='Generate submission with optional full tuning')
    parser.add_argument('--full', action='store_true', help='Run full tuning before training (slower)')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials when --full is set')
    parser.add_argument('--ensemble', action='store_true', help='Train LightGBM and ensemble predictions (average of XGB+LGB)')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining of models even if saved models exist')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # load/build data
    raw_train = pd.read_csv(DATA_DIR / "raw/water_quality.csv")
    raw_train = cleaning(raw_train)
    raw_train = add_time_features(raw_train)
    if 'station_id' not in raw_train.columns:
        raw_train = create_station_id(raw_train)

    submit_df = build_submission_features()

    submission = submit_df[['Latitude','Longitude','Sample Date']].copy()

    r2_scores = {}

    for TARGET in TARGETS:
        xgb_model_path = MODELS_DIR / f"xgb_model_{safe_name(TARGET)}.pkl"
        lgb_model_path = MODELS_DIR / f"lgb_model_{safe_name(TARGET)}.pkl"
        print(f"Processing target: {TARGET}")

        # populate lag/rolling features for submission using training history
        submit_with_hist = populate_history_features(raw_train, submit_df, TARGET)

        # now prepare X for prediction
        # drop target columns if present
        X_submit = submit_with_hist.drop(columns=[c for c in submit_with_hist.columns if c in TARGETS], errors='ignore')

        # If the model exists load and predict, else train a model and save
        # Try to use existing saved XGBoost model if present, but validate features after feature engineering
        retrain_needed = False
        loaded_xgb_model = None
        if xgb_model_path.exists():
            print(f"Found existing model at {xgb_model_path} - will validate features before using.")
            loaded_xgb_model = joblib.load(xgb_model_path)
        else:
            retrain_needed = True
        # Force retrain for Dissolved Reactive Phosphorus when doing full tuning to ensure log-transform is applied
        if TARGET == 'Dissolved Reactive Phosphorus' and args.full:
            retrain_needed = True
        # Global force-retrain flag (override existing models)
        if args.force_retrain:
            print("--force-retrain set: forcing retrain for this target")
            retrain_needed = True

        # Prepare training features for evaluation/training
        df_train = raw_train.copy()
        other_targets = [t for t in TARGETS if t != TARGET]
        if other_targets:
            df_train = df_train.drop(columns=other_targets, errors='ignore')

        df_train = add_lag_features(df_train, TARGET)
        df_train = add_rolling_features(df_train, TARGET)
        X_train, y_train = finalize_features(df_train, TARGET, exclude_columns=other_targets)

        # Validate loaded model features against current training features and targets
        model = None
        if loaded_xgb_model is not None:
            model_features = getattr(loaded_xgb_model, 'feature_names_in_', None)
            if model_features is not None and any(t in model_features for t in other_targets):
                print("Existing model depends on other target columns; retraining to remove cross-target leakage.")
                retrain_needed = True
            if model_features is not None:
                missing = set(model_features) - set(X_train.columns)
                extra = set(X_train.columns) - set(model_features)
                if missing or extra:
                    print(f"Existing model features mismatch. Missing: {missing}; Extra (new features): {extra}. Will retrain.")
                    retrain_needed = True
            if not retrain_needed:
                print(f"Using existing model from {xgb_model_path}")
                model = loaded_xgb_model

        # Save groups (station_id) aligned to X_train index for group CV
        groups = None
        if 'station_id' in df_train.columns:
            groups = df_train.loc[X_train.index, 'station_id']

        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

        # Apply log1p transform for Dissolved Reactive Phosphorus
        transform = None
        inv_transform = None
        y_tr_orig = y_tr.copy()
        y_val_orig = y_val.copy()
        if TARGET == 'Dissolved Reactive Phosphorus':
            import numpy as np
            # transformation functions
            transform = np.log1p
            inv_transform = np.expm1
            y_tr = transform(y_tr)
            y_val = transform(y_val)

        if retrain_needed:
            if args.full:
                print(f"Running hyperparameter tuning for {TARGET} with {args.n_trials} trials...")
                # use group-aware tuning if groups available
                groups_tr = None
                if groups is not None:
                    groups_tr = groups.loc[X_tr.index]
                best_params = tune_xgb_model(X_tr, y_tr, X_val=X_val, y_val=y_val, groups=groups_tr, n_trials=args.n_trials)
                print(f"Tuned XGBoost params for {TARGET}: {best_params}")
                # train with tuned params
                xgb_model = train_xgb(X_tr, y_tr, X_val, y_val, output_path=str(xgb_model_path), params=best_params)
            else:
                print(f"Training quick model for {TARGET}...")
                xgb_model = train_xgb(X_tr, y_tr, X_val, y_val, output_path=str(xgb_model_path), params=QUICK_PARAMS)
        else:
            xgb_model = joblib.load(xgb_model_path)

        # Ensure uniform variable 'model' references the final XGBoost estimator
        model = xgb_model

        # Optionally train LightGBM and form an ensemble
        lgb_model = None
        if args.ensemble:
            retrain_lgb = True
            if lgb_model_path.exists() and not args.force_retrain:
                lgb_model = joblib.load(lgb_model_path)
                # check if LGB model depends on other targets
                mf = getattr(lgb_model, 'feature_name_', None) or getattr(lgb_model, 'feature_names_in_', None)
                if mf is not None and any(t in mf for t in [t for t in TARGETS if t != TARGET]):
                    print("Existing LGB model depends on other target columns; retraining LGB to remove leakage.")
                    retrain_lgb = True
                else:
                    retrain_lgb = False
            else:
                # if --force-retrain provided, always retrain LGB
                if args.force_retrain and lgb_model_path.exists():
                    print("--force-retrain set: forcing retrain of existing LightGBM model")
                retrain_lgb = True
            if retrain_lgb:
                if args.full:
                    print(f"Running LightGBM hyperparameter tuning for {TARGET} with {args.n_trials} trials...")
                    groups_tr = None
                    if groups is not None:
                        groups_tr = groups.loc[X_tr.index]
                    best_lgb_params = tune_lgb_model(X_tr, y_tr, groups=groups_tr, n_trials=args.n_trials)
                    print(f"Tuned LightGBM params for {TARGET}: {best_lgb_params}")
                    lgb_model = train_lgb(X_tr, y_tr, X_val, y_val, output_path=str(lgb_model_path), params=best_lgb_params)
                else:
                    lgb_model = train_lgb(X_tr, y_tr, X_val, y_val, output_path=str(lgb_model_path), params=QUICK_PARAMS)

        # Evaluate model on validation set and record R2
        # If we applied a transform, invert predictions before computing metrics
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        preds_val = model.predict(X_val)
        # invert transform for evaluation if applied
        if TARGET == 'Dissolved Reactive Phosphorus' and inv_transform is not None:
            import numpy as np
            # determine a safe upper bound in log-space based on training max (allow, e.g., 10x max)
            train_max = y_tr_orig.max() if 'y_tr_orig' in locals() else None
            if train_max is not None and np.isfinite(train_max):
                max_log = float(np.log1p(train_max * 10.0))
            else:
                max_log = 50.0
            preds_clipped = np.clip(preds_val, a_min=None, a_max=max_log)
            preds_val_orig = inv_transform(preds_clipped)
            # replace non-finite values and enforce non-negative targets
            preds_val_orig = np.where(np.isfinite(preds_val_orig), preds_val_orig, np.nan)
            preds_val_orig = np.nan_to_num(preds_val_orig, nan=0.0, posinf=1e6, neginf=0.0)
            preds_val_orig = np.maximum(0.0, preds_val_orig)
            y_eval = y_val_orig
            # warn if clipping occurred
            if np.any(preds_val > max_log):
                print(f"Warning: {np.sum(preds_val > max_log)} validation predictions clipped to exp({max_log:.2f}) to avoid overflow")
        else:
            preds_val_orig = preds_val
            y_eval = y_val_orig
        import numpy as np
        rmse = float(np.sqrt(mean_squared_error(y_eval, preds_val_orig)))
        mae = mean_absolute_error(y_eval, preds_val_orig)
        r2 = r2_score(y_eval, preds_val_orig)
        print(f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}")
        r2_scores[TARGET] = r2

        # Ensure feature columns in X_submit match model input
        model_features = getattr(model, 'feature_names_in_', None)
        if model_features is not None:
            missing = set(model_features) - set(X_submit.columns)
            if missing:
                print(f"Warning: missing features for model prediction (filling zeros): {missing}")
                for m in missing:
                    X_submit[m] = 0.0
            X_submit = X_submit[model_features]

        preds = model.predict(X_submit)
        # invert/clip predictions for transformed targets (Dissolved Reactive Phosphorus)
        if TARGET == 'Dissolved Reactive Phosphorus' and inv_transform is not None:
            import numpy as np
            max_log = 50.0
            preds = np.clip(preds, a_min=None, a_max=max_log)
            preds = inv_transform(preds)
            preds = np.where(np.isfinite(preds), preds, np.nan)
            preds = np.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=0.0)
            preds = np.maximum(0.0, preds)
            if np.any(preds > 1e6):
                print(f"Warning: some submission predictions exceeded 1e6 after inversion; clipped to 1e6")
                preds = np.clip(preds, 0.0, 1e6)
        submission[TARGET] = preds

    # Compute average R2
    avg_r2 = sum([v for v in r2_scores.values() if v is not None]) / len(r2_scores)
    print('\nPer-target R2 scores:')
    for k, v in r2_scores.items():
        print(f" - {k}: {v:.4f}")
    print(f"Average R2 across targets: {avg_r2:.4f}")

    # Save metrics
    import json
    metrics_out = {
        'per_target_r2': r2_scores,
        'average_r2': avg_r2
    }
    with open(SUBMISSION_DIR / 'metrics.json', 'w') as fh:
        json.dump(metrics_out, fh, indent=2)

    # Format Sample Date as dd-mm-yyyy
    if 'Sample Date' in submission.columns:
        submission['Sample Date'] = pd.to_datetime(submission['Sample Date'], errors='coerce').dt.strftime('%d-%m-%Y')

    out_path = SUBMISSION_DIR / 'submission.csv'
    submission.to_csv(out_path, index=False)
    print(f"Submission written to {out_path}")


if __name__ == '__main__':
    main()
