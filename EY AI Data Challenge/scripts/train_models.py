#!/usr/bin/env python
"""Train ensemble models with comprehensive features."""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import re
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import QuantileRegressor
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "outputs/models"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

print("="*80)
print("INTEGRATED PIPELINE: FEATURE ENGINEERING + ENSEMBLE TRAINING")
print("="*80)

# Load features
features_path = DATA_DIR / "processed/comprehensive_features.csv"
if not features_path.exists():
    from src.preprocessing import cleaning, create_station_id
    from src.comprehensive_features import create_full_feature_set
    print("\n[1/4] Computing comprehensive features...")
    raw_train = pd.read_csv(DATA_DIR / "raw/water_quality.csv")
    raw_train['Sample Date'] = pd.to_datetime(raw_train['Sample Date'], format='mixed', dayfirst=True)
    raw_train = cleaning(raw_train)
    raw_train = create_station_id(raw_train)
    df_engineered = create_full_feature_set(raw_train, target='Dissolved Reactive Phosphorus')
    df_engineered.to_csv(features_path, index=False)
else:
    print("\n[1/4] Loading pre-computed features...")
    df_engineered = pd.read_csv(features_path)

print(f"Features shape: {df_engineered.shape}")

# Feature selection function
def prepare_features_for_target(df, target):
    """Prepare features for target (remove other targets, fill NaN, scale)."""
    drop_cols = ['station_id', 'Sample Date', 'Latitude', 'Longitude']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    for t in TARGETS:
        if t != target and t in X.columns:
            X = X.drop(columns=[t])
    
    if target in X.columns:
        X = X.drop(columns=[target])
    
    X = X.select_dtypes(include=[np.number])
    X = X.dropna(axis=1, how='all')
    
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    y = df.loc[X.index, target]
    return X, y

# Train models
print("\n[3/4] TRAINING MODELS FOR EACH TARGET")
print("-" * 80)

xgb_models = {}
lgb_models = {}
qr_models = {}
feature_names_per_target = {}
scalers_per_target = {}
results = {}

for TARGET in TARGETS:
    print(f"\n{TARGET}:")
    safe_target_name = re.sub(r"\W+", "_", TARGET)
    
    X, y = prepare_features_for_target(df_engineered, TARGET)
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    feature_names_per_target[TARGET] = X.columns.tolist()
    
    print(f"  Features: {X.shape[1]} | Samples: {X.shape[0]}")
    print(f"  Target: mean={y.mean():.2f}, std={y.std():.2f}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scalers_per_target[TARGET] = scaler
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_scaled, y)
    xgb_models[TARGET] = xgb_model
    
    y_pred_xgb = xgb_model.predict(X_scaled)
    r2_xgb = r2_score(y, y_pred_xgb)
    mae_xgb = mean_absolute_error(y, y_pred_xgb)
    print(f"  XGBoost R²: {r2_xgb:.4f} | MAE: {mae_xgb:.4f}")
    
    joblib.dump(xgb_model, MODELS_DIR / f"xgb_full_model_{safe_target_name}.pkl")
    
    # LightGBM
    import lightgbm as lgb
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    lgb_model.fit(X_scaled, y)
    lgb_models[TARGET] = lgb_model
    
    y_pred_lgb = lgb_model.predict(X_scaled)
    r2_lgb = r2_score(y, y_pred_lgb)
    mae_lgb = mean_absolute_error(y, y_pred_lgb)
    print(f"  LightGBM R²: {r2_lgb:.4f} | MAE: {mae_lgb:.4f}")
    
    joblib.dump(lgb_model, MODELS_DIR / f"lgb_full_model_{safe_target_name}.pkl")
    
    # Quantile Regression for DRP
    if TARGET == 'Dissolved Reactive Phosphorus':
        qr_models[TARGET] = {}
        for q in [0.25, 0.5, 0.75]:
            qr = QuantileRegressor(quantile=q, alpha=0.01, solver='highs')
            qr.fit(X_scaled, y)
            qr_models[TARGET][q] = qr
            
            y_pred_qr = qr.predict(X_scaled)
            r2_qr = r2_score(y, y_pred_qr)
            print(f"  Quantile {q}: R²={r2_qr:.4f}")
            
            joblib.dump(qr, MODELS_DIR / f"qr_full_model_q{int(q*100)}_{safe_target_name}.pkl")
        
        y_pred_ensemble = 0.4 * y_pred_xgb + 0.35 * y_pred_lgb + 0.25 * qr_models[TARGET][0.5].predict(X_scaled)
    else:
        y_pred_ensemble = 0.5 * y_pred_xgb + 0.5 * y_pred_lgb
    
    r2_ensemble = r2_score(y, y_pred_ensemble)
    mae_ensemble = mean_absolute_error(y, y_pred_ensemble)
    rmse_ensemble = np.sqrt(mean_squared_error(y, y_pred_ensemble))
    
    print(f"  Ensemble R²: {r2_ensemble:.4f} | MAE: {mae_ensemble:.4f} | RMSE: {rmse_ensemble:.4f}")
    
    results[TARGET] = {
        "r2_ensemble": r2_ensemble,
        "mae_ensemble": mae_ensemble,
        "rmse_ensemble": rmse_ensemble
    }

# Save metadata
print("\n[4/4] SAVING MODELS AND METADATA")
print("-" * 80)

metadata = {
    "feature_names": feature_names_per_target,
    "scalers": scalers_per_target,
    "targets": TARGETS,
    "timestamp": pd.Timestamp.now().isoformat()
}
joblib.dump(metadata, MODELS_DIR / "pipeline_metadata.pkl")
print(f"Saved: pipeline_metadata.pkl")

# Save results
with open(SUBMISSION_DIR / "integrated_pipeline_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: integrated_pipeline_results.json")

print("\n" + "="*80)
print("✓ PIPELINE COMPLETE")
print("="*80)
for target, metrics in results.items():
    print(f"  {target}: R²={metrics['r2_ensemble']:.4f}, MAE={metrics['mae_ensemble']:.4f}")
