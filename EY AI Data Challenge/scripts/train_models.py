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
from sklearn.multioutput import MultiOutputRegressor
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
print("\n[3/4] TRAINING CONSOLIDATED MODELS")
print("-" * 80)

# Build training dataset for multi-output models: require rows with all targets present
df_num = df_engineered.copy()
for t in TARGETS:
    if t not in df_num.columns:
        df_num[t] = np.nan

train_mask = df_num[TARGETS].notna().all(axis=1)
df_train = df_num.loc[train_mask].reset_index(drop=True)

# Features: numeric columns excluding target columns and identifiers
drop_cols = ['station_id', 'Sample Date', 'Latitude', 'Longitude']
X_cols = [c for c in df_train.select_dtypes(include=[np.number]).columns if c not in TARGETS]
X_train = df_train[X_cols]
y_train = df_train[TARGETS]

print(f"  Training samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")

# Fill missing feature values with median (robust)
X_train = X_train.copy()
medians = X_train.median()
# Drop columns where median is NaN (no valid values)
drop_nanmedian = medians[medians.isna()].index.tolist()
if drop_nanmedian:
    X_train = X_train.drop(columns=drop_nanmedian)
    X_cols = [c for c in X_cols if c not in drop_nanmedian]
    medians = medians.drop(index=drop_nanmedian)

# Fill remaining NaNs with median values
X_train = X_train.fillna(medians)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train global XGBoost (multi-output)
from sklearn.multioutput import MultiOutputRegressor as _MOR
xgb_base = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0
)
global_xgb = _MOR(xgb_base)
global_xgb.fit(X_train_scaled, y_train)
joblib.dump(global_xgb, MODELS_DIR / "xgb_global.pkl")

# Train global LightGBM (multi-output)
import lightgbm as lgb
lgb_base = lgb.LGBMRegressor(
    n_estimators=300, max_depth=7, learning_rate=0.05,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=-1
)
global_lgb = _MOR(lgb_base)
global_lgb.fit(X_train_scaled, y_train)
joblib.dump(global_lgb, MODELS_DIR / "lgb_global.pkl")

# Quantile regressors for Dissolved Reactive Phosphorus (DRP)
qr_models = {}
drp_y = y_train['Dissolved Reactive Phosphorus']
for q in [0.25, 0.5, 0.75]:
    qr = QuantileRegressor(quantile=q, alpha=0.01, solver='highs')
    qr.fit(X_train_scaled, drp_y)
    qr_models[q] = qr
    joblib.dump(qr, MODELS_DIR / f"qr_drp_model_q{int(q*100)}.pkl")

# Quick evaluation on train set
pred_xgb = pd.DataFrame(global_xgb.predict(X_train_scaled), columns=TARGETS)
pred_lgb = pd.DataFrame(global_lgb.predict(X_train_scaled), columns=TARGETS)
pred_ensemble = 0.5 * pred_xgb + 0.5 * pred_lgb
pred_ensemble['Dissolved Reactive Phosphorus'] = (
    0.4 * pred_xgb['Dissolved Reactive Phosphorus']
    + 0.35 * pred_lgb['Dissolved Reactive Phosphorus']
    + 0.25 * qr_models[0.5].predict(X_train_scaled)
)

results = {}
for t in TARGETS:
    r2 = r2_score(y_train[t], pred_ensemble[t])
    mae = mean_absolute_error(y_train[t], pred_ensemble[t])
    rmse = np.sqrt(mean_squared_error(y_train[t], pred_ensemble[t]))
    results[t] = {"r2_ensemble": r2, "mae_ensemble": mae, "rmse": rmse}
    print(f"  {t}: R²={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}")

# Save scaler and metadata
metadata = {
    "feature_names": X_cols,
    "scaler": scaler,
    "targets": TARGETS,
    "timestamp": pd.Timestamp.now().isoformat()
}
joblib.dump(metadata, MODELS_DIR / "pipeline_metadata.pkl")
print("\n[4/4] CREATING SUBMISSION")
print("-" * 80)

# Load submission template and align dates
template_path = DATA_DIR / "submission_template.csv"
df_sub = pd.read_csv(template_path)
df_sub['Sample Date'] = pd.to_datetime(df_sub['Sample Date'], dayfirst=True, errors='coerce')
df_engineered['Sample Date'] = pd.to_datetime(df_engineered['Sample Date'], errors='coerce')

# Merge engineered features into submission rows
df_merge = pd.merge(
    df_sub,
    df_engineered,
    on=['Latitude', 'Longitude', 'Sample Date'],
    how='left',
    suffixes=('', '_eng')
)

# Build test features using X_cols; fill missing with median from training
X_test = pd.DataFrame(index=df_merge.index)
for col in X_cols:
    if col in df_merge.columns:
        X_test[col] = df_merge[col]
    else:
        X_test[col] = np.nan

for col in X_test.columns:
    if X_test[col].isna().any():
        median_val = X_train[col].median() if col in X_train.columns else 0.0
        X_test[col] = X_test[col].fillna(median_val)

# Scale and predict
X_test_scaled = scaler.transform(X_test)
pred_xgb_test = pd.DataFrame(global_xgb.predict(X_test_scaled), columns=TARGETS)
pred_lgb_test = pd.DataFrame(global_lgb.predict(X_test_scaled), columns=TARGETS)

# Ensemble predictions
pred_test = 0.5 * pred_xgb_test + 0.5 * pred_lgb_test
# DRP ensemble includes QR median
pred_test['Dissolved Reactive Phosphorus'] = (
    0.4 * pred_xgb_test['Dissolved Reactive Phosphorus']
    + 0.35 * pred_lgb_test['Dissolved Reactive Phosphorus']
    + 0.25 * qr_models[0.5].predict(X_test_scaled)
)

# Attach predictions to submission frame in same order/cols as template
df_out = df_sub.copy()
for t in TARGETS:
    df_out[t] = pred_test[t].values

# Save submission as CSV
submission_path = SUBMISSION_DIR / "submission.csv"
df_out.to_csv(submission_path, index=False)
print(f"Saved submission: {submission_path}")

with open(SUBMISSION_DIR / "integrated_pipeline_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: integrated_pipeline_results.json")

print("\n" + "="*80)
print("✓ PIPELINE COMPLETE")
print("="*80)
for target, metrics in results.items():
    print(f"  {target}: R²={metrics['r2_ensemble']:.4f}, MAE={metrics['mae_ensemble']:.4f}")
