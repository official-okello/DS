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
    print("\n[1/3] Computing comprehensive features...")
    raw_train = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
    raw_train['Sample Date'] = pd.to_datetime(raw_train['Sample Date'], format='mixed', dayfirst=True)
    raw_train = cleaning(raw_train)
    raw_train = create_station_id(raw_train)
    df_engineered = create_full_feature_set(raw_train, target='Dissolved Reactive Phosphorus')
    df_engineered.to_csv(features_path, index=False)
else:
    print("\n[1/3] Loading pre-computed features...")
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
print("\n[2/3] TRAINING MODELS")
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

# Split training data into train/test for evaluation
from sklearn.model_selection import train_test_split
X_tr, X_test, y_tr, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"  Train samples: {X_tr.shape[0]} | Test samples: {X_test.shape[0]}")

# Scale features (fit on train portion only)
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_test_scaled = scaler.transform(X_test)

# Train global XGBoost (multi-output)
from sklearn.multioutput import MultiOutputRegressor as _MOR
xgb_base = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0
)
global_xgb = _MOR(xgb_base)
global_xgb.fit(X_tr_scaled, y_tr)
joblib.dump(global_xgb, MODELS_DIR / "xgb_global.pkl")

# Train global LightGBM (multi-output)
import lightgbm as lgb
lgb_base = lgb.LGBMRegressor(
    n_estimators=300, max_depth=7, learning_rate=0.05,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=-1
)
global_lgb = _MOR(lgb_base)
global_lgb.fit(X_tr_scaled, y_tr)
joblib.dump(global_lgb, MODELS_DIR / "lgb_global.pkl")

# Train separate, simplified models specifically for DRP to improve generalization
drp_y_tr = y_tr['Dissolved Reactive Phosphorus']

# Feature selection for DRP: use feature importance from global XGB
xgb_importance = global_xgb.estimators_[2].feature_importances_  # Index 2 is DRP in multi-output
top_feature_indices = np.argsort(xgb_importance)[-50:]  # Top 50 features
X_tr_drp = X_tr_scaled[:, top_feature_indices]

print(f"  DRP feature selection: {len(top_feature_indices)} features (from {X_tr_scaled.shape[1]})")

# Train simplified, regularized XGBoost for DRP
xgb_drp = xgb.XGBRegressor(
    n_estimators=150, max_depth=4, learning_rate=0.1,  # Reduced depth & increased lr
    subsample=0.7, colsample_bytree=0.7,  # Increased dropout
    reg_alpha=1.0, reg_lambda=2.0,  # L1/L2 regularization
    objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0
)
xgb_drp.fit(X_tr_drp, drp_y_tr)
joblib.dump({'model': xgb_drp, 'feature_indices': top_feature_indices}, 
             MODELS_DIR / "xgb_drp_model.pkl")

# Train simplified, regularized LightGBM for DRP
lgb_drp = lgb.LGBMRegressor(
    n_estimators=150, max_depth=4, learning_rate=0.1,  # Reduced depth & increased lr
    num_leaves=15, subsample=0.7, colsample_bytree=0.7,  # Reduced leaves, increased dropout
    reg_alpha=1.0, reg_lambda=2.0,  # L1/L2 regularization
    random_state=42, n_jobs=-1, verbosity=-1
)
lgb_drp.fit(X_tr_drp, drp_y_tr)
joblib.dump({'model': lgb_drp, 'feature_indices': top_feature_indices}, 
             MODELS_DIR / "lgb_drp_model.pkl")

# Quantile regressors for DRP with strong regularization
qr_models = {}
for q in [0.25, 0.5, 0.75]:
    qr = QuantileRegressor(quantile=q, alpha=5.0, solver='highs')  # Increased alpha for regularization
    qr.fit(X_tr_drp, drp_y_tr)
    qr_models[q] = qr
    joblib.dump({'model': qr, 'feature_indices': top_feature_indices}, 
                 MODELS_DIR / f"qr_drp_model_q{int(q*100)}.pkl")

# Evaluate on train set
pred_xgb_tr = pd.DataFrame(global_xgb.predict(X_tr_scaled), columns=TARGETS)
pred_lgb_tr = pd.DataFrame(global_lgb.predict(X_tr_scaled), columns=TARGETS)
pred_ens_tr = 0.5 * pred_xgb_tr + 0.5 * pred_lgb_tr

# DRP predictions use specialized regularized models
X_tr_drp_subset = X_tr_scaled[:, top_feature_indices]
pred_drp_xgb_tr = xgb_drp.predict(X_tr_drp_subset)
pred_drp_lgb_tr = lgb_drp.predict(X_tr_drp_subset)
pred_drp_qr_tr = qr_models[0.5].predict(X_tr_drp_subset)

pred_ens_tr['Dissolved Reactive Phosphorus'] = (
    0.45 * pred_drp_xgb_tr + 0.45 * pred_drp_lgb_tr + 0.10 * pred_drp_qr_tr
)

# Evaluate on test set
pred_xgb_test = pd.DataFrame(global_xgb.predict(X_test_scaled), columns=TARGETS)
pred_lgb_test = pd.DataFrame(global_lgb.predict(X_test_scaled), columns=TARGETS)
pred_ens_test = 0.5 * pred_xgb_test + 0.5 * pred_lgb_test

# DRP predictions use specialized regularized models
X_test_drp_subset = X_test_scaled[:, top_feature_indices]
pred_drp_xgb_test = xgb_drp.predict(X_test_drp_subset)
pred_drp_lgb_test = lgb_drp.predict(X_test_drp_subset)
pred_drp_qr_test = qr_models[0.5].predict(X_test_drp_subset)

pred_ens_test['Dissolved Reactive Phosphorus'] = (
    0.45 * pred_drp_xgb_test + 0.45 * pred_drp_lgb_test + 0.10 * pred_drp_qr_test
)

results = {}
print("\n  Train Set Metrics:")
for t in TARGETS:
    r2_tr = r2_score(y_tr[t], pred_ens_tr[t])
    mae_tr = mean_absolute_error(y_tr[t], pred_ens_tr[t])
    rmse_tr = np.sqrt(mean_squared_error(y_tr[t], pred_ens_tr[t]))
    print(f"    {t}: R²={r2_tr:.4f} | MAE={mae_tr:.4f} | RMSE={rmse_tr:.4f}")

print("\n  Test Set Metrics:")
for t in TARGETS:
    r2_test = r2_score(y_test[t], pred_ens_test[t])
    mae_test = mean_absolute_error(y_test[t], pred_ens_test[t])
    rmse_test = np.sqrt(mean_squared_error(y_test[t], pred_ens_test[t]))
    
    r2_tr = r2_score(y_tr[t], pred_ens_tr[t])
    mae_tr = mean_absolute_error(y_tr[t], pred_ens_tr[t])
    rmse_tr = np.sqrt(mean_squared_error(y_tr[t], pred_ens_tr[t]))
    
    results[t] = {
        "r2_train": r2_tr, "mae_train": mae_tr, "rmse_train": rmse_tr,
        "r2_test": r2_test, "mae_test": mae_test, "rmse_test": rmse_test
    }
    print(f"    {t}: R²={r2_test:.4f} | MAE={mae_test:.4f} | RMSE={rmse_test:.4f}")

# Save scaler and metadata
metadata = {
    "feature_names": X_cols,
    "scaler": scaler,
    "targets": TARGETS,
    "timestamp": pd.Timestamp.now().isoformat()
}
joblib.dump(metadata, MODELS_DIR / "pipeline_metadata.pkl")

print("\n[3/3] CREATING SUBMISSION")
print("-" * 80)

# Load validation template and raw validation features
template_path = DATA_DIR / "submission_template.csv"
df_sub = pd.read_csv(template_path)

# Load validation Landsat and TerraClimate features
landsat_val = pd.read_csv(DATA_DIR / "landsat_features_validation.csv")
terraclimate_val = pd.read_csv(DATA_DIR / "terraclimate_features_validation.csv")

# Build raw validation data (same structure as training)
val_raw = pd.DataFrame({
    'Latitude': landsat_val['Latitude'].values,
    'Longitude': landsat_val['Longitude'].values,
    'Sample Date': landsat_val['Sample Date'].values,
    'nir': landsat_val['nir'].values,
    'green': landsat_val['green'].values,
    'swir16': landsat_val['swir16'].values,
    'swir22': landsat_val['swir22'].values,
    'NDMI': landsat_val['NDMI'].values,
    'MNDWI': landsat_val['MNDWI'].values,
    'pet': terraclimate_val['pet'].values,
})

# Try to compute engineered features for validation data
try:
    # Import feature engineering functions
    from src.comprehensive_features import create_full_feature_set
    print("  Computing engineered features for validation data...")
    
    # Apply same feature engineering pipeline
    df_val_engineered = create_full_feature_set(val_raw, target='Dissolved Reactive Phosphorus')
except Exception as e:
    print(f"  Warning: Could not engineer features ({e}). Using raw features only.")
    df_val_engineered = val_raw

# Extract features matching training feature names
val_features = pd.DataFrame(index=df_val_engineered.index)
for col in X_cols:
    if col in df_val_engineered.columns:
        val_features[col] = df_val_engineered[col]
    else:
        # Fill missing engineered features with training median
        val_features[col] = np.nan

# Fill NaNs with training medians
for col in val_features.columns:
    if val_features[col].isna().any():
        median_val = X_train[col].median() if col in X_train.columns else 0.0
        val_features[col] = val_features[col].fillna(median_val)

# Scale validation features
val_scaled = scaler.transform(val_features)

# Generate predictions on validation data
pred_xgb_val = pd.DataFrame(global_xgb.predict(val_scaled), columns=TARGETS)
pred_lgb_val = pd.DataFrame(global_lgb.predict(val_scaled), columns=TARGETS)

# Ensemble predictions
pred_val = 0.5 * pred_xgb_val + 0.5 * pred_lgb_val

# DRP predictions use specialized regularized models
val_drp_subset = val_scaled[:, top_feature_indices]
pred_drp_xgb_val = xgb_drp.predict(val_drp_subset)
pred_drp_lgb_val = lgb_drp.predict(val_drp_subset)
pred_drp_qr_val = qr_models[0.5].predict(val_drp_subset)

pred_val['Dissolved Reactive Phosphorus'] = (
    0.45 * pred_drp_xgb_val + 0.45 * pred_drp_lgb_val + 0.10 * pred_drp_qr_val
)

# Build final submission with template structure
submission_df = pd.DataFrame({
    'Latitude': df_sub['Latitude'].values,
    'Longitude': df_sub['Longitude'].values,
    'Sample Date': df_sub['Sample Date'].values,
    'Total Alkalinity': pred_val['Total Alkalinity'].values,
    'Electrical Conductance': pred_val['Electrical Conductance'].values,
    'Dissolved Reactive Phosphorus': pred_val['Dissolved Reactive Phosphorus'].values
})

# Save submission as CSV
submission_path = SUBMISSION_DIR / "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission: {submission_path}")

with open(SUBMISSION_DIR / "integrated_pipeline_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: integrated_pipeline_results.json")

print("\n" + "="*80)
print("✓ PIPELINE COMPLETE")
print("="*80)
for target, metrics in results.items():
    print(f"  {target}: Train R²={metrics['r2_train']:.4f} | Test R²={metrics['r2_test']:.4f}")
