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
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, learning_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "outputs/models"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

print("="*80)
print("INTEGRATED PIPELINE: FEATURE ENGINEERING + ENSEMBLE TRAINING")
print("="*80)


# Load engineered features
features_path = DATA_DIR / "processed/comprehensive_features.csv"
if features_path.exists():
    print("\n[1/3] Loading pre-computed features...")
    df_engineered = pd.read_csv(features_path)
else:
    from src.preprocessing import cleaning, create_station_id
    from src.comprehensive_features import create_full_feature_set
    print("\n[1/3] Computing comprehensive features...")
    raw_train = pd.read_csv(DATA_DIR / "water_quality_training_dataset.csv")
    raw_train['Sample Date'] = pd.to_datetime(raw_train['Sample Date'], format='mixed', dayfirst=True)
    raw_train = cleaning(raw_train)
    raw_train = create_station_id(raw_train)
    df_engineered = create_full_feature_set(raw_train, target='Dissolved Reactive Phosphorus')
    df_engineered.to_csv(features_path, index=False)

# Load PCA features from fetch_preprocess_external_features.py
pca_features_path = PROJECT_ROOT / "EY AI Data Challenge/scripts/samples_with_pca_features.csv"
if pca_features_path.exists():
    print("[1b/3] Loading external PCA features...")
    df_pca = pd.read_csv(pca_features_path)
    # Merge on sample_id if available, else concat
    if "sample_id" in df_pca.columns and "sample_id" in df_engineered.columns:
        df_engineered = df_engineered.merge(df_pca, on="sample_id", how="left", suffixes=("", "_pca"))
    else:
        # If no sample_id, just concat columns (assume same order)
        df_engineered = pd.concat([df_engineered, df_pca], axis=1)
    print(f"Features shape after PCA merge: {df_engineered.shape}")
else:
    print("[1b/3] PCA features not found, skipping external PCA merge.")

print(f"Features shape: {df_engineered.shape}")

# Temporal encoding for Sample Date
if 'Sample Date' in df_engineered.columns:
    df_engineered['Sample Date'] = pd.to_datetime(df_engineered['Sample Date'], errors='coerce')
    df_engineered['year'] = df_engineered['Sample Date'].dt.year
    df_engineered['month'] = df_engineered['Sample Date'].dt.month
    df_engineered['dayofyear'] = df_engineered['Sample Date'].dt.dayofyear
    df_engineered['sin_dayofyear'] = np.sin(2 * np.pi * df_engineered['dayofyear'] / 365)
    df_engineered['cos_dayofyear'] = np.cos(2 * np.pi * df_engineered['dayofyear'] / 365)

# K-Fold CV and Learning Curve for DRP/XGB/LGB
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("R² Score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.show()

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

# Time-based train/test split
if 'Sample Date' in df_train.columns:
    df_train = df_train.sort_values('Sample Date')
    split_idx = int(0.8 * len(df_train))
    X_train = df_train[X_cols][:split_idx]
    y_train = df_train[TARGETS][:split_idx]
    X_test = df_train[X_cols][split_idx:]
    y_test = df_train[TARGETS][split_idx:]
    print(f"  Time-based split: Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
else:
    # fallback to random split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df_train[X_cols], df_train[TARGETS], test_size=0.2, random_state=42
    )
    print(f"  Random split: Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# Scale features (fit on train portion only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train global XGBoost (multi-output)
from sklearn.multioutput import MultiOutputRegressor as _MOR
xgb_base = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=2.0, reg_lambda=4.0,  # Stronger regularization
    objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0
)
global_xgb = _MOR(xgb_base)
global_xgb.fit(X_train_scaled, y_train)
joblib.dump(global_xgb, MODELS_DIR / "xgb_global.pkl")

# K-Fold CV for XGB
kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv_scores = []
for train_idx, test_idx in kf.split(X_train):
    xgb_cv = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=4.0,
        objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_cv.fit(X_train.iloc[train_idx], y_train['Dissolved Reactive Phosphorus'].iloc[train_idx],
              verbose=0)
    score = xgb_cv.score(X_train.iloc[test_idx], y_train['Dissolved Reactive Phosphorus'].iloc[test_idx])
    xgb_cv_scores.append(score)
print(f"XGB DRP K-Fold CV R²: {np.mean(xgb_cv_scores):.4f}")
plot_learning_curve(xgb_base, X_train_scaled, y_train['Dissolved Reactive Phosphorus'], "XGB DRP Learning Curve")
# Train global LightGBM (multi-output)
import lightgbm as lgb
lgb_base = lgb.LGBMRegressor(
    n_estimators=300, max_depth=7, learning_rate=0.05,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=2.0, reg_lambda=4.0,  # Stronger regularization
    random_state=42, n_jobs=-1, verbosity=-1
)
global_lgb = _MOR(lgb_base)
global_lgb.fit(X_train_scaled, y_train)
joblib.dump(global_lgb, MODELS_DIR / "lgb_global.pkl")

# K-Fold CV for LGB
lgb_cv_scores = []
for train_idx, test_idx in kf.split(X_train):
    lgb_cv = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        num_leaves=15, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=4.0,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    lgb_cv.fit(X_train.iloc[train_idx], y_train['Dissolved Reactive Phosphorus'].iloc[train_idx],
              eval_set=[(X_train.iloc[test_idx], y_train['Dissolved Reactive Phosphorus'].iloc[test_idx])])
    score = lgb_cv.score(X_train.iloc[test_idx], y_train['Dissolved Reactive Phosphorus'].iloc[test_idx])
    lgb_cv_scores.append(score)
print(f"LGB DRP K-Fold CV R²: {np.mean(lgb_cv_scores):.4f}")
plot_learning_curve(lgb_base, X_train_scaled, y_train['Dissolved Reactive Phosphorus'], "LGB DRP Learning Curve")
# Train separate, simplified models specifically for DRP to improve generalization
drp_y_train = y_train['Dissolved Reactive Phosphorus']


# Feature selection for DRP: use feature importance from global XGB
xgb_importance = global_xgb.estimators_[2].feature_importances_  # Index 2 is DRP in multi-output
top_feature_indices = np.argsort(xgb_importance)[-100:]  # Top 100 features (was 50)
X_train_drp = X_train_scaled[:, top_feature_indices]
print(f"  DRP feature selection: {len(top_feature_indices)} features (from {X_train_scaled.shape[1]})")


# Tuned XGBoost for DRP (more features, relaxed regularization, more trees)
xgb_drp = xgb.XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.01,  # More trees, deeper
    subsample=0.7, colsample_bytree=0.7,  # Less dropout
    reg_alpha=4.0, reg_lambda=8.0,  # Relaxed regularization
    objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0
)
X_test_drp = X_test_scaled[:, top_feature_indices]
drp_y_test = y_test['Dissolved Reactive Phosphorus']
xgb_drp.fit(X_train_drp, drp_y_train, eval_set=[(X_test_drp, drp_y_test)])
joblib.dump({'model': xgb_drp, 'feature_indices': top_feature_indices},
             MODELS_DIR / "xgb_drp_model.pkl")


# Tuned LightGBM for DRP (more features, relaxed regularization, more trees)
lgb_drp = lgb.LGBMRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.01,  # More trees, deeper
    num_leaves=7, subsample=0.7, colsample_bytree=0.7,  # Less dropout
    reg_alpha=4.0, reg_lambda=8.0,  # Relaxed regularization
    random_state=42, n_jobs=-1, verbosity=-1
)
lgb_drp.fit(X_train_drp, drp_y_train, eval_set=[(X_test_drp, drp_y_test)])
joblib.dump({'model': lgb_drp, 'feature_indices': top_feature_indices},
             MODELS_DIR / "lgb_drp_model.pkl")

# Impute NaNs in DRP feature subset for QuantileRegressor
X_train_drp_qr = X_train_drp.copy()
if np.isnan(X_train_drp_qr).any():
    # Fill NaNs with column median
    medians = np.nanmedian(X_train_drp_qr, axis=0)
    inds = np.where(np.isnan(X_train_drp_qr))
    X_train_drp_qr[inds] = np.take(medians, inds[1])


# Quantile regressors for DRP with moderate regularization
qr_models = {}
for q in [0.25, 0.5, 0.75]:
    qr = QuantileRegressor(quantile=q, alpha=2.0, solver='highs')  # Relaxed alpha
    qr.fit(X_train_drp_qr, drp_y_train)
    qr_models[q] = qr
    joblib.dump({'model': qr, 'feature_indices': top_feature_indices}, 
                 MODELS_DIR / f"qr_drp_model_q{int(q*100)}.pkl")

# Evaluate on train set
pred_xgb_tr = pd.DataFrame(global_xgb.predict(X_train_scaled), columns=TARGETS)
pred_lgb_tr = pd.DataFrame(global_lgb.predict(X_train_scaled), columns=TARGETS)
pred_ens_tr = 0.5 * pred_xgb_tr + 0.5 * pred_lgb_tr

# DRP predictions use specialized regularized models
X_train_drp_subset = X_train_scaled[:, top_feature_indices]

# Impute NaNs in DRP feature subset for QuantileRegressor predictions (train)
X_train_drp_subset_qr = X_train_drp_subset.copy()
if np.isnan(X_train_drp_subset_qr).any():
    medians = np.nanmedian(X_train_drp_subset_qr, axis=0)
    inds = np.where(np.isnan(X_train_drp_subset_qr))
    X_train_drp_subset_qr[inds] = np.take(medians, inds[1])

pred_drp_xgb_tr = xgb_drp.predict(X_train_drp_subset)
pred_drp_lgb_tr = lgb_drp.predict(X_train_drp_subset)
pred_drp_qr_tr = qr_models[0.5].predict(X_train_drp_subset_qr)

pred_ens_tr['Dissolved Reactive Phosphorus'] = (
    0.45 * pred_drp_xgb_tr + 0.45 * pred_drp_lgb_tr + 0.10 * pred_drp_qr_tr
)

# Evaluate on test set
pred_xgb_test = pd.DataFrame(global_xgb.predict(X_test_scaled), columns=TARGETS)
pred_lgb_test = pd.DataFrame(global_lgb.predict(X_test_scaled), columns=TARGETS)
pred_ens_test = 0.5 * pred_xgb_test + 0.5 * pred_lgb_test

# DRP predictions use specialized regularized models
X_test_drp_subset = X_test_scaled[:, top_feature_indices]

# Impute NaNs in DRP feature subset for QuantileRegressor predictions (test)
X_test_drp_subset_qr = X_test_drp_subset.copy()
if np.isnan(X_test_drp_subset_qr).any():
    medians = np.nanmedian(X_test_drp_subset_qr, axis=0)
    inds = np.where(np.isnan(X_test_drp_subset_qr))
    X_test_drp_subset_qr[inds] = np.take(medians, inds[1])

pred_drp_xgb_test = xgb_drp.predict(X_test_drp_subset)
pred_drp_lgb_test = lgb_drp.predict(X_test_drp_subset)
pred_drp_qr_test = qr_models[0.5].predict(X_test_drp_subset_qr)

pred_ens_test['Dissolved Reactive Phosphorus'] = (
    0.45 * pred_drp_xgb_test + 0.45 * pred_drp_lgb_test + 0.10 * pred_drp_qr_test
)

results = {}
print("\n  Train Set Metrics:")
for t in TARGETS:
    r2_tr = r2_score(y_train[t], pred_ens_tr[t])
    mae_tr = mean_absolute_error(y_train[t], pred_ens_tr[t])
    rmse_tr = np.sqrt(mean_squared_error(y_train[t], pred_ens_tr[t]))
    print(f"    {t}: R²={r2_tr:.4f} | MAE={mae_tr:.4f} | RMSE={rmse_tr:.4f}")

print("\n  Test Set Metrics:")
for t in TARGETS:
    r2_test = r2_score(y_test[t], pred_ens_test[t])
    mae_test = mean_absolute_error(y_test[t], pred_ens_test[t])
    rmse_test = np.sqrt(mean_squared_error(y_test[t], pred_ens_test[t]))
    
    r2_tr = r2_score(y_train[t], pred_ens_tr[t])
    mae_tr = mean_absolute_error(y_train[t], pred_ens_tr[t])
    rmse_tr = np.sqrt(mean_squared_error(y_train[t], pred_ens_tr[t]))
    
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
    # Check for column mismatch
    mismatched = set(X_cols) - set(df_val_engineered.columns)
    if mismatched:
        print(f"  [Validation Feature Engineering] Mismatched columns: {sorted(mismatched)}")
except Exception as e:
    print(f"  [FAIL] Could not engineer features ({e}). Fallback to raw features is not allowed.")
    raise RuntimeError("Validation feature engineering failed. Fallback to raw features is not permitted.")


# Extract features matching training feature names
val_features = pd.DataFrame(index=df_val_engineered.index)
missing_cols = []
for col in X_cols:
    if col in df_val_engineered.columns:
        val_features[col] = df_val_engineered[col]
    else:
        val_features[col] = np.nan
        missing_cols.append(col)

# Print and warn about missing columns
if missing_cols:
    print(f"[Validation] {len(missing_cols)} missing columns will be imputed: {missing_cols}")
    if len(missing_cols) > 10:
        print(f"[Validation][WARNING] More than 10 features missing in validation! Model performance may degrade.")

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
# Impute NaNs in val_drp_subset for QuantileRegressor
val_drp_subset_qr = val_drp_subset.copy()
if np.isnan(val_drp_subset_qr).any():
    medians = np.nanmedian(val_drp_subset_qr, axis=0)
    inds = np.where(np.isnan(val_drp_subset_qr))
    val_drp_subset_qr[inds] = np.take(medians, inds[1])

pred_drp_xgb_val = xgb_drp.predict(val_drp_subset)
pred_drp_lgb_val = lgb_drp.predict(val_drp_subset)
pred_drp_qr_val = qr_models[0.5].predict(val_drp_subset_qr)

pred_val['Dissolved Reactive Phosphorus'] = (
    0.45 * pred_drp_xgb_val + 0.45 * pred_drp_lgb_val + 0.10 * pred_drp_qr_val
)
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
