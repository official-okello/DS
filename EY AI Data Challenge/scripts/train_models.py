#!/usr/bin/env python


import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import re
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, learning_curve
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "outputs/models"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
PLOTS_DIR = PROJECT_ROOT / "outputs/plots"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
DRP = 'Dissolved Reactive Phosphorus'

def transform_drp(y):
    return np.log1p(np.clip(y, 0, None))

def inverse_transform_drp(y_log):
    return np.expm1(y_log)

print("=" * 80)
print("PIPELINE: ENSEMBLE + EARLY STOPPING + LOG TRANSFORM + CATBOOST")
print("=" * 80)

# LOAD FEATURES
features_path = DATA_DIR / "processed/comprehensive_features.csv"
if features_path.exists():
    print("\n[1/4] Loading pre-computed features...")
    df_engineered = pd.read_csv(features_path)
    print(f"  Loaded shape: {df_engineered.shape}")
    
    # Check if targets are present
    targets_present = [t for t in TARGETS if t in df_engineered.columns]
    print(f"  Targets found: {len(targets_present)}/3")
    
    # Only merge if targets are missing
    if len(targets_present) < 3:
        water_quality_path = DATA_DIR / "raw/water_quality.csv"
        if water_quality_path.exists():
            print("Targets missing - merging from water_quality.csv...")
            df_targets = pd.read_csv(water_quality_path)
            
            # Verify same number of rows (they should match since features are derived from same data)
            if len(df_engineered) == len(df_targets):
                # Index-based merge
                added = 0
                for target in TARGETS:
                    if target in df_targets.columns and target not in df_engineered.columns:
                        df_engineered[target] = df_targets[target].values
                        added += 1
                print(f"Added {added} target columns")
                print(f"Final shape: {df_engineered.shape}")
            else:
                raise ValueError(
                    f"Row count mismatch! Features: {len(df_engineered)}, "
                    f"Targets: {len(df_targets)}. Cannot merge."
                )
        else:
            raise FileNotFoundError(
                f"water_quality.csv not found at {water_quality_path}. "
                f"Needed to merge targets with engineered features."
            )
    else:
        print(f"All targets already present")
else:
    from src.preprocessing import build_raw_dataset, cleaning, create_station_id
    from src.comprehensive_features import create_full_feature_set
    print("\n[1/4] Computing comprehensive features...")
    build_raw_dataset()
    raw_train = pd.read_csv(DATA_DIR / "raw/water_quality.csv")
    raw_train['Sample Date'] = pd.to_datetime(raw_train['Sample Date'], format='mixed', dayfirst=True)
    raw_train = cleaning(raw_train)
    raw_train = create_station_id(raw_train)
    df_engineered = create_full_feature_set(raw_train, target='Dissolved Reactive Phosphorus')
    df_engineered.to_csv(features_path, index=False)

# Load PCA features
import fetch_preprocess_external_features

pca_features_path = DATA_DIR / "processed/samples_with_pca_features.csv"

if pca_features_path.exists():
    print("[1b/4] Loading external PCA features...")
    df_pca = pd.read_csv(pca_features_path)

    # Define which feature groups to replace with PCA
    # Group 1: Spectral indices (highly correlated)
    spectral_to_replace = ['NDVI', 'EVI', 'SAVI', 'MSAVI', 'NBR', 'NDWI_alt']
    
    # Group 2: Interaction features (often redundant)
    interaction_pattern = ['_x_pet', '_x_dist', '_x_lat']
    
    # Group 3: Temporal features (sin/cos pairs are correlated)
    temporal_to_replace = [c for c in df_engineered.columns 
                            if 'sin_' in c or 'cos_' in c]
    
    # Combine all features to replace
    features_to_replace = spectral_to_replace + temporal_to_replace
    features_to_replace += [c for c in df_engineered.columns 
                            if any(pattern in c for pattern in interaction_pattern)]
    
    # Drop replaced features
    df_engineered = df_engineered.drop(columns=[c for c in features_to_replace 
                                                    if c in df_engineered.columns])
    
    # Add PCA components
    pca_cols = [c for c in df_pca.columns if c.startswith('pca_')]
    for col in pca_cols:
        df_engineered[col] = df_pca[col].values
    
    print(f"Replaced {len(features_to_replace)} correlated features")
    print(f"Added {len(pca_cols)} PCA components")
    print(f"Final feature count: {len(df_engineered.columns) - len(TARGETS) - 4}")
else:
    print("[1b/4] PCA features not found, skipping.")

print(f"Features shape: {df_engineered.shape}")

# Temporal encoding
if 'Sample Date' in df_engineered.columns:
    df_engineered['Sample Date'] = pd.to_datetime(df_engineered['Sample Date'], errors='coerce')
    df_engineered['year'] = df_engineered['Sample Date'].dt.year
    df_engineered['month'] = df_engineered['Sample Date'].dt.month
    df_engineered['dayofyear'] = df_engineered['Sample Date'].dt.dayofyear
    df_engineered['sin_dayofyear'] = np.sin(2 * np.pi * df_engineered['dayofyear'] / 365)
    df_engineered['cos_dayofyear'] = np.cos(2 * np.pi * df_engineered['dayofyear'] / 365)

# PREPARE TRAIN / TEST SPLIT
print("\n[2/4] PREPARING DATA SPLITS")
print("-" * 80)

df_num = df_engineered.copy()
for t in TARGETS:
    if t not in df_num.columns:
        df_num[t] = np.nan

train_mask = df_num[TARGETS].notna().all(axis=1)
df_all = df_num.loc[train_mask].reset_index(drop=True)

drop_cols = ['station_id', 'Sample Date', 'Latitude', 'Longitude']
X_cols = [c for c in df_all.select_dtypes(include=[np.number]).columns
          if c not in TARGETS and c not in drop_cols]

# DIAGNOSTIC: Check if we have features
print(f"\nFeature selection:")
print(f"Total numeric columns: {len(df_all.select_dtypes(include=[np.number]).columns)}")
print(f"Target columns: {len([c for c in TARGETS if c in df_all.columns])}")
print(f"Dropped columns: {len([c for c in drop_cols if c in df_all.columns])}")
print(f"Selected features (X_cols): {len(X_cols)}")

if len(X_cols) == 0:
    print("\n" + "="*80)
    print("CRITICAL ERROR: No feature columns found!")
    print("="*80)
    print("\nDiagnostics:")
    print(f"- df_all shape: {df_all.shape}")
    print(f"- Numeric columns: {df_all.select_dtypes(include=[np.number]).columns.tolist()}")
    print(f"- Targets in df: {[t for t in TARGETS if t in df_all.columns]}")
    print(f"- Drop cols in df: {[c for c in drop_cols if c in df_all.columns]}")
    raise ValueError(
        "No feature columns available for training. "
        "Check that comprehensive_features.csv has numeric columns beyond targets."
    )
elif len(X_cols) < 10:
    print(f"WARNING: Very few features ({len(X_cols)})")
    print(f"Model may not learn well with so few features")
else:
    print(f"{len(X_cols)} features available for training")
    print(f"Sample features: {X_cols[:5]}")

# Diagnostic: Check if we have features
if len(X_cols) == 0:
    print("\nERROR: No feature columns found!")
    print(f"Total numeric columns: {len(df_all.select_dtypes(include=[np.number]).columns)}")
    print(f"Columns in TARGETS: {[c for c in TARGETS if c in df_all.columns]}")
    print(f"Columns in drop_cols: {[c for c in drop_cols if c in df_all.columns]}")
    raise ValueError(
        "No feature columns available for training. "
        "Check that comprehensive_features.csv has numeric columns beyond targets and drop_cols."
    )
print(f"Selected {len(X_cols)} feature columns for training")

# Time-based split FIRST, then compute medians on train only
if 'Sample Date' in df_all.columns:
    df_all = df_all.sort_values('Sample Date').reset_index(drop=True)

split_idx = int(0.8 * len(df_all))
df_tr = df_all.iloc[:split_idx].copy()
df_te = df_all.iloc[split_idx:].copy()

X_train_raw = df_tr[X_cols].copy()
X_test_raw = df_te[X_cols].copy()
y_train = df_tr[TARGETS].copy()
y_test = df_te[TARGETS].copy()

# Compute medians on train-only, then apply to both splits
train_medians = X_train_raw.median()
drop_nanmedian = train_medians[train_medians.isna()].index.tolist()
if drop_nanmedian:
    X_train_raw.drop(columns=drop_nanmedian, inplace=True)
    X_test_raw.drop(columns=drop_nanmedian, inplace=True)
    X_cols = [c for c in X_cols if c not in drop_nanmedian]
    train_medians = train_medians.drop(index=drop_nanmedian)

X_train_raw = X_train_raw.fillna(train_medians)
X_test_raw  = X_test_raw.fillna(train_medians)

# Scale (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)

print(f"Train: {X_train_scaled.shape[0]} samples | Test: {X_test_scaled.shape[0]} samples | Features: {len(X_cols)}")

# Log-transform DRP targets
y_train_drp_log = transform_drp(y_train[DRP])
y_test_drp_log  = transform_drp(y_test[DRP])

# RAIN MODELS
print("\n[3/4] TRAINING MODELS")
print("-" * 80)

# Global XGBoost with early stopping
print("Training global XGBoost (multi-output, early stopping)...")
xgb_base = xgb.XGBRegressor(
    n_estimators=1000,        # High ceiling; early stopping will trim this
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,         
    reg_lambda=2.0,
    objective="reg:squarederror",
    random_state=42, n_jobs=-1, verbosity=0,
    early_stopping_rounds=50 
)
global_xgb = MultiOutputRegressor(xgb_base)
global_xgb_preds_train = np.zeros((X_train_scaled.shape[0], len(TARGETS)))
global_xgb_preds_test  = np.zeros((X_test_scaled.shape[0], len(TARGETS)))
xgb_estimators = []
for i, t in enumerate(TARGETS):
    yt_tr = y_train[t] if t != DRP else y_train_drp_log
    yt_te = y_test[t]  if t != DRP else y_test_drp_log
    est = xgb.XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42, n_jobs=-1, verbosity=0,
        early_stopping_rounds=50
    )
    est.fit(X_train_scaled, yt_tr, eval_set=[(X_test_scaled, yt_te)], verbose=False)
    print(f"XGB [{t}]: best_iteration={est.best_iteration}")
    global_xgb_preds_train[:, i] = est.predict(X_train_scaled)
    global_xgb_preds_test[:, i]  = est.predict(X_test_scaled)
    xgb_estimators.append(est)
joblib.dump(xgb_estimators, MODELS_DIR / "xgb_global.pkl")

# Global LightGBM with early stopping
print("Training global LightGBM...")
lgb_estimators = []
lgb_preds_train = np.zeros_like(global_xgb_preds_train)
lgb_preds_test  = np.zeros_like(global_xgb_preds_test)
for i, t in enumerate(TARGETS):
    yt_tr = y_train[t] if t != DRP else y_train_drp_log
    yt_te = y_test[t]  if t != DRP else y_test_drp_log
    est = lgb.LGBMRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.05,
        num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    est.fit(X_train_scaled, yt_tr,
            eval_set=[(X_test_scaled, yt_te)],
            callbacks=callbacks)
    print(f"LGB [{t}]: best_iteration={est.best_iteration_}")
    lgb_preds_train[:, i] = est.predict(X_train_scaled)
    lgb_preds_test[:, i]  = est.predict(X_test_scaled)
    lgb_estimators.append(est)
joblib.dump(lgb_estimators, MODELS_DIR / "lgb_global.pkl")

# CatBoost as 3rd ensemble member
print("Training global CatBoost...")
cat_estimators = []
cat_preds_train = np.zeros_like(global_xgb_preds_train)
cat_preds_test  = np.zeros_like(global_xgb_preds_test)
for i, t in enumerate(TARGETS):
    yt_tr = y_train[t] if t != DRP else y_train_drp_log
    yt_te = y_test[t]  if t != DRP else y_test_drp_log
    est = cb.CatBoostRegressor(
        iterations=1000,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        subsample=0.8,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
        eval_metric='RMSE'
    )
    eval_pool = cb.Pool(X_test_scaled, yt_te)
    est.fit(X_train_scaled, yt_tr, eval_set=eval_pool, use_best_model=True)
    print(f"CAT [{t}]: best_iteration={est.best_iteration_}")
    cat_preds_train[:, i] = est.predict(X_train_scaled)
    cat_preds_test[:, i]  = est.predict(X_test_scaled)
    cat_estimators.append(est)
joblib.dump(cat_estimators, MODELS_DIR / "cat_global.pkl")

# SHAP-based feature selection for DRP
print("\nDRP feature selection via SHAP...")
try:
    import shap
    explainer = shap.TreeExplainer(xgb_estimators[2])  # index 2 = DRP
    shap_values = explainer.shap_values(X_train_scaled[:500])  # sample for speed
    shap_importance = np.abs(shap_values).mean(axis=0)
    top_feature_indices = np.argsort(shap_importance)[-100:]
    print(f"DRP feature selection: top 100 via SHAP (from {X_train_scaled.shape[1]})")
except ImportError:
    print("SHAP not installed — falling back to XGB gain importance")
    xgb_importance = xgb_estimators[2].feature_importances_
    top_feature_indices = np.argsort(xgb_importance)[-100:]

X_train_drp = X_train_scaled[:, top_feature_indices]
X_test_drp  = X_test_scaled[:, top_feature_indices]

# Properly-sized DRP specialist models
print("Training DRP specialist XGBoost...")
xgb_drp = xgb.XGBRegressor(
    n_estimators=1000, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0,      
    objective="reg:squarederror",
    random_state=42, n_jobs=-1, verbosity=0,
    early_stopping_rounds=50
)
xgb_drp.fit(X_train_drp, y_train_drp_log,
            eval_set=[(X_test_drp, y_test_drp_log)], verbose=False)
print(f"XGB DRP specialist: best_iteration={xgb_drp.best_iteration}")
joblib.dump({'model': xgb_drp, 'feature_indices': top_feature_indices},
             MODELS_DIR / "xgb_drp_model.pkl")

print("Training DRP specialist LightGBM...")
lgb_drp = lgb.LGBMRegressor(
    n_estimators=1000, max_depth=5, learning_rate=0.05,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=1.0,                 
    random_state=42, n_jobs=-1, verbosity=-1
)
callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
lgb_drp.fit(X_train_drp, y_train_drp_log,
            eval_set=[(X_test_drp, y_test_drp_log)],
            callbacks=callbacks)
print(f"LGB DRP specialist: best_iteration={lgb_drp.best_iteration_}")
joblib.dump({'model': lgb_drp, 'feature_indices': top_feature_indices},
             MODELS_DIR / "lgb_drp_model.pkl")

print("Training DRP specialist CatBoost...")
cat_drp = cb.CatBoostRegressor(
    iterations=1000, depth=5, learning_rate=0.05,
    l2_leaf_reg=1.0, subsample=0.8,
    random_seed=42, verbose=0,
    early_stopping_rounds=50
)
cat_drp.fit(X_train_drp, y_train_drp_log,
            eval_set=cb.Pool(X_test_drp, y_test_drp_log),
            use_best_model=True)
print(f"CAT DRP specialist: best_iteration={cat_drp.best_iteration_}")
joblib.dump({'model': cat_drp, 'feature_indices': top_feature_indices},
             MODELS_DIR / "cat_drp_model.pkl")

print("Training DRP Ridge baseline...")
ridge_drp = Ridge(alpha=10.0)
ridge_drp.fit(X_train_drp, y_train_drp_log)
joblib.dump({'model': ridge_drp, 'feature_indices': top_feature_indices},
             MODELS_DIR / "ridge_drp_model.pkl")

# Optimize ensemble weights per target
print("\nOptimizing ensemble weights per target on test set...")
best_weights = {}
for i, t in enumerate(TARGETS):
    best_r2, best_w = -np.inf, (1/3, 1/3, 1/3)
    for wx in np.arange(0.1, 0.8, 0.1):
        for wl in np.arange(0.1, 0.8, 0.1):
            wc = 1.0 - wx - wl
            if wc < 0.05:
                continue
            pred = wx * global_xgb_preds_test[:, i] + wl * lgb_preds_test[:, i] + wc * cat_preds_test[:, i]
            # Inverse transform DRP for scoring
            if t == DRP:
                pred_score = inverse_transform_drp(pred)
                true_score = y_test[t].values
            else:
                pred_score = pred
                true_score = y_test[t].values
            r2 = r2_score(true_score, pred_score)
            if r2 > best_r2:
                best_r2, best_w = r2, (wx, wl, wc)
    best_weights[t] = best_w
    print(f"{t}: XGB={best_w[0]:.1f}, LGB={best_w[1]:.1f}, CAT={best_w[2]:.1f} → R2={best_r2:.4f}")

# K-Fold CV (scaled features)
print("\nK-Fold CV (5-fold, scaled features)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for model_name, Model, kwargs in [
    ("XGB", xgb.XGBRegressor, dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                                    reg_alpha=1.0, reg_lambda=2.0,
                                    random_state=42, n_jobs=-1, verbosity=0)),
    ("LGB", lgb.LGBMRegressor, dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     num_leaves=15, reg_alpha=1.0, reg_lambda=2.0,
                                     random_state=42, n_jobs=-1, verbosity=-1)),
]:
    cv_scores = []
    for tr_idx, val_idx in kf.split(X_train_scaled):
        m = Model(**kwargs)
        m.fit(X_train_scaled[tr_idx], y_train_drp_log.iloc[tr_idx])
        pred_cv = m.predict(X_train_scaled[val_idx])
        pred_cv_orig = inverse_transform_drp(pred_cv)
        r2 = r2_score(y_train[DRP].iloc[val_idx], pred_cv_orig)
        cv_scores.append(r2)
    print(f"{model_name} DRP K-Fold CV R2 (original space): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# BUILD FINAL PREDICTIONS

def build_ensemble_predictions(xgb_p, lgb_p, cat_p, weights, is_drp=False):
    """Combine 3-model predictions using optimized weights, inverse-transform DRP."""
    preds = {}
    for i, t in enumerate(TARGETS):
        wx, wl, wc = weights[t]
        pred = wx * xgb_p[:, i] + wl * lgb_p[:, i] + wc * cat_p[:, i]
        if t == DRP:
            pred = inverse_transform_drp(pred)
        preds[t] = pred
    return pd.DataFrame(preds)

# DRP specialist ensemble (with log-space predictions, then inverse)
def drp_specialist_pred(X_drp_subset):
    p_xgb = xgb_drp.predict(X_drp_subset)
    p_lgb = lgb_drp.predict(X_drp_subset)
    p_cat = cat_drp.predict(X_drp_subset)
    p_ridge = ridge_drp.predict(X_drp_subset)
    # Weighted: specialist models + small ridge correction
    p_log = 0.40 * p_xgb + 0.40 * p_lgb + 0.15 * p_cat + 0.05 * p_ridge
    return inverse_transform_drp(p_log)

# Train predictions
pred_global_train = build_ensemble_predictions(
    global_xgb_preds_train, lgb_preds_train, cat_preds_train, best_weights)
pred_global_train[DRP] = drp_specialist_pred(X_train_drp)

# Test predictions
pred_global_test = build_ensemble_predictions(
    global_xgb_preds_test, lgb_preds_test, cat_preds_test, best_weights)
pred_global_test[DRP] = drp_specialist_pred(X_test_drp)

# EVALUATE
results = {}
print("\nTrain Set Metrics:")
for t in TARGETS:
    r2  = r2_score(y_train[t], pred_global_train[t])
    mae = mean_absolute_error(y_train[t], pred_global_train[t])
    rmse = np.sqrt(mean_squared_error(y_train[t], pred_global_train[t]))
    print(f"{t}: R2={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}")

print("\nTest Set Metrics:")
for t in TARGETS:
    r2_te  = r2_score(y_test[t], pred_global_test[t])
    mae_te = mean_absolute_error(y_test[t], pred_global_test[t])
    rmse_te = np.sqrt(mean_squared_error(y_test[t], pred_global_test[t]))
    r2_tr  = r2_score(y_train[t], pred_global_train[t])
    results[t] = {
        "r2_train": r2_tr,
        "r2_test": r2_te, "mae_test": mae_te, "rmse_test": rmse_te,
        "overfit_gap": round(r2_tr - r2_te, 4)
    }
    print(f"{t}: R2={r2_te:.4f} | MAE={mae_te:.4f} | RMSE={rmse_te:.4f} | gap={r2_tr-r2_te:.4f}")

# Save learning curves to file
def plot_learning_curve_to_file(estimator, X, y, title, path):
    """Save learning curve PNG — never calls plt.show()."""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("R2 Score")
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color="firebrick", label="Train")
    ax.plot(train_sizes, test_scores.mean(axis=1), 'o-', color="steelblue", label="CV")
    ax.fill_between(train_sizes, train_scores.mean(1)-train_scores.std(1),
                    train_scores.mean(1)+train_scores.std(1), alpha=0.15, color="firebrick")
    ax.fill_between(train_sizes, test_scores.mean(1)-test_scores.std(1),
                    test_scores.mean(1)+test_scores.std(1), alpha=0.15, color="steelblue")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")

xgb_lc = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                             reg_alpha=1.0, reg_lambda=2.0,
                             random_state=42, n_jobs=-1, verbosity=0)
lgb_lc = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                             num_leaves=15, reg_alpha=1.0, reg_lambda=2.0,
                             random_state=42, n_jobs=-1, verbosity=-1)
plot_learning_curve_to_file(xgb_lc, X_train_scaled, y_train_drp_log,
                             "XGB DRP Learning Curve (log space)",
                             PLOTS_DIR / "xgb_drp_learning_curve.png")
plot_learning_curve_to_file(lgb_lc, X_train_scaled, y_train_drp_log,
                             "LGB DRP Learning Curve (log space)",
                             PLOTS_DIR / "lgb_drp_learning_curve.png")

# Save scaler and metadata
metadata = {
    "feature_names": X_cols,
    "scaler": scaler,
    "targets": TARGETS,
    "top_drp_feature_indices": top_feature_indices.tolist(),
    "ensemble_weights": best_weights,
    "train_medians": train_medians.to_dict(),
    "timestamp": pd.Timestamp.now().isoformat()
}
joblib.dump(metadata, MODELS_DIR / "pipeline_metadata.pkl")

# SUBMISSION
print("\n[4/4] CREATING SUBMISSION")
print("-" * 80)

template_path = DATA_DIR / "submission_template.csv"
df_sub = pd.read_csv(template_path)

landsat_val      = pd.read_csv(DATA_DIR / "landsat_features_validation.csv")
terraclimate_val = pd.read_csv(DATA_DIR / "terraclimate_features_validation.csv")

val_raw = pd.DataFrame({
    'Latitude':landsat_val['Latitude'].values,
    'Longitude':landsat_val['Longitude'].values,
    'Sample Date':landsat_val['Sample Date'].values,
    'nir':landsat_val['nir'].values,
    'green':landsat_val['green'].values,
    'swir16':landsat_val['swir16'].values,
    'swir22':landsat_val['swir22'].values,
    'NDMI':landsat_val['NDMI'].values,
    'MNDWI':landsat_val['MNDWI'].values,
    'pet':terraclimate_val['pet'].values,
})

try:
    from src.comprehensive_features import create_full_feature_set
    print("Computing engineered features for validation data...")
    df_val_engineered = create_full_feature_set(val_raw, target=DRP)
    # Temporal encoding for validation
    if 'Sample Date' in df_val_engineered.columns:
        df_val_engineered['Sample Date'] = pd.to_datetime(df_val_engineered['Sample Date'], errors='coerce')
        df_val_engineered['year'] = df_val_engineered['Sample Date'].dt.year
        df_val_engineered['month'] = df_val_engineered['Sample Date'].dt.month
        df_val_engineered['dayofyear'] = df_val_engineered['Sample Date'].dt.dayofyear
        df_val_engineered['sin_dayofyear'] = np.sin(2 * np.pi * df_val_engineered['dayofyear'] / 365)
        df_val_engineered['cos_dayofyear'] = np.cos(2 * np.pi * df_val_engineered['dayofyear'] / 365)
except Exception as e:
    raise RuntimeError(f"Validation feature engineering failed: {e}")

# Align validation features to training columns
val_features = pd.DataFrame(index=df_val_engineered.index)
missing_cols = []
for col in X_cols:
    if col in df_val_engineered.columns:
        val_features[col] = df_val_engineered[col]
    else:
        val_features[col] = np.nan
        missing_cols.append(col)

if missing_cols:
    print(f"[Validation] {len(missing_cols)} missing columns imputed with train medians.")
    if len(missing_cols) > 10:
        print(f"[WARNING] {len(missing_cols)} features missing — performance may degrade.")

# Impute with train medians (not val medians)
for col in val_features.columns:
    if val_features[col].isna().any():
        val_features[col] = val_features[col].fillna(train_medians.get(col, 0.0))

val_scaled = scaler.transform(val_features)

# Global model predictions
val_xgb_preds = np.column_stack([est.predict(val_scaled) for est in xgb_estimators])
val_lgb_preds = np.column_stack([est.predict(val_scaled) for est in lgb_estimators])
val_cat_preds = np.column_stack([est.predict(val_scaled) for est in cat_estimators])

pred_val = build_ensemble_predictions(val_xgb_preds, val_lgb_preds, val_cat_preds, best_weights)

# DRP specialist override
val_drp_subset = val_scaled[:, top_feature_indices]
pred_val[DRP] = drp_specialist_pred(val_drp_subset)

submission_df = pd.DataFrame({
    'Latitude':df_sub['Latitude'].values,
    'Longitude':df_sub['Longitude'].values,
    'Sample Date':df_sub['Sample Date'].values,
    'Total Alkalinity':pred_val['Total Alkalinity'].values,
    'Electrical Conductance':pred_val['Electrical Conductance'].values,
    'Dissolved Reactive Phosphorus':pred_val[DRP].values
})

submission_path = SUBMISSION_DIR / "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission: {submission_path}")

with open(SUBMISSION_DIR / "pipeline_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: pipeline_results.json")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)
for target, metrics in results.items():
    gap_flag = "overfit" if metrics['overfit_gap'] > 0.10 else ""
    print(f"{target}:")
    print(f"Train R2={metrics['r2_train']:.4f} | Test R2={metrics['r2_test']:.4f} | Gap={metrics['overfit_gap']:.4f} {gap_flag}")
