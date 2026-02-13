import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "outputs/models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("EXTERNAL FEATURE ENGINEERING")
print("=" * 80)

# Load water quality samples WITH TARGETS
features_path = DATA_DIR / "processed/comprehensive_features.csv"
if not features_path.exists():
    raise FileNotFoundError(
        f"{features_path} not found. Run comprehensive_features.py first."
    )

samples = pd.read_csv(features_path)
print(f"\n[1] Loaded comprehensive features: {samples.shape}")

# Verify targets are present
TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
missing_targets = [t for t in TARGETS if t not in samples.columns]

if missing_targets:
    print(f"\nMissing targets: {missing_targets}")
    print("Loading from water_quality.csv...")
    water_quality = pd.read_csv(DATA_DIR / "raw/water_quality.csv")
    for t in missing_targets:
        if t in water_quality.columns:
            samples[t] = water_quality[t].values
    print(f"Added {len(missing_targets)} target columns")

# CRITICAL: Split BEFORE any feature engineering
samples["date"] = pd.to_datetime(samples["Sample Date"], format="mixed", dayfirst=True)
samples = samples.sort_values("date").reset_index(drop=True)
split_idx = int(0.8 * len(samples))

samples_train = samples.iloc[:split_idx].copy()
samples_test = samples.iloc[split_idx:].copy()

print(f"\n[2] Train/Test split (temporal):")
print(f"Train: {len(samples_train)} samples ({samples_train['date'].min()} to {samples_train['date'].max()})")
print(f"Test:  {len(samples_test)} samples ({samples_test['date'].min()} to {samples_test['date'].max()})")

# Define columns to exclude from PCA
exclude_cols = TARGETS + ['Sample Date', 'station_id', 'Latitude', 'Longitude', 'date']
feature_cols = [c for c in samples_train.columns 
                if c not in exclude_cols 
                and samples_train[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

print(f"\n[3] Feature selection for PCA:")
print(f"Total columns: {len(samples.columns)}")
print(f"Excluded: {len(exclude_cols)} (targets + metadata)")
print(f"Selected for PCA: {len(feature_cols)}")

if len(feature_cols) == 0:
    raise ValueError("No numeric features found for PCA!")

# Fit on TRAIN only
X_train = samples_train[feature_cols].fillna(0).values
X_test = samples_test[feature_cols].fillna(0).values

print(f"\n[4] Fitting transformers on TRAIN data only...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # FIT on train
X_test_scaled = scaler.transform(X_test)        # TRANSFORM test (no fit!)

print(f"Scaler fitted on {X_train.shape[0]} train samples")

pca = PCA(n_components=0.95, svd_solver="full")
X_train_pca = pca.fit_transform(X_train_scaled)  # FIT on train
X_test_pca = pca.transform(X_test_scaled)        # TRANSFORM test (no fit!)

n_components = X_train_pca.shape[1]
print(f"PCA fitted on train only: {n_components} components (95% variance)")

# Create PCA dataframes
pca_col_names = [f"pca_{i}" for i in range(n_components)]

pca_train_df = pd.DataFrame(X_train_pca, columns=pca_col_names, index=samples_train.index)
pca_train_df["_split"] = "train"

pca_test_df = pd.DataFrame(X_test_pca, columns=pca_col_names, index=samples_test.index)
pca_test_df["_split"] = "test"

pca_all_df = pd.concat([pca_train_df, pca_test_df]).sort_index()

# Save outputs
out_dir = DATA_DIR / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

pca_all_df.to_csv(out_dir / "samples_with_pca_features.csv", index=False)
print(f"\n[5] Saved: {out_dir / 'samples_with_pca_features.csv'}")

# Save pipeline for validation data
joblib.dump(
    {
        "scaler": scaler,
        "pca": pca,
        "feature_cols": feature_cols,
        "n_components": n_components
    },
    MODELS_DIR / "pca_pipeline.pkl"
)
print(f"Saved: {MODELS_DIR / 'pca_pipeline.pkl'}")

print("\n" + "=" * 80)
print("EXTERNAL FEATURES PROCESSING PIPELINE COMPLETED")
print("=" * 80)
print(f"""
Summary:
  Scaler fitted ONLY on train data ({len(samples_train)} samples)
  PCA fitted ONLY on train data
  Test data transformed using train-fitted models
  
  Train samples: {len(samples_train)} | Test samples: {len(samples_test)}
  PCA components: {n_components}
  Features used: {len(feature_cols)} 
""")


def transform_validation_data(val_samples: pd.DataFrame) -> pd.DataFrame:
    """Transform validation data using train-fitted pipeline."""
    pipeline = joblib.load(MODELS_DIR / "pca_pipeline.pkl")
    scaler_ = pipeline["scaler"]
    pca_ = pipeline["pca"]
    feat_cols_ = pipeline["feature_cols"]

    # Ensure validation has same features
    X_val = val_samples.reindex(columns=feat_cols_).fillna(0).values
    X_val_scaled = scaler_.transform(X_val)  # Transform only, no fit
    X_val_pca = pca_.transform(X_val_scaled)  # Transform only, no fit

    col_names = [f"pca_{i}" for i in range(X_val_pca.shape[1])]
    result = pd.DataFrame(X_val_pca, columns=col_names, index=val_samples.index)

    return result