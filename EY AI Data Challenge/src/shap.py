import os
import glob
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from src.preprocessing import load_data, cleaning, add_time_features, create_station_id
from src.features import add_lag_features, add_rolling_features, finalize_features

# -----------------------
# Paths
# -----------------------
OUT_DIR = "outputs/shap"
MODELS_DIR = "outputs/models"
PROCESSED_PATH = "data/processed/feature_table.csv"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Find a model
# -----------------------
model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl")) + glob.glob(os.path.join(MODELS_DIR, "*.joblib"))
if not model_files:
    raise FileNotFoundError(f"No model file found in {MODELS_DIR}. Please run training first.")
model_path = model_files[0]
print(f"Using model: {model_path}")
model = joblib.load(model_path)

# Attempt to infer target name from model filename: xgb_model_{TARGET}.pkl
m = re.search(r"xgb_model_(.+)\.pkl", os.path.basename(model_path))
if m:
    TARGET = m.group(1).replace("_", " ")
    print(f"Inferred target from model filename: '{TARGET}'")
else:
    TARGET = None

# -----------------------
# Load or build data
# -----------------------
if os.path.exists(PROCESSED_PATH):
    df = pd.read_csv(PROCESSED_PATH)
    if TARGET and TARGET not in df.columns:
        print(f"Warning: inferred target '{TARGET}' not found in processed file; attempting to continue with available columns.")
else:
    # Try to build a small processed feature table from raw data
    raw_path = "data/raw/water_quality.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError("No processed feature table and no raw data found. Run the preprocessing and training pipeline first.")
    print("Building small processed feature table for SHAP from raw data...")
    df_raw = load_data(raw_path)
    df_raw = cleaning(df_raw)
    df_raw = add_time_features(df_raw)
    if "station_id" not in df_raw.columns:
        df_raw = create_station_id(df_raw)
    # If we inferred a TARGET use it, otherwise pick first numeric column that isn't lat/lon or Sample Date
    if TARGET is None:
        numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
        possible = [c for c in numeric_cols if c not in ["Latitude", "Longitude"]]
        if not possible:
            raise ValueError("No numeric target column could be inferred from raw data")
        TARGET = possible[0]
        print(f"Auto-selected target: {TARGET}")
    df_feat = df_raw.copy()
    df_feat = add_lag_features(df_feat, TARGET, groupby_col="station_id")
    df_feat = add_rolling_features(df_feat, TARGET, groupby_col="station_id")
    X, y = finalize_features(df_feat, TARGET)
    df = pd.concat([X, y], axis=1)
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

# Ensure TARGET exists
if TARGET is None or TARGET not in df.columns:
    raise ValueError("Could not determine target column to explain. Please provide a processed feature table with the target included.")

X = df.drop(columns=[TARGET])

# Optional: sample for speed
X_sample = X.sample(min(500, len(X)), random_state=42)

# -----------------------
# SHAP Explainer
# -----------------------
print("Initializing SHAP explainer...")
explainer = shap.TreeExplainer(model)

print("Computing SHAP values...")
shap_values = explainer.shap_values(X_sample)

# Save raw SHAP outputs
np.save(f"{OUT_DIR}/shap_values.npy", shap_values)
np.save(f"{OUT_DIR}/expected_value.npy", explainer.expected_value)

# -----------------------
# Global Feature Importance
# -----------------------
print("Generating global feature importance plot...")
plt.figure()
shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=300)
plt.close()

# -----------------------
# Bar plot (mean |SHAP|)
# -----------------------
print("Generating SHAP bar plot...")
plt.figure()
shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_bar.png", dpi=300)
plt.close()

# -----------------------
# Single Prediction Explanation
# -----------------------
print("Generating single prediction explanation...")
idx = 0  # change to any row index

shap.force_plot(
    explainer.expected_value,
    shap_values[idx],
    X_sample.iloc[idx],
    matplotlib=True,
    show=False
)

plt.savefig(f"{OUT_DIR}/shap_force_{idx}.png", dpi=300)
plt.close()

print("SHAP explainability completed successfully.")