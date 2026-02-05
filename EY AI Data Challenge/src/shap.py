import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
from pathlib import Path

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Try to import SHAP, fall back to sklearn feature importance if not available
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not available. Using sklearn feature importance instead.")

# Paths
OUT_DIR = os.path.join(PROJECT_ROOT, "submissions", "shap")
MODELS_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TARGETS = [
    'Total Alkalinity',
    'Electrical Conductance',
    'Dissolved Reactive Phosphorus'
]

os.makedirs(OUT_DIR, exist_ok=True)

# Import the feature building functions
import sys
sys.path.insert(0, PROJECT_ROOT)
from src.preprocessing import load_data, cleaning, add_time_features, create_station_id
from src.features import (
    add_lag_features, 
    add_rolling_features, 
    add_cyclic_time_features,
    finalize_features
)

# -----------------------
# Feature Importance Analysis for each target
# -----------------------

# =====================
# Load Data
# =====================
print("Loading training data...")
raw_train = pd.read_csv(os.path.join(DATA_DIR, "raw/water_quality.csv"))
raw_train = cleaning(raw_train)
raw_train = add_time_features(raw_train)
if 'station_id' not in raw_train.columns:
    raw_train = create_station_id(raw_train)
raw_train = add_cyclic_time_features(raw_train)

print("Building submission template features...")
sub = pd.read_csv(os.path.join(DATA_DIR, "submission_template.csv"))
landsat_val = pd.read_csv(os.path.join(DATA_DIR, "landsat_features_validation.csv"))
terra_val = pd.read_csv(os.path.join(DATA_DIR, "terraclimate_features_validation.csv"))

submit_df = pd.concat([
    sub, 
    landsat_val.drop(columns=["Latitude", "Longitude", "Sample Date"], errors='ignore'), 
    terra_val.drop(columns=["Latitude", "Longitude", "Sample Date"], errors='ignore')
], axis=1)
submit_df = submit_df.loc[:, ~submit_df.columns.duplicated()]
submit_df = cleaning(submit_df)
submit_df = add_time_features(submit_df)
submit_df = add_cyclic_time_features(submit_df)
submit_df = create_station_id(submit_df)

shap_results = {}

for TARGET in TARGETS:
    print(f"\n{'='*60}")
    print(f"Feature Importance Analysis for: {TARGET}")
    print(f"{'='*60}")
    
    # Find model for this target
    safe_name = re.sub(r"\W+", "_", TARGET)
    model_path = os.path.join(MODELS_DIR, f"xgb_model_{safe_name}.pkl")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Skipping {TARGET}.")
        continue
    
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Build features exactly as in generate_submission.py
    print(f"Building features for {TARGET}...")
    
    # Create training features for analysis
    df_train = raw_train.copy()
    df_train = add_lag_features(df_train, TARGET, groupby_col="station_id")
    df_train = add_rolling_features(df_train, TARGET, groupby_col="station_id")
    
    # Prepare training X (same as what model was trained on)
    X_train, _ = finalize_features(df_train, TARGET, exclude_columns=[t for t in TARGETS if t != TARGET])
    
    print(f"Features shape: {X_train.shape} | Model expects: {model.n_features_in_} features")
    
    # Sample for faster analysis
    X_sample = X_train.sample(min(500, len(X_train)), random_state=42) if len(X_train) > 500 else X_train
    
    # Get feature importance from XGBoost model
    try:
        importances = model.feature_importances_
        
        # Match features to what the model was trained on
        # Model has n_features_in_ features; take first N columns
        n_features = model.n_features_in_
        X_train = X_train.iloc[:, :n_features]
        
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Normalize
        feature_importance['Importance_Normalized'] = (
            feature_importance['Importance'] / (feature_importance['Importance'].sum() + 1e-10) * 100
        )
        
        print(f"\nTop 10 Important Features for {TARGET}:")
        print(feature_importance.head(10)[['Feature', 'Importance_Normalized']].to_string(index=False))
        
        # Save feature importance to JSON
        shap_results[TARGET] = {
            'top_features': feature_importance.head(10)[['Feature', 'Importance_Normalized']].rename(
                columns={'Importance_Normalized': 'Importance_%'}
            ).to_dict('records'),
            'num_features': X_train.shape[1],
            'method': 'XGBoost Gain-based Feature Importance'
        }
        
        # Create target-specific output directory
        target_out_dir = os.path.join(OUT_DIR, safe_name)
        os.makedirs(target_out_dir, exist_ok=True)
        
        # Save feature importance CSV
        feature_importance.to_csv(f"{target_out_dir}/feature_importance.csv", index=False)
        print(f"Saved: {target_out_dir}/feature_importance.csv")
        
        # Feature importance plot
        print("Generating feature importance plot...")
        top_n = min(15, len(feature_importance))
        top_features = feature_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_features)), top_features['Importance_Normalized'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'].values)
        ax.set_xlabel('Importance (%)')
        ax.set_title(f'Top {top_n} Features for {TARGET}\n(XGBoost Gain-based)')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{target_out_dir}/feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {target_out_dir}/feature_importance.png")
        
        # SHAP if available
        if HAS_SHAP:
            print("Computing SHAP values...")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Save raw SHAP outputs
                np.save(f"{target_out_dir}/shap_values.npy", shap_values)
                np.save(f"{target_out_dir}/expected_value.npy", explainer.expected_value)
                
                # Summary plot
                plt.figure(figsize=(12, 6))
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.tight_layout()
                plt.savefig(f"{target_out_dir}/shap_summary.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved: {target_out_dir}/shap_summary.png")
                
                # Bar plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                plt.tight_layout()
                plt.savefig(f"{target_out_dir}/shap_bar.png", dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved: {target_out_dir}/shap_bar.png")
            except Exception as e:
                print(f"Could not compute SHAP values: {e}")
        
    except Exception as e:
        print(f"Error analyzing {TARGET}: {e}")
        import traceback
        traceback.print_exc()
        continue

# -----------------------
# Save Summary Report
# -----------------------
summary_path = os.path.join(OUT_DIR, "feature_importance_summary.json")
with open(summary_path, 'w') as f:
    json.dump(shap_results, f, indent=2)

print(f"\n{'='*60}")
print(f"Analysis completed successfully!")
print(f"Results saved to: {OUT_DIR}")
print(f"Summary report: {summary_path}")
print(f"{'='*60}")