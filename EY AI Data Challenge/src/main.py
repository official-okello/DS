from src.preprocessing import (
    load_data,
    cleaning,
    add_time_features,
    build_raw_dataset,
    create_station_id,
)
from src.features import add_lag_features, add_rolling_features, finalize_features
from src.train_xgb import train_xgb
from src.tune_xgb import tune_model
from sklearn.model_selection import train_test_split
import os
import re

TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
RAW_PATH = "data/raw/water_quality.csv"

# Ensure raw dataset exists
if not os.path.exists(RAW_PATH):
    build_raw_dataset(out_path=RAW_PATH)

# Load and prepare
if __name__ == "__main__":
    df = load_data(RAW_PATH)
    df = cleaning(df)
    df = add_time_features(df)

    # create station_id if not present
    if "station_id" not in df.columns:
        df = create_station_id(df)

    os.makedirs("outputs/models", exist_ok=True)

    for TARGET in TARGETS:
        print(f"Processing target: {TARGET}")
        # work on a copy per target to avoid mixing NaNs across targets
        df_target = df.copy()
        # Drop other target columns so model doesn't use them as features
        other_targets = [t for t in TARGETS if t != TARGET]
        if other_targets:
            df_target = df_target.drop(columns=other_targets, errors='ignore')

        df_target = add_lag_features(df_target, TARGET, groupby_col="station_id")
        df_target = add_rolling_features(df_target, TARGET, groupby_col="station_id")

        X, y = finalize_features(df_target, TARGET, exclude_columns=other_targets)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        best_params = tune_model(X_train, y_train, X_test, y_test)

        safe_name = re.sub(r"\W+", "_", TARGET)
        model = train_xgb(
            X_train,
            y_train,
            X_test,
            y_test,
            output_path=f"outputs/models/xgb_model_{safe_name}.pkl"
        )