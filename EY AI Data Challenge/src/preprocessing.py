import os
import pandas as pd
from typing import Tuple


def combine_datasets(dataset1: pd.DataFrame, dataset2: pd.DataFrame, dataset3: pd.DataFrame) -> pd.DataFrame:
    # Concatenate datasets column-wise and drop duplicated columns
    data = pd.concat([dataset1, dataset2, dataset3], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    return data


def build_raw_dataset(
    water_csv: str = "data/water_quality_training_dataset.csv",
    landsat_csv: str = "data/landsat_features_training.csv",
    terraclimate_csv: str = "data/terraclimate_features_training.csv",
    out_path: str = "data/raw/water_quality.csv"
) -> pd.DataFrame:
    """
    Read the raw CSVs, combine them and write combined CSV to out_path.
    Ensures target columns from water_quality_training_dataset.csv are preserved.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    water_df = pd.read_csv(water_csv)
    landsat_df = pd.read_csv(landsat_csv)
    terra_df = pd.read_csv(terraclimate_csv)
    
    # Combine datasets
    combined = combine_datasets(water_df, landsat_df, terra_df)
    
    # Verify target columns are present
    target_cols = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    missing_targets = [t for t in target_cols if t not in combined.columns]
    if missing_targets:
        print(f"WARNING: Missing target columns: {missing_targets}")
        print(f"These should come from {water_csv}")
    else:
        print(f"All target columns present in combined dataset")
    
    combined.to_csv(out_path, index=False)
    print(f"Saved combined dataset to {out_path} ({combined.shape[0]} rows, {combined.shape[1]} cols)")
    
    return combined


def cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: fill numeric NAs with median, drop duplicates and sort by Sample Date."""
    df = df.copy()
    if df.select_dtypes(include="number").shape[1] > 0:
        df = df.fillna(df.median(numeric_only=True))
    df = df.drop_duplicates()
    if "Sample Date" in df.columns:
        df["Sample Date"] = pd.to_datetime(df["Sample Date"], errors="coerce")
        df = df.sort_values("Sample Date")
    df = df.reset_index(drop=True)
    return df


def create_station_id(df: pd.DataFrame, lat_col: str = "Latitude", lon_col: str = "Longitude", precision: int = 4, inplace: bool = False) -> pd.DataFrame:
    """Create a `station_id` column from latitude/longitude by rounding to `precision` decimals."""
    if not inplace:
        df = df.copy()
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(f"Columns {lat_col} and {lon_col} must exist to create station_id")
    lat_str = df[lat_col].round(precision).astype(str)
    lon_str = df[lon_col].round(precision).astype(str)
    df["station_id"] = lat_str + "_" + lon_str
    return df