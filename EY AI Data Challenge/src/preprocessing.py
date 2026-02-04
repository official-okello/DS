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
    
    # Read the raw CSVs, combine them and write combined CSV to out_path.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    water_df = pd.read_csv(water_csv)
    landsat_df = pd.read_csv(landsat_csv)
    terra_df = pd.read_csv(terraclimate_csv)
    combined = combine_datasets(water_df, landsat_df, terra_df)
    combined.to_csv(out_path, index=False)
    return combined


def load_data(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file_path)


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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month/year/dayofyear features from Sample Date if present."""
    df = df.copy()
    if "Sample Date" in df.columns:
        df["month"] = df["Sample Date"].dt.month
        df["year"] = df["Sample Date"].dt.year
        df["dayofyear"] = df["Sample Date"].dt.dayofyear
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


if __name__ == "__main__":
    print("Building raw dataset at data/raw/water_quality.csv")
    build_raw_dataset()