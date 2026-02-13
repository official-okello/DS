import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def add_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Spectral indices from raw Landsat bands. No cross-row stats — safe."""
    df = df.copy()

    if 'nir' in df.columns and 'green' in df.columns:
        nir   = df['nir'].astype(float)
        green = df['green'].astype(float)
        df['NDVI'] = ((nir - green) / (nir + green + 1e-8)).clip(-1, 1)
        L = 0.5
        df['SAVI'] = (((nir - green) / (nir + green + L + 1e-8)) * (1 + L)).clip(-1, 1)
        # MSAVI
        df['MSAVI'] = ((2*nir + 1 - np.sqrt(np.clip((2*nir + 1)**2 - 8*(nir - green), 0, None))) / 2).clip(-1, 1)

    if 'nir' in df.columns and 'swir22' in df.columns:
        nir    = df['nir'].astype(float)
        swir22 = df['swir22'].astype(float)
        df['NBR'] = ((nir - swir22) / (nir + swir22 + 1e-8)).clip(-1, 1)

    if 'nir' in df.columns and 'green' in df.columns and 'swir16' in df.columns:
        nir    = df['nir'].astype(float)
        green  = df['green'].astype(float)
        swir16 = df['swir16'].astype(float)
        L, C1, C2 = 1.0, 6.0, 7.5
        df['EVI'] = (2.5 * (nir - green) / (nir + C1*green - C2*swir16 + L + 1e-8)).clip(-1, 3)

    if 'nir' in df.columns and 'swir16' in df.columns:
        nir    = df['nir'].astype(float)
        swir16 = df['swir16'].astype(float)
        df['NDWI_alt'] = ((nir - swir16) / (nir + swir16 + 1e-8)).clip(-1, 1)

    if 'swir16' in df.columns and 'green' in df.columns:
        df['SWIR_GREEN_RATIO'] = df['swir16'].astype(float) / (df['green'].astype(float) + 1e-8)

    if 'swir16' in df.columns and 'nir' in df.columns:
        df['SWIR_NIR_RATIO'] = df['swir16'].astype(float) / (df['nir'].astype(float) + 1e-8)

    return df


def add_climate_features(df: pd.DataFrame,
                          pet_quantiles: dict = None) -> pd.DataFrame:
    
    df = df.copy()
    
    # Check if 'pet' column exists
    if 'pet' not in df.columns:
        return df

    if pet_quantiles is not None:
        bins = [df['pet'].min() - 1,
                pet_quantiles[0.25],
                pet_quantiles[0.50],
                pet_quantiles[0.75],
                df['pet'].max() + 1]
        # Ensure bins are strictly increasing (edge case: duplicate quantiles)
        bins = sorted(set(bins))
        if len(bins) >= 2:
            df['pet_category'] = pd.cut(
                df['pet'], bins=bins,
                labels=False, include_lowest=True
            ).fillna(0).astype(int)
        else:
            df['pet_category'] = 0
    else:
        df['pet_category'] = 0

    return df


def compute_pet_stats(df_train: pd.DataFrame) -> dict:
    """Compute PET statistics from training data only. Returns None if 'pet' column is missing."""
    if 'pet' not in df_train.columns:
        print("  [INFO] 'pet' column not found — skipping PET stats.")
        return None
    pet = df_train['pet'].dropna()
    if len(pet) == 0:
        print("  [INFO] 'pet' column has no valid values — skipping PET stats.")
        return None
    return {
        'pet_quantiles': {0.25: float(pet.quantile(0.25)),
                          0.50: float(pet.quantile(0.50)),
                          0.75: float(pet.quantile(0.75))},
    }


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Spatial features from lat/lon. Row-wise only — no cross-row stats."""
    df = df.copy()
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        return df

    df['lat_lon_distance'] = np.sqrt(df['Latitude']**2 + df['Longitude']**2)

    # Approximate centroid of South Africa
    center_lat, center_lon = -30.5, 23.0
    df['dist_from_center'] = np.sqrt(
        (df['Latitude']  - center_lat)**2 +
        (df['Longitude'] - center_lon)**2
    )
    return df


def add_temporal_fourier_features(df: pd.DataFrame,
                                   date_col: str = "Sample Date",
                                   periods: list = None) -> pd.DataFrame:
    """Fourier features for seasonal patterns. Row-wise — no leakage."""
    if periods is None:
        periods = [7, 30, 365]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
    day_of_year = df[date_col].dt.dayofyear

    for period in periods:
        for harmonic in range(1, 3):
            df[f'sin_{period}d_h{harmonic}'] = np.sin(2 * np.pi * harmonic * day_of_year / period)
            df[f'cos_{period}d_h{harmonic}'] = np.cos(2 * np.pi * harmonic * day_of_year / period)

    return df


def add_rolling_features(df: pd.DataFrame,
                          feature_cols: list,
                          groupby_col: str = "station_id",
                          windows: list = None) -> pd.DataFrame:
    if windows is None:
        windows = [3, 7, 14, 30]

    df = df.copy()

    # Only roll features that actually exist in df
    cols_to_roll = [c for c in feature_cols if c in df.columns]

    for col in cols_to_roll:
        if groupby_col in df.columns:
            shifted = df.groupby(groupby_col)[col].shift(1)
        else:
            shifted = df[col].shift(1)

        for w in windows:
            rolled = shifted.rolling(window=w, min_periods=1)
            df[f'{col}_roll_mean_{w}'] = rolled.mean()
            df[f'{col}_roll_std_{w}']  = rolled.std().fillna(0)
            df[f'{col}_roll_min_{w}']  = rolled.min()
            df[f'{col}_roll_max_{w}']  = rolled.max()

    return df


def add_interaction_features(df: pd.DataFrame,
                              targets: list = None) -> pd.DataFrame:

    df = df.copy()

    # Spectral × climate interactions (safe — these are inputs, not targets)
    # Include both engineered indices AND pre-existing ones (like NDMI, MNDWI)
    potential_spectral = ['NDVI', 'EVI', 'SAVI', 'NDMI', 'MNDWI', 'NBR',
                          'NDWI_alt', 'SWIR_GREEN_RATIO', 'SWIR_NIR_RATIO', 'MSAVI']
    spectral_cols = [c for c in df.columns if c in potential_spectral]
    
    climate_cols  = [c for c in df.columns if c in ['pet']]
    spatial_cols  = [c for c in df.columns if c in ['dist_from_center', 'lat_lon_distance']]

    # Only create interactions if we have both spectral and climate/spatial features
    for spec in spectral_cols:
        for clim in climate_cols:
            df[f'{spec}_x_{clim}'] = df[spec] * df[clim]

        for spat in spatial_cols:
            df[f'{spec}_x_{spat}'] = df[spec] * df[spat]

    # Spectral band ratios (check existence first)
    if 'NDVI' in df.columns and 'NDWI_alt' in df.columns:
        df['NDVI_NDWI_diff'] = df['NDVI'] - df['NDWI_alt']
        df['NDVI_NDWI_prod'] = df['NDVI'] * df['NDWI_alt']

    if 'EVI' in df.columns and 'NDVI' in df.columns:
        df['EVI_NDVI_ratio'] = df['EVI'] / (df['NDVI'].abs() + 1e-8)
    
    # Additional interactions with pre-existing indices
    if 'NDVI' in df.columns and 'NDMI' in df.columns:
        df['NDVI_NDMI_prod'] = df['NDVI'] * df['NDMI']
    
    if 'NDVI' in df.columns and 'MNDWI' in df.columns:
        df['NDVI_MNDWI_prod'] = df['NDVI'] * df['MNDWI']

    # No target × anything interactions. Targets are outputs, not inputs.
    if targets:
        accidentally_included = [t for t in targets if t in df.columns]
        if accidentally_included:
            print(f"[WARNING] Dropping target columns that may leake into interaction features: "
                  f"{accidentally_included}")
            df = df.drop(columns=accidentally_included, errors='ignore')

    return df


def add_derived_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Composite indices from spectral features only. Row-wise — no leakage."""
    df = df.copy()

    # Check if indices exist (either created earlier or pre-existing in data)
    has_ndvi = 'NDVI' in df.columns
    has_mndwi = 'MNDWI' in df.columns
    has_ndwi_alt = 'NDWI_alt' in df.columns
    has_swir_green = 'SWIR_GREEN_RATIO' in df.columns

    if has_ndvi and has_mndwi:
        df['LSWI'] = ((df['NDVI'] - df['MNDWI']) / (df['NDVI'] + df['MNDWI'] + 1e-8))
        df['Stress_Index'] = (1 - df['NDVI']) * df['MNDWI']
        df['Wetness_Index'] = (df['MNDWI'] + df['NDVI']) / 2

    if has_ndvi and has_ndwi_alt:
        df['VMI'] = df['NDVI'] * df['NDWI_alt']

    if has_swir_green:
        df['Soil_Moisture_Index'] = 1 / (df['SWIR_GREEN_RATIO'] + 1e-8)

    return df


# Public API

def create_full_feature_set(df: pd.DataFrame,
                             target: str = None,
                             groupby_col: str = "station_id",
                             pet_stats: dict = None,
                             is_train: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    *** CRITICAL: THIS VERSION NEVER LOSES TARGET COLUMNS ***
    """
    TARGETS = ['Total Alkalinity', 'Electrical Conductance',
               'Dissolved Reactive Phosphorus']

    targets_backup = {}
    for t in TARGETS:
        if t in df.columns:
            targets_backup[t] = df[t].copy()
            print(f"Backed up target: {t}")
    
    # Track original columns
    original_cols = set(df.columns)
    
    print("Engineering spectral indices...")
    df = add_spectral_indices(df)
    
    if pet_stats is None and is_train:
        pet_stats = compute_pet_stats(df)
    
    print("Engineering climate features...")
    if pet_stats is not None:
        df = add_climate_features(df, **pet_stats)
    
    print("Engineering spatial features...")
    df = add_spatial_features(df)
    
    print("Engineering temporal Fourier features...")
    df = add_temporal_fourier_features(df)
    
    print("Engineering interaction features...")
    df = add_interaction_features(df, targets=TARGETS)
    
    print("Engineering derived indices...")
    df = add_derived_indices(df)
    
    print("\nRestoring targets...")
    for t, values in targets_backup.items():
        df[t] = values
        print(f"Restored: {t}")
    
    total_new = len(df.columns) - len(original_cols)
    print(f"\nTotal features: {len(df.columns)} ({total_new} new + {len(targets_backup)} targets)")
    
    return df

if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE FEATURE ENGINEERING - GUARANTEED TARGET PRESERVATION")
    print("=" * 70)

    from src.preprocessing import build_raw_dataset
    print("\nBuilding raw dataset...")
    build_raw_dataset()

    raw_data = pd.read_csv(DATA_DIR / "raw/water_quality.csv")
    raw_data['Sample Date'] = pd.to_datetime(
        raw_data['Sample Date'], format='mixed', dayfirst=True)

    from src.preprocessing import create_station_id
    if 'station_id' not in raw_data.columns:
        raw_data = create_station_id(raw_data)

    raw_data = raw_data.sort_values('Sample Date').reset_index(drop=True)
    
    # Compute PET stats from training portion
    split_idx = int(0.8 * len(raw_data))
    pet_stats = compute_pet_stats(raw_data.iloc[:split_idx])
    
    print("\n✓ Applying feature engineering...")
    df_engineered = create_full_feature_set(raw_data, pet_stats=pet_stats, is_train=True)
    
    # Verify targets
    TARGETS = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    final_targets = [t for t in TARGETS if t in df_engineered.columns]
    
    print(f"\nVerification: {len(final_targets)}/3 targets present")
    
    # Save
    output_path = DATA_DIR / "processed/comprehensive_features.csv"
    df_engineered.to_csv(output_path, index=False)
    
    print(f"\nSaved: {output_path}")
    print(f"Shape: {df_engineered.shape}")
    
    if len(final_targets) == 3:
        print(f"\nSUCCESS - All targets preserved!")
    else:
        print(f"\nERROR - Missing targets: {[t for t in TARGETS if t not in df_engineered.columns]}")