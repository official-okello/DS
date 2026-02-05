import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def add_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer spectral indices from raw Landsat bands.
    
    Available bands: nir (near-infrared), green, swir16 (shortwave), swir22
    """
    df = df.copy()
    
    # Normalized Difference Vegetation Index (NDVI)
    # Indicates vegetation health/density
    if 'nir' in df.columns and 'green' in df.columns:
        nir = df['nir'].astype(float)
        green = df['green'].astype(float)
        df['NDVI'] = (nir - green) / (nir + green + 1e-8)
        df['NDVI'] = df['NDVI'].clip(-1, 1)  # Bounds [-1, 1]
    
    # Normalized Difference Moisture Index (NDMI) - already in data but useful for reference
    # Higher = more water content in vegetation
    
    # Normalized Burn Ratio (NBR)
    # Useful for detecting disturbance/changes
    if 'nir' in df.columns and 'swir22' in df.columns:
        nir = df['nir'].astype(float)
        swir22 = df['swir22'].astype(float)
        df['NBR'] = (nir - swir22) / (nir + swir22 + 1e-8)
        df['NBR'] = df['NBR'].clip(-1, 1)
    
    # Enhanced Vegetation Index (EVI)
    # More sensitive to vegetation changes than NDVI
    if 'nir' in df.columns and 'green' in df.columns and 'swir16' in df.columns:
        nir = df['nir'].astype(float)
        green = df['green'].astype(float)
        swir16 = df['swir16'].astype(float)
        L, C1, C2 = 1.0, 6.0, 7.5  # Standard EVI coefficients
        df['EVI'] = 2.5 * (nir - green) / (nir + C1*green - C2*swir16 + L + 1e-8)
        df['EVI'] = df['EVI'].clip(-1, 3)  # Realistic bounds
    
    # Soil-Adjusted Vegetation Index (SAVI)
    # Better for areas with exposed soil
    if 'nir' in df.columns and 'green' in df.columns:
        nir = df['nir'].astype(float)
        green = df['green'].astype(float)
        L = 0.5  # Soil adjustment factor
        df['SAVI'] = (nir - green) / (nir + green + L + 1e-8) * (1 + L)
        df['SAVI'] = df['SAVI'].clip(-1, 1)
    
    # Normalized Difference Water Index (NDWI)
    # Indicates water in vegetation (flooding detection)
    if 'nir' in df.columns and 'swir16' in df.columns:
        nir = df['nir'].astype(float)
        swir16 = df['swir16'].astype(float)
        df['NDWI_alt'] = (nir - swir16) / (nir + swir16 + 1e-8)
        df['NDWI_alt'] = df['NDWI_alt'].clip(-1, 1)
    
    # Modified Soil-Adjusted Vegetation Index (MSAVI)
    # Improved SAVI for low vegetation areas
    if 'nir' in df.columns and 'green' in df.columns:
        nir = df['nir'].astype(float)
        green = df['green'].astype(float)
        rvi = nir / (green + 1e-8)
        df['MSAVI'] = (2*nir + 1 - np.sqrt((2*nir + 1)**2 - 8*(nir - green))) / 2
        df['MSAVI'] = df['MSAVI'].clip(-1, 1)
    
    # Spectral indices ratios (for nutrient/stress detection)
    if 'swir16' in df.columns and 'green' in df.columns:
        swir16 = df['swir16'].astype(float)
        green = df['green'].astype(float)
        df['SWIR_GREEN_RATIO'] = swir16 / (green + 1e-8)
    
    if 'swir16' in df.columns and 'nir' in df.columns:
        swir16 = df['swir16'].astype(float)
        nir = df['nir'].astype(float)
        df['SWIR_NIR_RATIO'] = swir16 / (nir + 1e-8)
    
    return df


def add_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer climate/weather features from TerrAClimate data.
    
    Current: pet (Potential EvapotransPiration)
    Can add: precipitation lags, drought indices
    """
    df = df.copy()
    
    if 'pet' not in df.columns:
        return df
    
    # PET statistics (variability in evaporative demand)
    df['pet_zscore'] = (df['pet'] - df['pet'].mean()) / (df['pet'].std() + 1e-8)
    
    # PET categories
    pet_quantiles = df['pet'].quantile([0.25, 0.5, 0.75])
    df['pet_category'] = pd.cut(df['pet'], 
                                bins=[0, pet_quantiles[0.25], pet_quantiles[0.5], 
                                      pet_quantiles[0.75], df['pet'].max()],
                                labels=['low', 'medium', 'high', 'very_high'],
                                include_lowest=True)
    
    # Convert to numeric
    df['pet_category'] = df['pet_category'].cat.codes
    
    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer spatial features from latitude/longitude.
    """
    df = df.copy()
    
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        return df
    
    # Spatial clustering: lat/lon interactions
    df['lat_lon_product'] = df['Latitude'] * df['Longitude']
    df['lat_lon_sum'] = df['Latitude'] + df['Longitude']
    df['lat_lon_distance'] = np.sqrt(df['Latitude']**2 + df['Longitude']**2)
    
    # Distance from center (approximate center of study area)
    center_lat, center_lon = -30.5, 23.0  # Approximate center of South Africa
    df['dist_from_center'] = np.sqrt((df['Latitude'] - center_lat)**2 + 
                                     (df['Longitude'] - center_lon)**2)
    
    # Regional indicators (quadrants/zones)
    df['latitude_zone'] = pd.cut(df['Latitude'], bins=4, labels=['south', 'south_mid', 'north_mid', 'north'])
    df['longitude_zone'] = pd.cut(df['Longitude'], bins=4, labels=['west', 'west_mid', 'east_mid', 'east'])
    df['latitude_zone'] = df['latitude_zone'].cat.codes
    df['longitude_zone'] = df['longitude_zone'].cat.codes
    
    return df


def add_temporal_fourier_features(df: pd.DataFrame, date_col: str = "Sample Date", periods: list = None) -> pd.DataFrame:
    """
    Add Fourier features for multiple seasonal periods.
    Better than just sin/cos for capturing complex seasonal patterns.
    """
    if periods is None:
        periods = [7, 30, 365]  # Weekly, monthly, yearly
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
    
    # Day of year as continuous variable
    day_of_year = df[date_col].dt.dayofyear
    
    for period in periods:
        for harmonic in range(1, 3):  # Use 2 harmonics per period
            df[f'sin_{period}d_h{harmonic}'] = np.sin(2 * np.pi * harmonic * day_of_year / period)
            df[f'cos_{period}d_h{harmonic}'] = np.cos(2 * np.pi * harmonic * day_of_year / period)
    
    return df


def add_advanced_rolling_features(df: pd.DataFrame, target: str, groupby_col: str = "station_id", 
                                 windows: list = None, quantiles: list = None) -> pd.DataFrame:
    """
    Add advanced rolling statistics including skewness, kurtosis, range.
    Useful for capturing data shape/distribution changes.
    """
    if windows is None:
        windows = [3, 7, 14, 30, 60]  # Extended windows
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]  # More quantiles
    
    df = df.copy()
    if target not in df.columns:
        return df
    
    grouped = df.groupby(groupby_col)[target].shift(1)
    
    for w in windows:
        shifted = grouped.rolling(window=w, min_periods=1)
        
        # Standard stats
        df[f"{target}_roll_mean_{w}"] = shifted.mean()
        df[f"{target}_roll_std_{w}"] = shifted.std()
        df[f"{target}_roll_med_{w}"] = shifted.median()
        df[f"{target}_roll_min_{w}"] = shifted.min()
        df[f"{target}_roll_max_{w}"] = shifted.max()
        
        # Range and IQR
        df[f"{target}_roll_range_{w}"] = df[f"{target}_roll_max_{w}"] - df[f"{target}_roll_min_{w}"]
        
        # Skewness and kurtosis (distribution shape)
        df[f"{target}_roll_skew_{w}"] = shifted.skew()
        df[f"{target}_roll_kurt_{w}"] = shifted.apply(lambda x: x.kurtosis() if len(x) > 3 else np.nan)
        
        # Coefficient of variation
        df[f"{target}_roll_cv_{w}"] = (df[f"{target}_roll_std_{w}"] / 
                                       (df[f"{target}_roll_mean_{w}"] + 1e-8)).fillna(0)
        
        # Quantiles (sparse + expanded)
        for q in quantiles:
            q_name = int(q * 100)
            try:
                df[f"{target}_roll_q{q_name}_{w}"] = shifted.quantile(q)
            except:
                df[f"{target}_roll_q{q_name}_{w}"] = shifted.apply(lambda x: x.quantile(q) if len(x) > 0 else np.nan)
    
    return df


def add_interaction_features(df: pd.DataFrame, targets: list = None) -> pd.DataFrame:
    """
    Add interaction features between different targets and spectral/climate data.
    Useful for capturing cross-variable dependencies.
    """
    df = df.copy()
    if targets is None:
        targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Target interactions
    available_targets = [t for t in targets if t in df.columns]
    if len(available_targets) >= 2:
        for i, t1 in enumerate(available_targets):
            for t2 in available_targets[i+1:]:
                df[f"{t1}_x_{t2}"] = df[t1] * df[t2]
                df[f"{t1}_div_{t2}"] = df[t1] / (df[t2] + 1e-8)
                df[f"{t1}_{t2}_sum"] = df[t1] + df[t2]
    
    # Spectral indices with targets
    spectral_cols = [c for c in df.columns if c in ['NDVI', 'EVI', 'SAVI', 'NDMI', 'MNDWI', 'NBR']]
    for target in available_targets:
        for spec in spectral_cols:
            if spec in df.columns:
                df[f"{target}_x_{spec}"] = df[target] * df[spec]
                df[f"{target}_{spec}_ratio"] = df[target] / (df[spec].abs() + 1e-8)
    
    # PET interactions
    if 'pet' in df.columns:
        for target in available_targets:
            df[f"{target}_per_pet"] = df[target] / (df['pet'] + 1e-8)
    
    return df


def add_derived_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived indices for water quality prediction.
    Combines multiple features to create meaningful composite metrics.
    """
    df = df.copy()
    
    # Land Surface Water Index (combines NDVI, NDWI, MNDWI)
    if 'NDVI' in df.columns and 'MNDWI' in df.columns:
        df['LSWI'] = (df['NDVI'] - df['MNDWI']) / (df['NDVI'] + df['MNDWI'] + 1e-8)
    
    # Vegetation Moisture Index
    if 'NDVI' in df.columns and 'NDWI_alt' in df.columns:
        df['VMI'] = df['NDVI'] * df['NDWI_alt']
    
    # Stress Index (low vegetation + high moisture = potential stress)
    if 'NDVI' in df.columns and 'MNDWI' in df.columns:
        df['Stress_Index'] = (1 - df['NDVI']) * df['MNDWI']
    
    # Soil Moisture Index (SWIR bands)
    if 'SWIR_GREEN_RATIO' in df.columns:
        df['Soil_Moisture_Index'] = 1 / (df['SWIR_GREEN_RATIO'] + 1e-8)
    
    # Wetness Index (water-related features)
    if 'MNDWI' in df.columns and 'NDVI' in df.columns:
        df['Wetness_Index'] = (df['MNDWI'] + df['NDVI']) / 2
    
    return df


def create_full_feature_set(df: pd.DataFrame, target: str = None, groupby_col: str = "station_id") -> pd.DataFrame:
    """
    Apply all feature engineering transformations in sequence.
    """
    print("Engineering spectral indices...")
    df = add_spectral_indices(df)
    
    print("Engineering climate features...")
    df = add_climate_features(df)
    
    print("Engineering spatial features...")
    df = add_spatial_features(df)
    
    print("Engineering temporal Fourier features...")
    df = add_temporal_fourier_features(df)
    
    if target and target in df.columns:
        print(f"Engineering advanced rolling features for {target}...")
        df = add_advanced_rolling_features(df, target, groupby_col)
    
    print("Engineering interaction features...")
    df = add_interaction_features(df)
    
    print("Engineering derived indices...")
    df = add_derived_indices(df)
    
    return df


if __name__ == "__main__":
    print("="*70)
    print("COMPREHENSIVE FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    raw_data = pd.read_csv(DATA_DIR / "raw/water_quality.csv")
    print(f"Initial shape: {raw_data.shape}")
    
    # Parse dates
    raw_data['Sample Date'] = pd.to_datetime(raw_data['Sample Date'], format='mixed', dayfirst=True)
    
    # Create station_id if not present
    from src.preprocessing import create_station_id, add_time_features
    if 'station_id' not in raw_data.columns:
        raw_data = create_station_id(raw_data)
    
    # Apply all transformations
    featured_df = create_full_feature_set(raw_data, target='Dissolved Reactive Phosphorus')
    
    print(f"\nFinal shape: {featured_df.shape}")
    print(f"New features added: {featured_df.shape[1] - raw_data.shape[1]}")
    
    # Show feature categories
    print("\n" + "="*70)
    print("FEATURE SUMMARY")
    print("="*70)
    
    spectral = [c for c in featured_df.columns if any(x in c for x in ['NDVI', 'EVI', 'SAVI', 'NBR', 'NDWI', 'SWIR'])]
    climate = [c for c in featured_df.columns if 'pet' in c]
    spatial = [c for c in featured_df.columns if any(x in c for x in ['lat', 'lon', 'dist', 'zone'])]
    temporal = [c for c in featured_df.columns if any(x in c for x in ['sin_', 'cos_', 'dayofyear', 'month'])]
    rolling = [c for c in featured_df.columns if 'roll_' in c]
    derived = [c for c in featured_df.columns if any(x in c for x in ['LSWI', 'VMI', 'Stress', 'Soil_', 'Wetness'])]
    interaction = [c for c in featured_df.columns if '_x_' in c or '_div_' in c or '_per_' in c]
    
    print(f"\nSpectral Indices ({len(spectral)}): {spectral[:5]}...")
    print(f"Climate Features ({len(climate)}): {climate}")
    print(f"Spatial Features ({len(spatial)}): {spatial[:5]}...")
    print(f"Temporal Features ({len(temporal)}): {temporal[:5]}...")
    print(f"Rolling Statistics ({len(rolling)}): {rolling[:5]}...")
    print(f"Derived Indices ({len(derived)}): {derived}")
    print(f"Interaction Features ({len(interaction)}): {interaction[:5]}...")
    
    # Save for inspection
    output_path = PROJECT_ROOT / "data/processed/comprehensive_features.csv"
    featured_df.to_csv(output_path, index=False)
    print(f"\nSaved comprehensive features to: {output_path}")
