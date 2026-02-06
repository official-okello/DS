import os
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load water quality samples
samples = pd.read_csv("water_quality_samples.csv")
samples["date"] = pd.to_datetime(samples["date"])

# 2. Download and preprocess CHIRPS rainfall (monthly)
chirps_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.2020.monthly.nc"
chirps_nc = "chirps-v2.0.2020.monthly.nc"
if not os.path.exists(chirps_nc):
    import requests
    print(f"Downloading {chirps_url} ...")
    r = requests.get(chirps_url, stream=True)
    with open(chirps_nc, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Downloaded CHIRPS.")
assert os.path.exists(chirps_nc), f"Download {chirps_nc} from CHIRPS website."
ds = xr.open_dataset(chirps_nc)
ds = ds.rio.write_crs("EPSG:4326")
sa = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres")).query("name == 'South Africa'")
chirps_sa = ds.rio.clip(sa.geometry, sa.crs)
chirps_df = chirps_sa["precip"].to_dataframe().reset_index()
chirps_df["time"] = pd.to_datetime(chirps_df["time"])
tree = cKDTree(chirps_df[["lat", "lon"]].values)

def extract_rain_features(row):
    dist, idx = tree.query([row.lat, row.lon])
    cell = chirps_df.iloc[idx]
    past = chirps_df[
        (chirps_df["lat"] == cell.lat) &
        (chirps_df["lon"] == cell.lon) &
        (chirps_df["time"] <= row.date)
    ]
    return pd.Series({
        "rain_1m": past.tail(1)["precip"].sum(),
        "rain_3m": past.tail(3)["precip"].sum(),
        "rain_6m": past.tail(6)["precip"].sum(),
    })

samples = samples.join(samples.apply(extract_rain_features, axis=1))

# 3. Download and preprocess ERA5 climate features
# Requires CDS API setup: https://cds.climate.copernicus.eu/api-how-to
# You must have ~/.cdsapirc configured with your credentials
try:
    import cdsapi
    era5_nc = "era5_sa.nc"
    if not os.path.exists(era5_nc):
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-land-monthly-means",
            {
                "variable": [
                    "2m_temperature",
                    "total_precipitation",
                    "soil_moisture_volumetric_layer_1",
                ],
                "year": ["2018", "2019", "2020"],
                "month": [f"{m:02d}" for m in range(1,13)],
                "time": "00:00",
                "area": [-22, 16, -35, 33],  # SA bounding box
                "format": "netcdf",
            },
            era5_nc
        )
    era = xr.open_dataset(era5_nc)
    era = era.rio.write_crs("EPSG:4326")
    era_df = era.to_dataframe().reset_index()
    era_df["time"] = pd.to_datetime(era_df["time"])
    era_tree = cKDTree(era_df[["latitude", "longitude"]].values)
    def extract_era5_features(row):
        dist, idx = era_tree.query([row.lat, row.lon])
        cell = era_df.iloc[idx]
        past = era_df[
            (era_df["latitude"] == cell.latitude) &
            (era_df["longitude"] == cell.longitude) &
            (era_df["time"] <= row.date)
        ]
        return pd.Series({
            "temp_1m": past.tail(1)["t2m"].mean() if "t2m" in past else np.nan,
            "soil_moisture_3m": past.tail(3)["swvl1"].mean() if "swvl1" in past else np.nan,
            "precip_1m": past.tail(1)["tp"].sum() if "tp" in past else np.nan,
        })
    samples = samples.join(samples.apply(extract_era5_features, axis=1))
except Exception as e:
    print(f"ERA5 download or processing skipped: {e}")

# 4. Download and preprocess Digital Earth Africa water quality indices
try:
    from pystac_client import Client
    from odc.stac import load
    import rasterio
    dea_catalog = Client.open("https://explorer.digitalearth.africa/stac")
    search = dea_catalog.search(
        collections=["deafrica-ls-water-quality"],
        bbox=[16, -35, 33, -22],
        datetime="2011-01-01/2015-12-31"
    )
    items = list(search.get_items())
    ds = load(
        items,
        bands=["chlorophyll_a", "turbidity", "tsm"],
        crs="EPSG:4326",
        resolution=30,
    )
    def extract_dea_features(row):
        pixel = ds.sel(x=row.lon, y=row.lat, method="nearest")
        return pd.Series({
            "chl_a": float(pixel.chlorophyll_a.mean().values),
            "turbidity": float(pixel.turbidity.mean().values),
            "tsm": float(pixel.tsm.mean().values),
        })
    samples = samples.join(samples.apply(extract_dea_features, axis=1))
except Exception as e:
    print(f"DEA download or processing skipped: {e}")

# 5. Download and compute distance to rivers
try:
    import osmnx as ox
    from shapely.geometry import Point
    rivers = ox.geometries_from_place(
        "South Africa",
        tags={"waterway": "river"}
    )
    rivers = rivers.to_crs("EPSG:32735")  # metric CRS
    samples_gdf = gpd.GeoDataFrame(
        samples,
        geometry=gpd.points_from_xy(samples.lon, samples.lat),
        crs="EPSG:4326"
    ).to_crs("EPSG:32735")
    samples["dist_to_river_m"] = samples_gdf.geometry.apply(
        lambda x: rivers.distance(x).min()
    )
except Exception as e:
    print(f"River distance skipped: {e}")

# 6. Dimensionality reduction (PCA)
features = [
    c for c in samples.columns if c not in ["sample_id", "lat", "lon", "date"]
]
X = samples[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95, svd_solver='full')  # retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# Save processed features
pca_df = pd.DataFrame(X_pca, index=samples.index)
pca_df["sample_id"] = samples["sample_id"]
pca_df.to_csv("samples_with_pca_features.csv", index=False)

print("Done! PCA features saved to samples_with_pca_features.csv")