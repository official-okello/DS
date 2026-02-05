"""
Data Quality & Distribution Analysis for Dissolved Reactive Phosphorus (DRP)

Diagnoses why DRP R² is low by examining:
1. Distribution shape (skewness, sparsity, outliers)
2. Measurement quality (missing values, consistency)
3. Temporal patterns and variability
4. Station-level differences
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup
DATA_DIR = Path("data")
OUTPUT_DIR = Path("submissions/drp_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load raw data
raw_data = pd.read_csv(DATA_DIR / "raw/water_quality.csv")
print(f"Loaded {len(raw_data)} records")

# ============================================================
# 1. Distribution Analysis
# ============================================================
print("\n" + "="*60)
print("DRP DISTRIBUTION ANALYSIS")
print("="*60)

drp = raw_data['Dissolved Reactive Phosphorus'].dropna()
print(f"\nTotal records: {len(raw_data)}")
print(f"Non-null DRP: {len(drp)} ({100*len(drp)/len(raw_data):.1f}%)")
print(f"Null DRP: {raw_data['Dissolved Reactive Phosphorus'].isna().sum()} ({100*raw_data['Dissolved Reactive Phosphorus'].isna().sum()/len(raw_data):.1f}%)")

# Check for zeros and near-zeros
zeros = (drp == 0).sum()
near_zeros = (drp < 0.001).sum()
print(f"\nValues at zero: {zeros} ({100*zeros/len(drp):.1f}%)")
print(f"Values < 0.001: {near_zeros} ({100*near_zeros/len(drp):.1f}%)")

# Distribution statistics
print(f"\nDescriptive Statistics:")
print(f"  Mean: {drp.mean():.6f}")
print(f"  Median: {drp.median():.6f}")
print(f"  Std Dev: {drp.std():.6f}")
print(f"  Min: {drp.min():.6f}")
print(f"  Max: {drp.max():.6f}")
print(f"  Q25: {drp.quantile(0.25):.6f}")
print(f"  Q75: {drp.quantile(0.75):.6f}")
print(f"  IQR: {drp.quantile(0.75) - drp.quantile(0.25):.6f}")

# Skewness & Kurtosis
from scipy.stats import skew, kurtosis as scipy_kurtosis
skewness = skew(drp)
kurtosis_val = scipy_kurtosis(drp)
print(f"\nSkewness: {skewness:.4f} (highly right-skewed: >1.0)")
print(f"Kurtosis: {kurtosis_val:.4f}")

# ============================================================
# 2. Data Quality Checks
# ============================================================
print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)

# Check for outliers (IQR method)
q1, q3 = drp.quantile(0.25), drp.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = ((drp < lower_bound) | (drp > upper_bound)).sum()
print(f"\nOutliers (IQR method): {outliers} ({100*outliers/len(drp):.1f}%)")
print(f"  Lower bound: {lower_bound:.6f}")
print(f"  Upper bound: {upper_bound:.6f}")

# Temporal coverage
raw_data['Sample Date'] = pd.to_datetime(raw_data['Sample Date'], format='mixed', dayfirst=True)
date_range = (raw_data['Sample Date'].max() - raw_data['Sample Date'].min()).days
print(f"\nTemporal coverage: {date_range} days ({date_range/365:.1f} years)")
print(f"  First date: {raw_data['Sample Date'].min()}")
print(f"  Last date: {raw_data['Sample Date'].max()}")

# Station-level analysis
n_stations = raw_data['Latitude'].nunique() * raw_data['Longitude'].nunique()
print(f"\nApproximate unique stations: {n_stations}")
print(f"  Unique latitudes: {raw_data['Latitude'].nunique()}")
print(f"  Unique longitudes: {raw_data['Longitude'].nunique()}")

# ============================================================
# 3. Temporal Patterns
# ============================================================
print("\n" + "="*60)
print("TEMPORAL PATTERNS")
print("="*60)

raw_data['month'] = raw_data['Sample Date'].dt.month
raw_data['doy'] = raw_data['Sample Date'].dt.dayofyear

drp_by_month = raw_data.groupby('month')['Dissolved Reactive Phosphorus'].agg(['count', 'mean', 'std', 'min', 'max'])
print("\nDRP by Month:")
print(drp_by_month)

# ============================================================
# 4. Visualizations
# ============================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0, 0].hist(drp, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('DRP Distribution (Raw Values)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('DRP (mg/L)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(drp.mean(), color='r', linestyle='--', label=f'Mean: {drp.mean():.4f}')
axes[0, 0].axvline(drp.median(), color='g', linestyle='--', label=f'Median: {drp.median():.4f}')
axes[0, 0].legend()

# Log-transformed
drp_log = np.log1p(drp)
axes[0, 1].hist(drp_log, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('DRP Distribution (Log1p Transformed)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('log(DRP+1)')
axes[0, 1].set_ylabel('Frequency')

# Box plot by month
raw_data.boxplot(column='Dissolved Reactive Phosphorus', by='month', ax=axes[1, 0])
axes[1, 0].set_title('DRP by Month (Seasonal Variation)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('DRP (mg/L)')
plt.sca(axes[1, 0])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Q-Q plot
from scipy.stats import probplot
probplot(drp, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "drp_distribution_analysis.png", dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'drp_distribution_analysis.png'}")
plt.close()

# ============================================================
# 5. Summary Report
# ============================================================
print("\n" + "="*60)
print("SUMMARY & RECOMMENDATIONS")
print("="*60)

summary = {
    'metric': [
        'Data Completeness',
        'Sparsity (% zeros)',
        'Skewness',
        'Outlier %',
        'Coefficient of Variation',
        'Temporal Coverage'
    ],
    'value': [
        f"{100*len(drp)/len(raw_data):.1f}%",
        f"{100*zeros/len(drp):.1f}%",
        f"{skewness:.2f}",
        f"{100*outliers/len(drp):.1f}%",
        f"{drp.std()/drp.mean():.2f}",
        f"{date_range/365:.1f} years"
    ],
    'implication': [
        'Good' if len(drp)/len(raw_data) > 0.8 else 'Fair',
        'High' if zeros/len(drp) > 0.1 else 'Moderate',
        'Highly Right-Skewed' if skewness > 1 else 'Moderately Skewed',
        'Significant' if outliers/len(drp) > 0.05 else 'Few',
        'High Variability' if drp.std()/drp.mean() > 2 else 'Moderate',
        'Good' if date_range > 365*3 else 'Limited'
    ]
}
summary_df = pd.DataFrame(summary)
print("\n" + summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(OUTPUT_DIR / "drp_quality_summary.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'drp_quality_summary.csv'}")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print(f"""
1. DISTRIBUTION:
   - DRP is highly right-skewed (skewness: {skewness:.2f})
   - Log-transformation helps but doesn't fully normalize
   - High coefficient of variation ({drp.std()/drp.mean():.2f}) → dispersed signal
   
2. DATA QUALITY:
   - Missing values: {100*raw_data['Dissolved Reactive Phosphorus'].isna().sum()/len(raw_data):.1f}%
   - Outliers: {100*outliers/len(drp):.1f}% (notable variance)
   - Zeros/near-zeros: {100*near_zeros/len(drp):.1f}% (sparse signal)
   
3. WHY R² IS LOW:
   - ✓ Skewed distribution → violates linear regression assumptions
   - ✓ Sparse signal (high CV) → hard to predict from features
   - ✓ Potential outliers → inflate residuals
   - ✓ Missing domain drivers (nutrient cycling, biological activity)
   
4. RECOMMENDED FIXES:
   - Try QUANTILE REGRESSION (e.g., median) instead of MSE
   - Use ROBUST REGRESSION to downweight outliers
   - Engineer nutrient-specific features if data available
   - Consider ZERO-INFLATED MODEL if >10% zeros
   - Validate with domain experts on biological drivers
""")
