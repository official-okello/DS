# Feature Importance Analysis - Water Quality Prediction

## Executive Summary

Feature importance analysis has been completed for all three water quality targets using XGBoost's gain-based feature importance metrics. The analysis reveals which features drive model predictions for each target.

**Key Findings:**
- **Seasonal/Temporal Features Dominate**: Cyclic time encodings (sin/cos of day-of-year and month) are the most important features across all three targets
- **Lag Features Contribute**: Recent historical values (lag_1) are significant predictors, especially for EC and DRP
- **Geographic Features Matter**: Latitude is consistently important, but Longitude less so
- **Remote Sensing Features**: Landsat indices (NDMI, MNDWI) and climate variables (pet, precipitation) contribute meaningfully

---

## Target-Specific Analysis

### 1. **Total Alkalinity** (R² = 0.7384)
**Status**: Good performance with strong seasonal patterns

**Top 5 Features:**
| Feature | Importance (%) |
|---------|-----------------|
| sin_doy (sin of day-of-year) | 12.74% |
| cos_month | 10.47% |
| sin_month | 7.37% |
| cos_doy | 5.12% |
| Total Alkalinity_lag_1 | 4.54% |

**Interpretation:**
- Strong seasonal cycle: alkalinity varies predictably by time of year
- Recent historical values help (lag_1 at 4.5%)
- Spatial information (Latitude 2.9%, Longitude 2.3%) has moderate impact
- Remote sensing indices (NDMI, MNDWI, 2.5-2.5%) provide supplementary signals

**Recommendation:** Model is well-calibrated. Monitor seasonal patterns for validat forecasts.

---

### 2. **Electrical Conductance** (R² = 0.8086)
**Status**: Excellent performance with clear seasonal dependence

**Top 5 Features:**
| Feature | Importance (%) |
|---------|-----------------|
| sin_doy | 19.40% |
| cos_month | 14.18% |
| Electrical Conductance_lag_1 | 6.93% |
| sin_month | 5.68% |
| cos_doy | 4.22% |

**Interpretation:**
- **Extremely strong seasonal signal** (sin_doy dominates at 19.4%)
- EC is highly autocorrelated (lag_1 at 6.9%, suggesting persistence)
- Weaker spatial dependence than TA (Latitude 3.9%)
- Landsat indices and climate variables less important (3-4%)

**Recommendation:** Seasonal patterns are primary driver. Consider external validation against water chemistry databases. Lag features indicate good intra-station persistence.

---

### 3. **Dissolved Reactive Phosphorus** (R² = 0.3884)
**Status**: Low R² indicates challenging target with dispersed signal

**Top 5 Features:**
| Feature | Importance (%) |
|---------|-----------------|
| cos_doy | 10.11% |
| sin_doy | 8.52% |
| sin_month | 7.38% |
| DRP_lag_1 | 4.83% |
| cos_month | 4.58% |

**Key Observations:**
1. **Weaker Seasonal Pattern**: Unlike TA and EC, no single time feature dominates (max 10% vs 19%)
2. **Lower Lag Importance**: lag_1 is only 4.8% vs 4.5-6.9% for other targets
3. **More Balanced Feature Distribution**: Top 10 features collectively explain only ~54% of importance
4. **Spatial Dependence**: Latitude (3.37%) is relatively less important
5. **Rolling Statistics Help**: rolling_std_3 (3.11%), rolling_q25_7 (3.07%), rolling_med_3 (2.97%) suggest variability is important

**Why is DRP R² so low?**
- **Hypothesis 1: Sparse/Skewed Distribution**: DRP often has many zero/near-zero values. Models struggle with highly sparse targets
- **Hypothesis 2: Missing Domain Features**: Nutrient cycling (e.g., N-to-P ratios, biological activity proxies) may not be captured
- **Hypothesis 3: Temporal Non-Stationarity**: DRP patterns may vary by season/event in ways not captured by cyclic encodings
- **Hypothesis 4: Measurement/Data Quality**: DRP measurements may have higher noise or irregular sampling

---

## Feature Importance Files

All analysis results saved to `submissions/shap/`:

```
submissions/shap/
├── feature_importance_summary.json          # JSON summary with top 10 features per target
├── Total_Alkalinity/
│   ├── feature_importance.csv              # Full feature ranking
│   └── feature_importance.png              # Visualization (top 15 features)
├── Electrical_Conductance/
│   ├── feature_importance.csv
│   └── feature_importance.png
└── Dissolved_Reactive_Phosphorus/
    ├── feature_importance.csv
    └── feature_importance.png
```

---

## Recommendations for Model Improvement

### For Total Alkalinity & Electrical Conductance (Already Good)
1. **Cross-validate** seasonal patterns with external water quality databases
2. **Monitor for shifts** in baseline levels (e.g., due to land-use changes)
3. **Consider ensemble** with additional weak learners (ARIMA, SVR) for robustness

### For Dissolved Reactive Phosphorus (Priority)
1. **Data Inspection**:
   - Check distribution: is DRP highly skewed or sparse?
   - Verify measurement quality and sampling frequency
   - Look for outliers or measurement artifacts

2. **Feature Engineering**:
   - Add **nutrient-related features**: 
     - N-to-P ratio (if nitrogen data available)
     - Runoff proxies (precipitation lag + slope)
     - Land-use/soil type indicators
   - Longer lags (lag_7, lag_14, lag_30) already in model but may need window optimization
   - **Rolling quantiles** showing promise (3.07-3.11%) → expand to more quantiles (0.1, 0.9)

3. **Modeling Approach**:
   - Try **quantile regression** to capture DRP uncertainty better than mean regression
   - Test **zero-inflated models** if DRP has many zeros
   - Experiment with **log-normal or gamma GLM** instead of linear regression on log-transformed values

4. **Domain Expertise**:
   - Consult with limnologists/hydrologists on DRP drivers
   - DRP often related to erosion events → consider storm/event-based features
   - Algal blooms affect DRP → consider chlorophyll/turbidity proxies

---

## Next Steps

1. **Immediate**: Run this analysis alongside model predictions to validate feature contributions
2. **Short-term**: Implement DRP-specific features (nutrient ratios, event indicators)
3. **Medium-term**: Evaluate quantile regression or alternative loss functions for DRP
4. **Long-term**: Integrate external water quality datasets for model validation

---

## Model Performance Summary

| Target | R² | Key Driver | Confidence |
|--------|-----|-----------|-----------|
| Total Alkalinity | 0.7384 | Strong seasonal cycle | High |
| Electrical Conductance | 0.8086 | Very strong seasonal + autocorrelation | Very High |
| Dissolved Reactive Phosphorus | 0.3884 | Weak/dispersed signal | Low |

**Overall Average R²**: 0.6451

---

**Analysis Date**: February 2025  
**Method**: XGBoost Gain-based Feature Importance  
**Feature Count per Model**: 41 features  
**Training Data**: 9,319 historical observations across 162 unique stations
