#!/usr/bin/env python
"""
Run SHAP analysis for all targets and generate interpretable reports.
Focus on understanding why DRP has low R2.
"""
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Run SHAP analysis
print("Running SHAP analysis for all targets...")
print("=" * 80)

try:
    # Execute the enhanced shap.py
    import src.shap as shap_module
    print("\n" + "=" * 80)
    print("SHAP analysis completed successfully!")
    print("\nInterpretation Guide:")
    print("-" * 80)
    print("1. shap_summary.png: Beeswarm plot showing feature impact on model predictions")
    print("   - Each dot = one prediction instance")
    print("   - Dot position (x-axis) = impact on prediction")
    print("   - Color = feature value (red=high, blue=low)")
    print("\n2. shap_bar.png: Mean absolute SHAP values per feature")
    print("   - Bar height = average feature importance")
    print("   - Top features have most impact on predictions")
    print("\n3. feature_importance.csv: Ranked list of important features")
    print("-" * 80)
    print("\nFor DRP (low R2 target):")
    print("  - Review top features in feature_importance.csv")
    print("  - Check if lag/rolling features are helping")
    print("  - Consider: log transform may need tuning, or more domain features needed")
    print("  - Check raw data distribution - DRP may have skewed/sparse values")
    print("-" * 80)
    
except Exception as e:
    print(f"Error during SHAP analysis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
