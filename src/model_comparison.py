"""
=========================================================
Model Comparison
=========================================================
"""

import pandas as pd


def compare_models(results):

    comparison = pd.DataFrame(results)

    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(comparison)

    best_model = comparison.sort_values(
        by="R2",
        ascending=False
    ).iloc[0]

    print("\nBest Model")
    print("-" * 40)

    print(f"Model : {best_model['Model']}")
    print(f"R²    : {best_model['R2']:.4f}")

    return comparison