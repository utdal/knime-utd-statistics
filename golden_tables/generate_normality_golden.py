"""
Golden Table Generator — Statistical Normality Tests
=====================================================
Standalone auditor script.  Does NOT import from src/.

Replicates the exact output of the 'Statistical Normality Tests' KNIME node
for both available methods.

KNIME Node Configuration to match:
  - Alpha        : 0.05
  - Columns      : normal_col, uniform_col, skewed_col

Run from repo root:
    python golden_tables/generate_normality_golden.py

Outputs written to golden_tables/data/:
    normality_input.csv
    normality_golden_anderson.csv
    normality_golden_cramer.csv
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import cramervonmises, norm
from statsmodels.stats.diagnostic import normal_ad

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 0.05
COLUMNS = ["normal_col", "uniform_col", "skewed_col"]


def save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"  Wrote {name}: {df.shape[0]} rows x {df.shape[1]} cols — {list(df.columns)}")


# ---------------------------------------------------------------------------
# Synthetic Input Data  (seed 42 → reproducible)
# ---------------------------------------------------------------------------
np.random.seed(42)
df_input = pd.DataFrame(
    {
        "normal_col": np.random.normal(0, 1, 100),
        "uniform_col": np.random.uniform(0, 1, 100),
        "skewed_col": np.random.exponential(1, 100),
    }
)
save(df_input, "normality_input.csv")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _decision(p_value: float) -> str:
    return "Reject normality" if p_value <= ALPHA else "Do not reject normality"


# ---------------------------------------------------------------------------
# Anderson-Darling  — statsmodels.stats.diagnostic.normal_ad
# Node rounding: NONE (p-values passed through at full float precision)
# ---------------------------------------------------------------------------
rows_ad = []
for col in COLUMNS:
    arr = df_input[col].values
    statistic, p_value = normal_ad(arr)
    rows_ad.append(
        {
            "Column Tested": col,
            "Test Method": "Anderson-Darling",
            "Sample Size (n)": int(len(arr)),      # int32 in KNIME schema
            "Test Statistic": float(statistic),
            "P-Value": float(p_value),
            "Statistical Decision": _decision(p_value),
        }
    )

df_ad = pd.DataFrame(rows_ad)
df_ad["Sample Size (n)"] = df_ad["Sample Size (n)"].astype("int32")
save(df_ad, "normality_golden_anderson.csv")


# ---------------------------------------------------------------------------
# Cramer-von Mises  — scipy.stats.cramervonmises
# Node rounding: NONE
# ---------------------------------------------------------------------------
rows_cvm = []
for col in COLUMNS:
    arr = df_input[col].values
    result = cramervonmises(arr, norm.cdf)
    rows_cvm.append(
        {
            "Column Tested": col,
            "Test Method": "Cramer-von Mises",
            "Sample Size (n)": int(len(arr)),
            "Test Statistic": float(result.statistic),
            "P-Value": float(result.pvalue),
            "Statistical Decision": _decision(result.pvalue),
        }
    )

df_cvm = pd.DataFrame(rows_cvm)
df_cvm["Sample Size (n)"] = df_cvm["Sample Size (n)"].astype("int32")
save(df_cvm, "normality_golden_cramer.csv")

print("\nDone — normality golden tables written.")
