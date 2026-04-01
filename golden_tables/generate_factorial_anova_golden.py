"""
Golden Table Generator — Factorial ANOVA
=========================================
Standalone auditor script.  Does NOT import from src/.

Replicates the exact output of the 'Factorial ANOVA' KNIME node
(Basic ANOVA table, Advanced ANOVA table, and Coefficients table).

KNIME Node Configuration to match:
  - Response Column       : Response
  - Factor Columns        : FactorA, FactorB
  - Include Interactions  : True
  - Max Interaction Order : 2
  - Alpha                 : 0.05
  - ANOVA Type            : Type III (advanced settings)
  - Advanced Output       : False → factorial_anova_golden_basic.csv
  - Advanced Output       : True  → factorial_anova_golden_advanced.csv
  - Coefficients output always → factorial_anova_golden_coefficients.csv

Run from repo root:
    python golden_tables/generate_factorial_anova_golden.py

Outputs written to golden_tables/data/:
    factorial_anova_input.csv
    factorial_anova_golden_basic.csv
    factorial_anova_golden_advanced.csv
    factorial_anova_golden_coefficients.csv
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 0.05
ANOVA_TYPE = 3   # Type III SS


def save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"  Wrote {name}: {df.shape[0]} rows x {df.shape[1]} cols — {list(df.columns)}")


def _format_p_value(p) -> float:
    """Full-precision float — matches factorial_anova/utils.py format_p_value (no rounding)."""
    if p is None:
        return float("nan")
    try:
        val = float(p)
    except (TypeError, ValueError):
        return float("nan")
    import math
    if math.isnan(val):
        return float("nan")
    return val   # no rounding — matches the PR change in utils.py


# ---------------------------------------------------------------------------
# Synthetic Input Data  (seed 42 → reproducible)
# 2-factor balanced design: FactorA × FactorB, 10 obs per cell
# Strong effects → ANOVA will be clearly significant
# ---------------------------------------------------------------------------
np.random.seed(42)

levels_a = ["A1", "A2", "A3"]
levels_b = ["B1", "B2"]
factor_means = {"A1": 10.0, "A2": 20.0, "A3": 30.0}   # large main effect for A
b_shift = {"B1": 0.0, "B2": 5.0}                         # main effect for B

rows = []
for la in levels_a:
    for lb in levels_b:
        mu = factor_means[la] + b_shift[lb]
        responses = np.random.normal(mu, 2.0, 10)
        for r in responses:
            rows.append({"FactorA": la, "FactorB": lb, "Response": r})

df_input = pd.DataFrame(rows)
save(df_input, "factorial_anova_input.csv")

# ---------------------------------------------------------------------------
# Fit OLS model with Type III SS
# Formula: Response ~ C(FactorA) * C(FactorB)
# Note: column names "FactorA", "FactorB", "Response" are already valid
# Python identifiers — no sanitization needed, patsy uses them as-is.
# ---------------------------------------------------------------------------
formula = "Response ~ C(FactorA) * C(FactorB)"
model = ols(formula, data=df_input).fit()

anova_table = sm.stats.anova_lm(model, typ=ANOVA_TYPE)

# ---------------------------------------------------------------------------
# Residual SS (used for partial eta squared denominator)
# ---------------------------------------------------------------------------
ss_residual = anova_table.loc["Residual", "sum_sq"]

# ---------------------------------------------------------------------------
# Basic ANOVA Table
# Columns: Factor | F-Statistic | P-Value | Conclusion
# Residual row EXCLUDED in basic output (matches format_basic_anova_table)
# Node rounding: NONE (full precision)
# ---------------------------------------------------------------------------
basic_rows = []
for idx, row in anova_table.iterrows():
    if idx == "Residual":
        continue
    p = row["PR(>F)"]
    if pd.notna(p) and p <= ALPHA:
        conclusion = "Significant"
    elif pd.notna(p):
        conclusion = "Not Significant"
    else:
        conclusion = "Unexplained"

    basic_rows.append(
        {
            "Factor": str(idx),
            "F-Statistic": float(row["F"]) if pd.notna(row["F"]) else np.nan,
            "P-Value": _format_p_value(p),
            "Conclusion": conclusion,
        }
    )

df_basic = pd.DataFrame(basic_rows)
save(df_basic, "factorial_anova_golden_basic.csv")

# ---------------------------------------------------------------------------
# Advanced ANOVA Table (includes Residual row + Partial Eta Squared)
# Columns: Source | Sum Sq | Mean Sq | DF | F-Statistic | P-Value | Partial Eta Squared | Conclusion
# DF dtype = int64 (matches knext.int64() schema in node)
# Partial Eta Sq = SS_effect / (SS_effect + SS_residual)
# ---------------------------------------------------------------------------
advanced_rows = []
for idx, row in anova_table.iterrows():
    df_val = row["df"]
    mean_sq = row["sum_sq"] / df_val if df_val > 0 else np.nan
    p = row["PR(>F)"]

    if idx == "Residual":
        conclusion = "Unexplained"
        partial_eta_sq = np.nan
    elif pd.notna(p) and p <= ALPHA:
        conclusion = "Significant"
        partial_eta_sq = row["sum_sq"] / (row["sum_sq"] + ss_residual)
    elif pd.notna(p):
        conclusion = "Not Significant"
        partial_eta_sq = row["sum_sq"] / (row["sum_sq"] + ss_residual)
    else:
        conclusion = "Unexplained"
        partial_eta_sq = np.nan

    advanced_rows.append(
        {
            "Source": str(idx),
            "Sum Sq": float(row["sum_sq"]),
            "Mean Sq": float(mean_sq) if pd.notna(mean_sq) else np.nan,
            "DF": int(df_val) if pd.notna(df_val) else 0,
            "F-Statistic": float(row["F"]) if pd.notna(row["F"]) else np.nan,
            "P-Value": _format_p_value(p),
            "Partial Eta Squared": float(partial_eta_sq) if pd.notna(partial_eta_sq) else np.nan,
            "Conclusion": conclusion,
        }
    )

df_advanced = pd.DataFrame(advanced_rows)
df_advanced["DF"] = df_advanced["DF"].astype("int64")
save(df_advanced, "factorial_anova_golden_advanced.csv")

# ---------------------------------------------------------------------------
# Coefficient Table
# Columns: Term | Coefficient | Std Error | P-Value | CI Lower | CI Upper
# Term names are patsy-generated (e.g. "C(FactorA)[T.A2]")
# Node rounding: NONE (full precision)
# ---------------------------------------------------------------------------
conf_int = model.conf_int()
coef_rows = {
    "Term": model.params.index.tolist(),
    "Coefficient": model.params.values,
    "Std Error": model.bse.values,
    "P-Value": [_format_p_value(p) for p in model.pvalues.values],
    "CI Lower": conf_int[0].values,
    "CI Upper": conf_int[1].values,
}
df_coef = pd.DataFrame(coef_rows)
save(df_coef, "factorial_anova_golden_coefficients.csv")

print("\nDone — factorial ANOVA golden tables written.")
