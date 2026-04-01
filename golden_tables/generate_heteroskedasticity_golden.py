"""
Golden Table Generator — Heteroskedasticity Tests
==================================================
Standalone auditor script.  Does NOT import from src/.

Replicates the exact output of the 'Heteroskedasticity Tests' KNIME node
for all three available methods (Breusch-Pagan, White, Goldfeld-Quandt),
each in both Basic and Advanced model-summary modes.

KNIME Node Configuration to match:
  - Alpha             : 0.05
  - Target Column     : Y
  - Predictor Columns : X1, X2
  - Output Detail     : Basic  → *_model_basic.csv
  - Output Detail     : Advanced → *_model_advanced.csv

  For Goldfeld-Quandt also set:
    - Sort Variable     : X1
    - Split Fraction    : 0.33

Run from repo root:
    python golden_tables/generate_heteroskedasticity_golden.py

Outputs written to golden_tables/data/ (4 files per method × 3 methods = 12):
    heteroskedasticity_input.csv
    heteroskedasticity_golden_{bp|white|gq}_test.csv
    heteroskedasticity_golden_{bp|white|gq}_model_basic.csv
    heteroskedasticity_golden_{bp|white|gq}_model_advanced.csv
    heteroskedasticity_golden_{bp|white|gq}_predictions.csv
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_goldfeldquandt, het_white

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 0.05
TARGET_COL = "Y"
GQ_SORT_VAR = "X1"
GQ_SPLIT = 0.33


def save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"  Wrote {name}: {df.shape[0]} rows x {df.shape[1]} cols — {list(df.columns)}")


def _format_p_value(p) -> float:
    """Round p-value to 4 decimal places (matches heteroskedasticity/utils.py)."""
    if p is None:
        return float("nan")
    try:
        val = float(p)
    except (TypeError, ValueError):
        return float("nan")
    import math
    if math.isnan(val):
        return float("nan")
    return round(val, 4)


# ---------------------------------------------------------------------------
# Synthetic Input Data  (seed 42 → reproducible)
# Y = 2 + 3*X1 + 1.5*X2 + X1 * noise  → heteroskedastic by design
# ---------------------------------------------------------------------------
np.random.seed(42)
X1 = np.random.normal(0, 1, 100)
X2 = np.random.normal(5, 2, 100)
Y = 2 + 3 * X1 + 1.5 * X2 + X1 * np.random.normal(0, 1, 100)

df_input = pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})
save(df_input, "heteroskedasticity_input.csv")

# ---------------------------------------------------------------------------
# Fit shared OLS model
# Matches regression_core.fit_ols_model: sm.OLS(y, sm.add_constant(X)).fit()
# ---------------------------------------------------------------------------
X_df = df_input[["X1", "X2"]].copy()
y_series = df_input[TARGET_COL].copy()

X_with_const = sm.add_constant(X_df, has_constant="add")
model = sm.OLS(y_series, X_with_const).fit()

predictions = model.fittedvalues
residuals = y_series - predictions

# ---------------------------------------------------------------------------
# Model summary components
# Matches regression_core.extract_model_summary output exactly
# ---------------------------------------------------------------------------
coef_table = pd.DataFrame(
    {
        "Variable": model.params.index.tolist(),
        "Coefficient": model.params.values,
        "Std Error": model.bse.values,
        "t-statistic": model.tvalues.values,
        "P>|t|": model.pvalues.values,
        "[0.025": model.conf_int()[0].values,
        "0.975]": model.conf_int()[1].values,
    }
)

metrics_table = pd.DataFrame(
    {
        "Metric": [
            "R-squared",
            "Adjusted R-squared",
            "F-statistic",
            "Prob (F-statistic)",
            "No. Observations",
            "Df Residuals",
            "Df Model",
        ],
        "Value": [
            model.rsquared,
            model.rsquared_adj,
            model.fvalue,
            model.f_pvalue,
            float(model.nobs),
            float(model.df_resid),
            float(model.df_model),
        ],
    }
)

f_pvalue = metrics_table.loc[metrics_table["Metric"] == "Prob (F-statistic)", "Value"].values[0]


def _predictions_df() -> pd.DataFrame:
    """Output 1: target | prediction | residual  (matches node output exactly)."""
    return pd.DataFrame(
        {
            TARGET_COL: y_series.values,
            "prediction": predictions.values,
            "residual": residuals.values,
        }
    )


def _model_summary_basic_df() -> pd.DataFrame:
    """Output 2 Basic: Information | Measure | P-Value."""
    rows = []
    # Coefficients
    for _, row in coef_table.iterrows():
        rows.append(
            {
                "Information": row["Variable"],
                "Measure": row["Coefficient"],
                "P-Value": _format_p_value(row["P>|t|"]),
            }
        )
    # Key metrics
    key_metrics = ["R-squared", "Adjusted R-squared", "F-statistic", "No. Observations"]
    for metric_name in key_metrics:
        mrow = metrics_table.loc[metrics_table["Metric"] == metric_name]
        if not mrow.empty:
            p_val = _format_p_value(f_pvalue) if metric_name == "F-statistic" else np.nan
            rows.append(
                {
                    "Information": metric_name,
                    "Measure": float(mrow["Value"].values[0]),
                    "P-Value": p_val,
                }
            )
    return pd.DataFrame(rows)


def _model_summary_advanced_df() -> pd.DataFrame:
    """Output 2 Advanced: Type | Information | Measure | Std Error | P-Value."""
    rows = []
    # Coefficients (Type = "Coefficient")
    for _, row in coef_table.iterrows():
        rows.append(
            {
                "Type": "Coefficient",
                "Information": row["Variable"],
                "Measure": row["Coefficient"],
                "Std Error": row["Std Error"],
                "P-Value": _format_p_value(row["P>|t|"]),
            }
        )
    # Metrics (ordered)
    advanced_metrics = [
        "R-squared",
        "Adjusted R-squared",
        "F-statistic",
        "No. Observations",
        "Df Residuals",
        "Df Model",
    ]
    for metric_name in advanced_metrics:
        mrow = metrics_table.loc[metrics_table["Metric"] == metric_name]
        if not mrow.empty:
            if metric_name == "F-statistic":
                row_type = "Test statistic"
                p_val = _format_p_value(f_pvalue)
            else:
                row_type = "Model statistic"
                p_val = np.nan
            rows.append(
                {
                    "Type": row_type,
                    "Information": metric_name,
                    "Measure": float(mrow["Value"].values[0]),
                    "Std Error": np.nan,
                    "P-Value": p_val,
                }
            )
    return pd.DataFrame(rows)


def _test_result_df(test_name: str, statistic: float, p_value: float) -> pd.DataFrame:
    """Output 3: Test | Test Statistic | P-Value | Heteroskedasticity."""
    is_het = p_value < ALPHA  # strict < (matches heteroskedasticity_tests.py)
    return pd.DataFrame(
        [
            {
                "Test": test_name,
                "Test Statistic": float(statistic),
                "P-Value": _format_p_value(p_value),
                "Heteroskedasticity": "True" if is_het else "False",
            }
        ]
    )


def _save_method(prefix: str, test_name: str, stat: float, p_val: float) -> None:
    save(_test_result_df(test_name, stat, p_val), f"heteroskedasticity_golden_{prefix}_test.csv")
    save(_model_summary_basic_df(), f"heteroskedasticity_golden_{prefix}_model_basic.csv")
    save(_model_summary_advanced_df(), f"heteroskedasticity_golden_{prefix}_model_advanced.csv")
    save(_predictions_df(), f"heteroskedasticity_golden_{prefix}_predictions.csv")


# ---------------------------------------------------------------------------
# Breusch-Pagan
# het_breuschpagan(resid, X_with_const) → (lm_stat, lm_pval, f_stat, f_pval)
# Uses LM statistic and LM p-value
# ---------------------------------------------------------------------------
print("\n[Breusch-Pagan]")
lm_stat_bp, lm_pval_bp, _, _ = het_breuschpagan(model.resid, X_with_const)
_save_method("bp", "Breusch-Pagan", lm_stat_bp, lm_pval_bp)

# ---------------------------------------------------------------------------
# White
# het_white(resid, X_with_const) → (lm_stat, lm_pval, f_stat, f_pval)
# Uses LM statistic and LM p-value
# ---------------------------------------------------------------------------
print("\n[White]")
lm_stat_white, lm_pval_white, _, _ = het_white(model.resid, X_with_const)
_save_method("white", "White", lm_stat_white, lm_pval_white)

# ---------------------------------------------------------------------------
# Goldfeld-Quandt
# Sort by X1 (idx = column index of X1 in X_with_const, which is +1 for const)
# het_goldfeldquandt(y, X_with_const, idx=sort_idx, split=0.33)
# Returns (F_statistic, p_value, ordering) — uses F statistic (not LM)
# ---------------------------------------------------------------------------
print("\n[Goldfeld-Quandt]")
predictor_cols = list(X_df.columns)
sort_idx = predictor_cols.index(GQ_SORT_VAR) + 1  # +1 for const at index 0
f_stat_gq, gq_pval, _ = het_goldfeldquandt(y_series, X_with_const, idx=sort_idx, split=GQ_SPLIT)
_save_method("gq", "Goldfeld-Quandt", f_stat_gq, gq_pval)

print("\nDone — heteroskedasticity golden tables written.")
