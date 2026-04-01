"""
Golden Table Generator — Repeated Measures ANOVA
==================================================
Standalone auditor script.  Does NOT import from src/.

Replicates the exact output of the 'Repeated Measures ANOVA' KNIME node
(Basic and Advanced output tables).

KNIME Node Configuration to match:
  - Dependent Variable   : Score
  - Within-Subject Factor: Time Point
  - Subject Identifier   : Subject ID
  - Alpha                : 0.05
  - Advanced Output      : False → rm_anova_golden_basic.csv
  - Advanced Output      : True  → rm_anova_golden_advanced.csv

Run from repo root:
    python golden_tables/generate_rm_anova_golden.py

Outputs written to golden_tables/data/:
    rm_anova_input.csv
    rm_anova_golden_basic.csv
    rm_anova_golden_advanced.csv
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 0.05
DV = "Score"
WITHIN = "Time Point"
SUBJECT = "Subject ID"


def save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"  Wrote {name}: {df.shape[0]} rows x {df.shape[1]} cols — {list(df.columns)}")


def _format_p_value(p) -> float:
    """Round p-value to 4 decimal places (matches repeated_measures_anova/utils.py)."""
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
# Long format: 20 subjects × 4 time points = 80 rows
# Subject baselines vary; condition effects are large → significant result
# ---------------------------------------------------------------------------
np.random.seed(42)

n_subjects = 20
time_points = ["T1", "T2", "T3", "T4"]
condition_effects = {"T1": 0.0, "T2": 5.0, "T3": 10.0, "T4": 15.0}  # strong effect
subject_baselines = np.random.normal(50.0, 10.0, n_subjects)

rows = []
for s_idx in range(n_subjects):
    sid = f"S{s_idx + 1:02d}"
    for tp in time_points:
        score = subject_baselines[s_idx] + condition_effects[tp] + np.random.normal(0.0, 0.5)
        rows.append({SUBJECT: sid, WITHIN: tp, DV: score})

df_input = pd.DataFrame(rows)
save(df_input, "rm_anova_input.csv")

# ---------------------------------------------------------------------------
# Manual RM ANOVA Computation
# Mirrors _compute_ss_components and _mauchly_sphericity in rm_anova_core.py
# ---------------------------------------------------------------------------
levels = sorted(df_input[WITHIN].unique())
subjects = df_input[SUBJECT].unique()
k = len(levels)
n = len(subjects)

# Pivot: rows = subjects, columns = time points (sorted)
pivot = df_input.pivot_table(index=SUBJECT, columns=WITHIN, values=DV, aggfunc="mean")
pivot = pivot[levels]   # enforce sorted order
X_mat = pivot.values   # (n, k)

grand_mean = X_mat.mean()
condition_means = X_mat.mean(axis=0)   # (k,)
subject_means = X_mat.mean(axis=1)     # (n,)

SS_factor = float(n * np.sum((condition_means - grand_mean) ** 2))
SS_within = float(np.sum((X_mat - subject_means[:, np.newaxis]) ** 2))
SS_error = float(SS_within - SS_factor)

df_factor = float(k - 1)
df_error = float((k - 1) * (n - 1))
MS_factor = SS_factor / df_factor
MS_error = SS_error / df_error
F_val = MS_factor / MS_error
p_unc = float(f_dist.sf(F_val, df_factor, df_error))
np2 = SS_factor / (SS_factor + SS_error)

# ---------------------------------------------------------------------------
# Mauchly's sphericity test + Greenhouse-Geisser epsilon
# Mirrors _mauchly_sphericity in rm_anova_core.py (Helmert contrast matrix)
# ---------------------------------------------------------------------------
if k <= 2:
    mauchly_W, mauchly_p, eps = 1.0, 1.0, 1.0
else:
    # Helmert-style orthonormal contrast matrix C: (k, k-1)
    C = np.zeros((k, k - 1))
    for j in range(k - 1):
        C[: j + 1, j] = 1.0 / (j + 1)
        C[j + 1, j] = -1.0
        C[:, j] *= np.sqrt((j + 1) / (j + 2))

    Y = X_mat @ C          # (n, k-1)
    pp = k - 1
    S = np.cov(Y, rowvar=False, ddof=1)   # (pp, pp)

    det_S = np.linalg.det(S)
    trace_S = np.trace(S)

    W_raw = det_S / ((trace_S / pp) ** pp) if trace_S > 0 else 0.0
    mauchly_W = float(max(0.0, min(1.0, W_raw)))

    f_corr = (2.0 * pp * pp + pp + 2.0) / (6.0 * pp * (n - 1))
    chi2_stat = -((n - 1) - f_corr) * np.log(max(mauchly_W, 1e-300))
    chi2_df = pp * (pp + 1) / 2 - 1
    mauchly_p = float(chi2_dist.sf(chi2_stat, chi2_df)) if chi2_df > 0 else 1.0

    eps_num = np.trace(S) ** 2
    eps_den = pp * np.sum(S**2)
    eps_raw = float(eps_num / eps_den) if eps_den > 0 else 1.0
    eps = float(max(1.0 / pp, min(1.0, eps_raw)))

# Greenhouse-Geisser corrected p-value
df_gg_num = df_factor * eps
df_gg_den = df_error * eps
p_gg_corr = float(f_dist.sf(F_val, df_gg_num, df_gg_den))

# Significance decision (mirrors run_rm_anova: uses p_gg_corr; strict <)
p_decision = p_gg_corr if not np.isnan(p_gg_corr) else p_unc
is_significant = (not np.isnan(p_decision)) and (p_decision < ALPHA)
conclusion = "Significant" if is_significant else "Not Significant"

# ---------------------------------------------------------------------------
# Basic Output
# Columns: Source | P-Value (Greenhouse-Geisser Corrected) |
#           Effect Size (Partial Eta Squared) | Conclusion
# ---------------------------------------------------------------------------
df_basic = pd.DataFrame(
    [
        {
            "Source": WITHIN,
            "P-Value (Greenhouse-Geisser Corrected)": _format_p_value(p_gg_corr),
            "Effect Size (Partial Eta Squared)": float(np2),
            "Conclusion": conclusion,
        }
    ]
)
save(df_basic, "rm_anova_golden_basic.csv")

# ---------------------------------------------------------------------------
# Advanced Output (2 rows: factor + error)
# Columns: Source | Sum of Squares | Degrees of Freedom | Mean Square |
#          F Statistic | P-Value (Uncorrected) |
#          P-Value (Greenhouse-Geisser Corrected) |
#          Mauchly's W | Mauchly's P-Value | Epsilon (Greenhouse-Geisser) |
#          Effect Size (Partial Eta Squared) | Conclusion
# ---------------------------------------------------------------------------
adv_rows = [
    # Factor row
    {
        "Source": WITHIN,
        "Sum of Squares": float(SS_factor),
        "Degrees of Freedom": float(df_factor),
        "Mean Square": float(MS_factor),
        "F Statistic": float(F_val),
        "P-Value (Uncorrected)": _format_p_value(p_unc),
        "P-Value (Greenhouse-Geisser Corrected)": _format_p_value(p_gg_corr),
        "Mauchly's W": float(mauchly_W),
        "Mauchly's P-Value": _format_p_value(mauchly_p),
        "Epsilon (Greenhouse-Geisser)": float(eps),
        "Effect Size (Partial Eta Squared)": float(np2),
        "Conclusion": conclusion,
    },
    # Error row — sphericity / significance fields are NaN; Conclusion is ""
    {
        "Source": "Error",
        "Sum of Squares": float(SS_error),
        "Degrees of Freedom": float(df_error),
        "Mean Square": float(MS_error),
        "F Statistic": np.nan,
        "P-Value (Uncorrected)": np.nan,
        "P-Value (Greenhouse-Geisser Corrected)": np.nan,
        "Mauchly's W": np.nan,
        "Mauchly's P-Value": np.nan,
        "Epsilon (Greenhouse-Geisser)": np.nan,
        "Effect Size (Partial Eta Squared)": np.nan,
        "Conclusion": "",
    },
]
df_advanced = pd.DataFrame(adv_rows)
save(df_advanced, "rm_anova_golden_advanced.csv")

print("\nDone — RM ANOVA golden tables written.")
