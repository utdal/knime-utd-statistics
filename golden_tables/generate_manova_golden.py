"""
Golden Table Generator — One-Way MANOVA
========================================
Standalone auditor script.  Does NOT import from src/.

Replicates the exact output of the 'One-Way MANOVA' KNIME node
(Multivariate Results Basic, Multivariate Results Advanced, and Reliability Report).

KNIME Node Configuration to match:
  - Dependent Columns : DV1, DV2
  - Group Column      : Group
  - Alpha             : 0.05
  - Advanced Stats    : False → manova_golden_multivariate_basic.csv
  - Advanced Stats    : True  → manova_golden_multivariate_advanced.csv
  - Reliability Report always → manova_golden_reliability.csv

Run from repo root:
    python golden_tables/generate_manova_golden.py

Outputs written to golden_tables/data/:
    manova_input.csv
    manova_golden_multivariate_basic.csv
    manova_golden_multivariate_advanced.csv
    manova_golden_reliability.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.multivariate.manova import MANOVA

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 0.05
DEP_VARS = ["DV1", "DV2"]
GROUP_COL = "Group"


def save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"  Wrote {name}: {df.shape[0]} rows x {df.shape[1]} cols — {list(df.columns)}")


# ---------------------------------------------------------------------------
# Synthetic Input Data  (seed 42 → reproducible)
# 3 groups × 30 obs, well-separated means → MANOVA will be significant
# ---------------------------------------------------------------------------
np.random.seed(42)

group_means = {"A": (10.0, 5.0), "B": (15.0, 10.0), "C": (20.0, 15.0)}
rows = []
for grp, (mu1, mu2) in group_means.items():
    dv1 = np.random.normal(mu1, 1.0, 30)
    dv2 = np.random.normal(mu2, 1.0, 30)
    for i in range(30):
        rows.append({"DV1": dv1[i], "DV2": dv2[i], "Group": grp})

df_input = pd.DataFrame(rows)
save(df_input, "manova_input.csv")

# ---------------------------------------------------------------------------
# MANOVA via Pillai's Trace  — statsmodels.multivariate.manova.MANOVA
# Matches run_manova() in manova_core.py exactly:
#   - formula  : Q("DV1") + Q("DV2") ~ C(Q("Group"))
#   - group col cast to object dtype
#   - extract Pillai's trace row from result
#   - rounding: pillai/f_value → round(x, 6), p_value → round(x, 8)
# ---------------------------------------------------------------------------
df_manova = df_input.copy()
df_manova[GROUP_COL] = df_manova[GROUP_COL].astype(object)

dep_formula = " + ".join(f'Q("{v}")' for v in DEP_VARS)
formula = f'{dep_formula} ~ C(Q("{GROUP_COL}"))'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    maov = MANOVA.from_formula(formula, data=df_manova)
    result = maov.mv_test()

factor_key = f'C(Q("{GROUP_COL}"))'
stat_df = result.results[factor_key]["stat"]
pillai_row = stat_df.loc["Pillai's trace"]

pillai_value = round(float(pillai_row["Value"]), 6)
num_df = float(pillai_row["Num DF"])
den_df = float(pillai_row["Den DF"])
f_value = round(float(pillai_row["F Value"]), 6)
p_value = round(float(pillai_row["Pr > F"]), 8)
is_significant = p_value <= ALPHA


# Basic output: Factor | P-Value | Conclusion
df_basic = pd.DataFrame(
    {
        "Factor": [GROUP_COL],
        "P-Value": [p_value],
        "Conclusion": ["Significant" if is_significant else "Not Significant"],
    }
)
save(df_basic, "manova_golden_multivariate_basic.csv")

# Advanced output: Source | Statistic | Numerator Df | Denominator Df | F-Value | P-Value
df_advanced = pd.DataFrame(
    {
        "Source": [GROUP_COL],
        "Statistic": [pillai_value],
        "Numerator Df": [num_df],
        "Denominator Df": [den_df],
        "F-Value": [f_value],
        "P-Value": [p_value],
    }
)
save(df_advanced, "manova_golden_multivariate_advanced.csv")


# ---------------------------------------------------------------------------
# Box's M Test  — manual computation matching box_m.py exactly
# rounding: statistic/chi2_approx → round(x, 6), p_value → round(x, 8)
# Status: "Warning" if p_value <= 0.001 else "Pass"
# ---------------------------------------------------------------------------
unique_groups = df_input[GROUP_COL].unique()
g = len(unique_groups)
p = len(DEP_VARS)

n_list = []
S_list = []
for grp in unique_groups:
    grp_data = df_input.loc[df_input[GROUP_COL] == grp, DEP_VARS].values
    n_list.append(len(grp_data))
    S_list.append(np.cov(grp_data.T, ddof=1))

n_arr = np.array(n_list, dtype=float)
N = np.sum(n_arr)

# Pooled covariance matrix
S_pooled = np.zeros((p, p))
for i in range(g):
    S_pooled += (n_arr[i] - 1) * S_list[i]
S_pooled /= N - g

# Box's M statistic
sign_pooled, ln_det_pooled = np.linalg.slogdet(S_pooled)
M = (N - g) * ln_det_pooled
for i in range(g):
    _, ln_det_i = np.linalg.slogdet(S_list[i])
    M -= (n_arr[i] - 1) * ln_det_i

# Chi-square approximation
sum_inv = np.sum(1.0 / (n_arr - 1)) - 1.0 / (N - g)
c1 = ((2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))) * sum_inv
df_chi2 = p * (p + 1) * (g - 1) / 2.0
chi2_approx = M * (1 - c1)
box_m_pvalue = 1.0 - stats.chi2.cdf(chi2_approx, df_chi2)

statistic_r = round(float(M), 6)
chi2_r = round(float(chi2_approx), 6)
pval_r = round(float(box_m_pvalue), 8)
status = "Warning" if pval_r <= 0.001 else "Pass"

df_reliability = pd.DataFrame(
    {
        "Test": ["Box's M"],
        "Statistic": [statistic_r],
        "Chi-Square Approx": [chi2_r],
        "Degrees of Freedom": [float(df_chi2)],
        "P-Value": [pval_r],
        "Status": [status],
    }
)
save(df_reliability, "manova_golden_reliability.csv")

print("\nDone — MANOVA golden tables written.")
