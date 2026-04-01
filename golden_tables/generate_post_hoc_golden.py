"""
Golden Table Generator — Post-Hoc Analysis
==========================================
Standalone auditor script.  Does NOT import from src/.

Replicates the exact output of the 'Post-Hoc Analysis' KNIME node
for both available methods (Tukey HSD and Holm-Bonferroni).

KNIME Node Configuration to match:
  - Alpha          : 0.05
  - Data Column    : Score
  - Group Column   : Group

  For Tukey HSD run:    set Test Method = "Tukey HSD"
  For Holm-Bonferroni:  set Test Method = "Holm-Bonferroni"

Run from repo root:
    python golden_tables/generate_post_hoc_golden.py

Outputs written to golden_tables/data/:
    post_hoc_input.csv
    post_hoc_golden_tukey_summary.csv
    post_hoc_golden_tukey_pairwise.csv
    post_hoc_golden_bonferroni_summary.csv
    post_hoc_golden_bonferroni_pairwise.csv
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALPHA = 0.05
DATA_COL = "Score"
GROUP_COL = "Group"


def save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"  Wrote {name}: {df.shape[0]} rows x {df.shape[1]} cols — {list(df.columns)}")


# ---------------------------------------------------------------------------
# Synthetic Input Data  (seed 42 → reproducible)
# Three balanced groups with clearly separated means → ANOVA will be significant
# ---------------------------------------------------------------------------
np.random.seed(42)
scores_a = np.random.normal(10, 1, 20)
scores_b = np.random.normal(15, 1, 20)
scores_c = np.random.normal(20, 1, 20)

df_input = pd.DataFrame(
    {
        DATA_COL: np.concatenate([scores_a, scores_b, scores_c]),
        GROUP_COL: ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
    }
)
save(df_input, "post_hoc_input.csv")

# ---------------------------------------------------------------------------
# One-Way ANOVA  (shared between both methods)
# scipy.stats.f_oneway — no rounding applied to ANOVA p-value
# ---------------------------------------------------------------------------
group_data = {
    lbl: df_input.loc[df_input[GROUP_COL] == lbl, DATA_COL].values
    for lbl in sorted(df_input[GROUP_COL].unique())
}
_, anova_p = f_oneway(*group_data.values())
is_significant = anova_p <= ALPHA


def _anova_summary_df() -> pd.DataFrame:
    """Single-row ANOVA summary (same schema for both methods)."""
    return pd.DataFrame(
        {
            "Tested Variable": [DATA_COL],
            "Grouping Variable": [GROUP_COL],
            "Significance Level": [float(ALPHA)],
            "ANOVA p-Value": [float(anova_p)],
            "Overall Conclusion": ["Significant Difference Found" if is_significant else "No Difference Found"],
        }
    )


def _fallback_pairwise_df() -> pd.DataFrame:
    """Single fallback row when ANOVA is not significant."""
    return pd.DataFrame(
        {
            "Comparison": [f"ANOVA not significant (p = {anova_p:.3f}). Comparisons were skipped."],
            "Post-Hoc Method": ["N/A"],
            "Mean Difference": [np.nan],
            "Corrected p-Value": [np.nan],
            "Difference is Significant?": ["N/A"],
        }
    )


# ---------------------------------------------------------------------------
# Tukey HSD
# statsmodels.stats.multicomp.pairwise_tukeyhsd
# Pair order follows itertools.combinations(groupsunique, 2)
# Node rounding: NONE on p-values
# ---------------------------------------------------------------------------
def tukey_pairwise_df() -> pd.DataFrame:
    if not is_significant:
        return _fallback_pairwise_df()

    endog = df_input[DATA_COL].values
    groups = df_input[GROUP_COL].values
    tukey = pairwise_tukeyhsd(endog, groups, alpha=ALPHA)
    groupsunique = tukey.groupsunique

    rows = []
    for idx, (i, j) in enumerate(itertools.combinations(range(len(groupsunique)), 2)):
        g1 = str(groupsunique[i])
        g2 = str(groupsunique[j])
        mean_diff = float(tukey.meandiffs[idx])
        corr_p = float(tukey.pvalues[idx])
        rows.append(
            {
                "Comparison": f"{g1} vs {g2}",
                "Post-Hoc Method": "Tukey HSD",
                "Mean Difference": mean_diff,
                "Corrected p-Value": corr_p,
                "Difference is Significant?": "Yes" if corr_p <= ALPHA else "No",
            }
        )
    return pd.DataFrame(rows)


save(_anova_summary_df(), "post_hoc_golden_tukey_summary.csv")
save(tukey_pairwise_df(), "post_hoc_golden_tukey_pairwise.csv")


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# Per-pair scipy.stats.ttest_ind (equal_var=False) + multipletests(method='holm')
# Pair order follows np.unique(groups) with i < j
# Node rounding: NONE on p-values
# ---------------------------------------------------------------------------
def bonferroni_pairwise_df() -> pd.DataFrame:
    if not is_significant:
        return _fallback_pairwise_df()

    unique_groups = np.unique(df_input[GROUP_COL].values)
    pairs = [(i, j) for i in range(len(unique_groups)) for j in range(i + 1, len(unique_groups))]

    raw_pvals, mean_diffs, comparisons = [], [], []
    for i, j in pairs:
        g1 = df_input.loc[df_input[GROUP_COL] == unique_groups[i], DATA_COL].values
        g2 = df_input.loc[df_input[GROUP_COL] == unique_groups[j], DATA_COL].values
        _, raw_p = ttest_ind(g1, g2, equal_var=False)
        raw_pvals.append(float(raw_p))
        mean_diffs.append(float(g1.mean() - g2.mean()))
        comparisons.append(f"{unique_groups[i]} vs {unique_groups[j]}")

    _, corrected_pvals, _, _ = multipletests(raw_pvals, alpha=ALPHA, method="holm")

    rows = []
    for k in range(len(pairs)):
        corr_p = float(corrected_pvals[k])
        rows.append(
            {
                "Comparison": comparisons[k],
                "Post-Hoc Method": "Holm-Bonferroni",
                "Mean Difference": mean_diffs[k],
                "Corrected p-Value": corr_p,
                "Difference is Significant?": "Yes" if corr_p <= ALPHA else "No",
            }
        )
    return pd.DataFrame(rows)


save(_anova_summary_df(), "post_hoc_golden_bonferroni_summary.csv")
save(bonferroni_pairwise_df(), "post_hoc_golden_bonferroni_pairwise.csv")

print("\nDone — post-hoc golden tables written.")
