from typing import Tuple
import itertools

import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats


def run_tukey_test(data, groups, alpha: float = 0.05) -> dict:
    # High-level: run Tukey's HSD (via statsmodels) on cleaned input arrays,
    # compute raw Welch t-test diagnostics for each pair (t-statistic and raw p),
    # assemble pairwise and group-summary DataFrames and return as a dict.
    data = np.asarray(data)
    groups = np.asarray(groups)

    # drop missing
    mask = ~(pd.isna(data) | pd.isna(groups))
    data_clean = data[mask]
    groups_clean = groups[mask]

    # If less than two groups or no data, return empty structures
    unique_groups = np.unique(groups_clean)
    if len(unique_groups) < 2 or len(data_clean) == 0:
        empty_cols = [
            "Comparison",
            "Group 1",
            "Group 2",
            "Mean Difference",
            "Lower CI",
            "Upper CI",
            "P-Value",
            "T-Statistic",
            "Raw P-Value",
            "Corrected P-Value",
            "Reject H0",
        ]
        results_df = pd.DataFrame(columns=empty_cols)
        summary = {
            "n_groups": int(len(unique_groups)),
            "n_comparisons": 0,
            "n_significant": 0,
            "family_wise_error_rate": alpha,
            "method": "Tukey HSD (Honest Significant Difference)",
            "group_summary": pd.DataFrame({"Group": unique_groups, "N": [], "Mean": [], "Std Dev": []}),
        }
        return {"test": "Tukey HSD", "alpha": alpha, "n_comparisons": 0, "results": results_df, "summary": summary}

    # Run Tukey HSD
    tukey_result = pairwise_tukeyhsd(data_clean, groups_clean, alpha=alpha)

    groupsunique = tukey_result.groupsunique
    # pairwise_tukeyhsd orders pairs in the same natural order as combinations of groupsunique
    group_pairs = list(itertools.combinations(groupsunique, 2))

    # Compute raw p-values (Welch t-test) and t-statistics for diagnostics
    raw_pvalues = []
    t_stats = []
    for g1, g2 in group_pairs:
        data1 = data_clean[groups_clean == g1]
        data2 = data_clean[groups_clean == g2]
        if len(data1) == 0 or len(data2) == 0:
            t_stats.append(np.nan)
            raw_pvalues.append(np.nan)
        else:
            t_stat, p_raw = stats.ttest_ind(data1, data2, equal_var=False)
            t_stats.append(float(t_stat))
            raw_pvalues.append(float(p_raw))

    # Build DataFrame ensuring alignment with tukey_result arrays
    corrected_p = np.asarray(tukey_result.pvalues, dtype=float)
    mean_diffs = np.asarray(tukey_result.meandiffs, dtype=float)
    confint = np.asarray(tukey_result.confint, dtype=float)
    lower_ci = confint[:, 0] if confint.size else np.array([])
    upper_ci = confint[:, 1] if confint.size else np.array([])
    rejects = np.asarray(tukey_result.reject, dtype=bool)

    results_df = pd.DataFrame(
        {
            "Group 1": [pair[0] for pair in group_pairs],
            "Group 2": [pair[1] for pair in group_pairs],
            "Mean Difference": mean_diffs,
            "Lower CI": lower_ci,
            "Upper CI": upper_ci,
            # keep 'P-Value' as the raw p-value for diagnostics (consistent with other modules)
            "P-Value": raw_pvalues,
            "T-Statistic": t_stats,
            "Raw P-Value": raw_pvalues,
            "Corrected P-Value": corrected_p,
            "Reject H0": rejects,
        }
    )

    results_df["Comparison"] = results_df["Group 1"].astype(str) + " vs " + results_df["Group 2"].astype(str)

    # Summary
    n_groups = int(len(groupsunique))
    n_comparisons = int(len(results_df))
    n_significant = int(np.nansum(results_df["Reject H0"].astype(int)))

    group_summary = pd.DataFrame(
        {
            "Group": groupsunique,
            "N": [int(np.sum(groups_clean == g)) for g in groupsunique],
            "Mean": [float(np.mean(data_clean[groups_clean == g])) for g in groupsunique],
            "Std Dev": [
                float(np.std(data_clean[groups_clean == g], ddof=1)) if np.sum(groups_clean == g) > 1 else float("nan") for g in groupsunique
            ],
        }
    )

    summary = {
        "n_groups": n_groups,
        "n_comparisons": n_comparisons,
        "n_significant": n_significant,
        "family_wise_error_rate": alpha,
        "method": "Tukey HSD (Honest Significant Difference)",
        "group_summary": group_summary,
    }

    # Reorder columns to a stable canonical order
    col_order = [
        "Comparison",
        "Group 1",
        "Group 2",
        "Mean Difference",
        "Lower CI",
        "Upper CI",
        "P-Value",
        "T-Statistic",
        "Raw P-Value",
        "Corrected P-Value",
        "Reject H0",
    ]
    results_df = results_df[col_order]

    return {"test": "Tukey HSD", "alpha": alpha, "n_comparisons": n_comparisons, "results": results_df, "summary": summary}


def format_tukey_results_for_knime(tukey_output: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # High-level: take the dictionary output from run_tukey_test and produce
    # two pandas DataFrames formatted for KNIME: pairwise comparisons and group summary.
    pairwise_df = tukey_output["results"].copy()

    pairwise_df["Test Method"] = tukey_output.get("test", "Tukey HSD")
    pairwise_df["Significance Level"] = tukey_output.get("alpha", np.nan)

    # Guarantee the presence of diagnostic columns
    if "Raw P-Value" not in pairwise_df.columns and "P-Value" in pairwise_df.columns:
        pairwise_df["Raw P-Value"] = pairwise_df["P-Value"]
    if "Corrected P-Value" not in pairwise_df.columns:
        pairwise_df["Corrected P-Value"] = np.nan

    # Ensure expected column order
    expected = [
        "Comparison",
        "Group 1",
        "Group 2",
        "Mean Difference",
        "Lower CI",
        "Upper CI",
        "P-Value",
        "T-Statistic",
        "Raw P-Value",
        "Corrected P-Value",
        "Reject H0",
        "Test Method",
        "Significance Level",
    ]
    # Add any missing expected columns as NaN so reindex works
    for c in expected:
        if c not in pairwise_df.columns:
            pairwise_df[c] = np.nan

    pairwise_df = pairwise_df[expected]

    # Coerce types safely
    float_cols = ["Mean Difference", "Lower CI", "Upper CI", "P-Value", "T-Statistic", "Raw P-Value", "Corrected P-Value"]
    for c in float_cols:
        pairwise_df[c] = pd.to_numeric(pairwise_df[c], errors="coerce").astype(float)

    # Keep Reject H0 as boolean where possible
    if pairwise_df["Reject H0"].dtype == object:
        # coerce strings like 'True'/'False' to boolean
        pairwise_df["Reject H0"] = pairwise_df["Reject H0"].map(
            lambda v: bool(v) if pd.notna(v) and str(v).lower() in ("true", "1") else (False if pd.notna(v) else np.nan)
        )
    pairwise_df["Reject H0"] = pairwise_df["Reject H0"].astype("boolean")

    # Prepare group summary
    group_summary_df = tukey_output["summary"]["group_summary"].copy()
    group_summary_df["Test Method"] = tukey_output.get("test", "Tukey HSD")
    group_summary_df["Significance Level"] = tukey_output.get("alpha", np.nan)
    # Ensure numeric columns
    for c in ("N", "Mean", "Std Dev"):
        if c in group_summary_df.columns:
            group_summary_df[c] = pd.to_numeric(group_summary_df[c], errors="coerce").astype(float)

    return pairwise_df, group_summary_df
