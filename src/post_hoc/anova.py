"""
One-way ANOVA implementation using statsmodels.

This module provides one-way ANOVA analysis with output format matching
the predefined KNIME table structure for seamless integration with
post-hoc multiple comparison tests.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings


def run_one_way_anova(data, groups, alpha=0.05):
    """
    Perform one-way ANOVA using statsmodels.

    Parameters:
    -----------
    data : array-like
        Numeric dependent variable values
    groups : array-like
        Group assignment labels
    alpha : float, default=0.05
        Significance level for the test

    Returns:
    --------
    dict
        Dictionary containing:
        - 'test': str, test name
        - 'alpha': float, significance level used
        - 'anova_table': pd.DataFrame, ANOVA results in predefined format
        - 'summary': dict, test summary statistics
        - 'significant': bool, whether ANOVA is significant
    """

    # Convert inputs to numpy arrays for consistency
    data = np.asarray(data)
    groups = np.asarray(groups)

    # Remove any missing values
    mask = ~(pd.isna(data) | pd.isna(groups))
    data_clean = data[mask]
    groups_clean = groups[mask]

    # Create DataFrame for statsmodels
    df = pd.DataFrame({"value": data_clean, "group": groups_clean})

    # Fit OLS model for ANOVA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ols("value ~ C(group)", data=df).fit()
        anova_results = anova_lm(model, typ=2)

    # Extract ANOVA components
    unique_groups = np.unique(groups_clean)
    n_groups = len(unique_groups)
    total_n = len(data_clean)

    # Extract values from ANOVA results (statsmodels only provides: sum_sq, df, F, PR(>F))
    ss_between = anova_results.loc["C(group)", "sum_sq"]
    ss_within = anova_results.loc["Residual", "sum_sq"]
    ss_total = ss_between + ss_within

    df_between = int(anova_results.loc["C(group)", "df"])
    df_within = int(anova_results.loc["Residual", "df"])
    df_total = df_between + df_within

    # Calculate Mean Squares manually (not provided by statsmodels)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    # Extract F-statistic and p-value
    f_statistic = anova_results.loc["C(group)", "F"]
    p_value = anova_results.loc["C(group)", "PR(>F)"]

    # Create ANOVA table in predefined format
    anova_table = pd.DataFrame(
        {
            "Source": ["Between Groups", "Within Groups", "Total"],
            "Sum of Squares": [ss_between, ss_within, ss_total],
            "df": [df_between, df_within, df_total],
            "Mean Square": [ms_between, ms_within, np.nan],  # Total row doesn't have Mean Square
            "F": [f_statistic, np.nan, np.nan],  # Only Between Groups has F-statistic
            "p-value": [p_value, np.nan, np.nan],  # Only Between Groups has p-value
        }
    )

    # Round numerical values for better presentation
    anova_table["Sum of Squares"] = anova_table["Sum of Squares"].round(6)
    anova_table["Mean Square"] = anova_table["Mean Square"].round(6)
    anova_table["F"] = anova_table["F"].round(6)
    anova_table["p-value"] = anova_table["p-value"].round(8)

    # Determine significance
    is_significant = p_value <= alpha

    # Calculate effect size (eta-squared)
    eta_squared = ss_between / ss_total

    # Create summary statistics
    summary = {
        "n_groups": n_groups,
        "total_n": total_n,
        "f_statistic": f_statistic,
        "p_value": p_value,
        "alpha": alpha,
        "significant": is_significant,
        "eta_squared": eta_squared,
        "method": "One-Way ANOVA",
        "model_formula": "value ~ C(group)",
    }

    return {
        "test": "One-Way ANOVA",
        "alpha": alpha,
        "anova_table": anova_table,
        "summary": summary,
        "significant": is_significant,
    }


def format_anova_results_for_knime(anova_output, data_column_name="Data", group_column_name="Group"):
    """
    Format ANOVA results for KNIME output table.

    Parameters:
    -----------
    anova_output : dict
        Output from run_one_way_anova function
    data_column_name : str, default="Data"
        Name of the dependent variable column for context
    group_column_name : str, default="Group"
        Name of the grouping variable column for context

    Returns:
    --------
    pd.DataFrame
        ANOVA results table formatted for KNIME with additional context columns
    """

    # Get the core ANOVA table
    anova_df = anova_output["anova_table"].copy()

    # Add context columns for KNIME
    anova_df.insert(0, "Test Column", data_column_name)
    anova_df.insert(0, "ID", range(len(anova_df)))

    # Add method and significance information
    anova_df["Test Method"] = anova_output["test"]
    anova_df["Significance Level"] = anova_output["alpha"]
    anova_df["Significant"] = str(anova_output["significant"])

    # Ensure proper data types for KNIME
    anova_df["ID"] = anova_df["ID"].astype("int32")  # Use int32 for KNIME
    anova_df["Sum of Squares"] = anova_df["Sum of Squares"].astype(float)
    anova_df["df"] = anova_df["df"].astype(float)  # Keep as float to allow NaN values
    anova_df["Mean Square"] = anova_df["Mean Square"].astype(float)
    anova_df["F"] = anova_df["F"].astype(float)
    anova_df["p-value"] = anova_df["p-value"].astype(float)

    return anova_df


def validate_anova_data(data, groups):
    """
    Validate data for ANOVA analysis requirements.

    Parameters:
    -----------
    data : array-like
        Numeric dependent variable values
    groups : array-like
        Group assignment labels

    Returns:
    --------
    dict
        Validation results and group statistics

    Raises:
    -------
    ValueError
        If data doesn't meet ANOVA requirements
    """
    # Convert to pandas for easier manipulation
    df = pd.DataFrame({"data": data, "group": groups})

    # Remove any missing values
    df_clean = df.dropna()
    if len(df_clean) < len(df):
        missing_count = len(df) - len(df_clean)
        raise ValueError(f"Found {missing_count} missing values. ANOVA requires complete data.")

    # Group analysis
    group_stats = df_clean.groupby("group")["data"].agg(["count", "mean", "std"]).reset_index()
    group_stats.columns = ["group", "n", "mean", "std"]

    # Check minimum number of groups
    n_groups = len(group_stats)
    if n_groups < 2:
        raise ValueError(f"Found only {n_groups} group(s). ANOVA requires at least 2 groups for comparison.")

    # Check minimum sample size per group (for reliable ANOVA)
    min_n = group_stats["n"].min()
    if min_n < 2:
        small_groups = group_stats[group_stats["n"] < 2]["group"].tolist()
        raise ValueError(f"Groups with insufficient sample size (n < 2): {small_groups}. Each group must have at least 2 observations for ANOVA.")

    # Check for constant data within groups (zero variance)
    zero_var_groups = group_stats[group_stats["std"] == 0]["group"].tolist()
    if zero_var_groups:
        raise ValueError(f"Groups with zero variance (constant values): {zero_var_groups}. ANOVA requires within-group variation.")

    return {
        "n_groups": n_groups,
        "total_n": len(df_clean),
        "group_stats": group_stats,
        "min_group_size": min_n,
        "max_group_size": group_stats["n"].max(),
    }
