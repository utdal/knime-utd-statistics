"""
Utility functions and parameters for the MANOVA node.

Provides parameter definitions, data validation, and column-type helpers
following the same conventions used across the UTD Statistics extension.
"""

import knime.extension as knext
import pandas as pd


def is_numeric(col: knext.Column) -> bool:
    """Helper function to filter for numeric columns."""
    return col.ktype in (knext.double(), knext.int32(), knext.int64())


def is_string(col: knext.Column) -> bool:
    """Helper function to filter for string columns."""
    return col.ktype == knext.string()


def format_p_value(p):
    """Format p-values for readable KNIME output (avoid scientific notation)."""
    if pd.isna(p) or p == "?":
        return "?"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}"


# ---------------------------------------------------------------------------
# Node parameters
# ---------------------------------------------------------------------------

dependent_columns_param = knext.MultiColumnParameter(
    label="Dependent Variables (Y)",
    description=(
        "Select two or more numeric columns as dependent variables for the "
        "multivariate analysis.  MANOVA tests whether group means differ "
        "across these variables simultaneously."
    ),
    column_filter=is_numeric,
)

group_column_param = knext.ColumnParameter(
    label="Grouping Variable",
    description="Categorical column containing the group assignments (independent variable).",
    column_filter=is_string,
)

alpha_param = knext.DoubleParameter(
    label="Significance Level (α)",
    description="Significance level for the MANOVA test (default: 0.05).",
    default_value=0.05,
    min_value=0.001,
    max_value=0.999,
)

advanced_stats_param = knext.BoolParameter(
    label="Include Advanced Statistics",
    description=(
        "Basic: Shows a simple summary with the factor name, "
        "Pillai's Trace p-value, and a Significant / Not Significant conclusion.\n\n"
        "Advanced: Shows the full multivariate test table with "
        "Pillai's Trace value, numerator / denominator degrees of freedom, "
        "F-value, and p-value."
    ),
    default_value=False,
)


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------


def validate_manova_data(df, dep_vars, group_col):
    """
    Validate input data for MANOVA analysis requirements.

    The checks mirror the ``validate_anova_data`` pattern used elsewhere in
    this extension: column existence → missing values → minimum groups →
    minimum group size → zero-variance check → singularity check.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    dep_vars : list of str
        Names of numeric dependent variable columns.
    group_col : str
        Name of the categorical grouping variable column.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame containing only the required columns with no
        missing values.

    Raises
    ------
    ValueError
        If the data does not meet any MANOVA requirement.
    """

    # ------------------------------------------------------------------
    # 1. Column existence
    # ------------------------------------------------------------------
    all_cols = list(dep_vars) + [group_col]
    missing_cols = [c for c in all_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in data: {', '.join(missing_cols)}")

    # ------------------------------------------------------------------
    # 2. Minimum dependent variables
    # ------------------------------------------------------------------
    if len(dep_vars) < 2:
        raise ValueError(
            f"MANOVA requires at least 2 dependent variables, but only "
            f"{len(dep_vars)} selected.  For a single dependent variable, "
            "use the ANOVA node instead."
        )

    # ------------------------------------------------------------------
    # 3. Missing values
    # ------------------------------------------------------------------
    df_sub = df[all_cols].copy()
    n_before = len(df_sub)
    df_clean = df_sub.dropna()
    n_after = len(df_clean)

    if n_after < n_before:
        n_dropped = n_before - n_after
        raise ValueError(f"Found {n_dropped} rows with missing values. MANOVA requires complete data with no missing values.")

    # ------------------------------------------------------------------
    # 4. Minimum number of groups
    # ------------------------------------------------------------------
    unique_groups = df_clean[group_col].unique()
    n_groups = len(unique_groups)
    if n_groups < 2:
        raise ValueError(f"Found only {n_groups} group(s). MANOVA requires at least 2 groups for comparison.")

    # ------------------------------------------------------------------
    # 5. Minimum sample size per group
    # ------------------------------------------------------------------
    group_sizes = df_clean.groupby(group_col).size()
    min_n = group_sizes.min()
    if min_n < 2:
        small = group_sizes[group_sizes < 2].index.tolist()
        raise ValueError(f"Groups with insufficient sample size (n < 2): {small}. Each group must have at least 2 observations for MANOVA.")

    # ------------------------------------------------------------------
    # 6. Zero-variance check (per variable per group)
    # ------------------------------------------------------------------
    for var in dep_vars:
        group_stds = df_clean.groupby(group_col)[var].std()
        zero_var = group_stds[group_stds == 0].index.tolist()
        if zero_var:
            raise ValueError(
                f"Variable '{var}' has zero variance (constant values) in "
                f"groups: {zero_var}. MANOVA requires within-group variation "
                "for all dependent variables."
            )

    return df_clean
