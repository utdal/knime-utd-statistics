import knime.extension as knext
import pandas as pd


def is_numeric(col: knext.Column) -> bool:
    """Helper function to filter for numeric columns."""
    return col.ktype in (knext.double(), knext.int32(), knext.int64())


def is_string(col: knext.Column) -> bool:
    """Helper function to filter for string columns."""
    return col.ktype == knext.string()


# Test type enumeration
class PostHocTestType(knext.EnumParameterOptions):
    TUKEY_HSD = (
        "Tukey HSD",
        "Honest Significant Difference - controls family-wise error rate, optimal for balanced designs.",
    )
    BONFERRONI = (
        "Holm-Bonferroni",
        "Sequential Bonferroni correction - less conservative than standard Bonferroni while controlling FWER.",
    )


# Individual parameters
test_type_param = knext.EnumParameter(
    label="Post-Hoc Test Method",
    description="Select the multiple comparison procedure to control family-wise error rate.",
    enum=PostHocTestType,
    default_value=PostHocTestType.TUKEY_HSD.name,
)

data_column_param = knext.ColumnParameter(
    label="Dependent Variable",
    description="Numeric column containing the dependent variable values.",
    column_filter=is_numeric,
)

group_column_param = knext.ColumnParameter(
    label="Grouping Variable",
    description="Categorical column containing the group assignments.",
    column_filter=is_string,
)

alpha_param = knext.DoubleParameter(
    label="Significance Level (α)",
    description="Significance level for post-hoc comparisons (default: 0.05)",
    default_value=0.05,
    min_value=0.001,
    max_value=0.999,
)


def validate_anova_prerequisite(anova_df, alpha):
    # Check for required columns
    required_cols = ["Source", "p-value"]
    missing_cols = [col for col in required_cols if col not in anova_df.columns]
    if missing_cols:
        raise ValueError(f"ANOVA table missing required columns: {missing_cols}")

    # Find the Between Groups row
    between_groups = anova_df[anova_df["Source"] == "Between Groups"]
    if between_groups.empty:
        raise ValueError("ANOVA table missing 'Between Groups' row. Ensure input is from a valid ANOVA analysis.")

    # Extract p-value
    anova_pvalue = between_groups["p-value"].iloc[0]

    # Check significance
    if anova_pvalue > alpha:
        raise ValueError(
            f"ANOVA not significant (p = {anova_pvalue:.4f} > α = {alpha}). "
            f"Post-hoc tests are not warranted when the overall ANOVA is non-significant."
        )

    return anova_pvalue


def validate_group_data(data, groups):
    # Convert to pandas for easier manipulation
    df = pd.DataFrame({"data": data, "group": groups})

    # Remove any missing values
    df_clean = df.dropna()
    if len(df_clean) < len(df):
        missing_count = len(df) - len(df_clean)
        raise ValueError(f"Found {missing_count} missing values. Post-hoc tests require complete data.")

    # Group analysis
    group_stats = df_clean.groupby("group")["data"].agg(["count", "mean", "std"]).reset_index()
    group_stats.columns = ["group", "n", "mean", "std"]

    # Check minimum number of groups
    n_groups = len(group_stats)
    if n_groups < 3:
        raise ValueError(f"Found only {n_groups} groups. Post-hoc tests require at least 3 groups to justify multiple comparisons.")

    # Check minimum sample size per group
    min_n = group_stats["n"].min()
    if min_n < 2:
        small_groups = group_stats[group_stats["n"] < 2]["group"].tolist()
        raise ValueError(
            f"Groups with insufficient sample size (n < 2): {small_groups}. "
            f"Each group must have at least 2 observations for reliable variance estimation."
        )

    # Check for constant data within groups (zero variance)
    zero_var_groups = group_stats[group_stats["std"] == 0]["group"].tolist()
    if zero_var_groups:
        raise ValueError(f"Groups with zero variance (constant values): {zero_var_groups}. Post-hoc tests require within-group variation.")

    return {
        "n_groups": n_groups,
        "total_n": len(df_clean),
        "group_stats": group_stats,
        "min_group_size": min_n,
        "max_group_size": group_stats["n"].max(),
    }
