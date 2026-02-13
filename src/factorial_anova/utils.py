"""
Factorial ANOVA Utilities.

Contains parameter definitions, enums, and helper functions for the Factorial ANOVA node.
"""

import knime.extension as knext


# =============================================================================
# Helper Functions
# =============================================================================


def is_numeric(col: knext.Column) -> bool:
    """Filter for numeric columns (double, int32, int64)."""
    return col.ktype in (knext.double(), knext.int32(), knext.int64())


def format_p_value(p) -> str:
    """
    Format p-value to avoid scientific notation.
    
    Returns 'NaN' for missing values, otherwise formats to 4 decimal places.
    """
    import pandas as pd
    
    if pd.isna(p):
        return "NaN"
    return f"{p:.4f}"


# =============================================================================
# Enum Definitions
# =============================================================================


class AnovaType(knext.EnumParameterOptions):
    """ANOVA sum of squares type options."""
    
    TYPE_I = (
        "Type I",
        "Sequential sum of squares. Results depend on the order of factors. "
        "Best for nested designs or when factor order is meaningful.",
    )
    TYPE_II = (
        "Type II",
        "Partial sum of squares. Tests each factor after accounting for other factors. "
        "Best for balanced designs without significant interactions.",
    )
    TYPE_III = (
        "Type III",
        "Marginal sum of squares. Recommended for unbalanced designs or when "
        "interactions are present. Standard in many other statistical packages.",
    )


# =============================================================================
# Parameter Definitions
# =============================================================================


# --- Core Analysis Parameters ---

response_column_param = knext.ColumnParameter(
    label="Response Variable",
    description="Numeric column containing the outcome you want to analyze.",
    column_filter=is_numeric,
)

factor_columns_param = knext.MultiColumnParameter(
    label="Factor Variables",
    description="Select one or more categorical grouping variables to test their effect on the outcome.",
)

include_interactions_param = knext.BoolParameter(
    label="Include Interaction Terms",
    description="""Test whether the effect of one factor depends on another factor.

• Checked: Include interaction effects (e.g., A + B + A:B)

• Unchecked: Main effects only (e.g., A + B)""",
    default_value=True,
)

max_interaction_order_param = knext.IntParameter(
    label="Maximum Interaction Order",
    description="""Highest-order interaction to include:

• 2: Two-way interactions (A:B) - recommended

• 3: Three-way interactions (A:B:C)

• 4: Four-way interactions (A:B:C:D)""",
    default_value=2,
    min_value=2,
    max_value=4,
)

anova_type_param = knext.EnumParameter(
    label="Sum of Squares Type",
    description=(
        "Method for calculating sum of squares. Type II is recommended for balanced data, "
        "while Type III is better for unbalanced data or interaction studies."
    ),
    enum=AnovaType,
    default_value=AnovaType.TYPE_II.name,
)

alpha_param = knext.DoubleParameter(
    label="Significance Level (α)",
    description="Factors with p-value ≤ α are considered significant (default: 0.05).",
    default_value=0.05,
    min_value=0.001,
    max_value=0.5,
)


# --- Output Format Parameter ---

advanced_output_param = knext.BoolParameter(
    label="Advanced Output",
    description="""Include detailed coefficient information in the results table.

• Unchecked: Basic summary (Factor, F-Statistic, P-Value, Conclusion)

• Checked: Advanced details with coefficients, standard errors, and T-stats""",
    default_value=False,
)