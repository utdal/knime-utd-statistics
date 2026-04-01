"""
Factorial ANOVA Utilities.

Contains parameter definitions, enums, and helper functions for the Factorial ANOVA node.
"""

import knime.extension as knext

from src.utils import alpha_param, format_p_value, is_categorical, is_numeric

# =============================================================================
# Enum Definitions
# =============================================================================


class AnovaType(knext.EnumParameterOptions):
    """ANOVA sum of squares type options."""

    TYPE_I = (
        "Type I",
        "Sequential sum of squares. Results depend on the order of factors. Best for nested designs or when factor order is meaningful.",
    )
    TYPE_II = (
        "Type II",
        "Partial sum of squares. Tests each factor after accounting for other factors. Best for balanced designs without significant interactions.",
    )
    TYPE_III = (
        "Type III",
        "Marginal sum of squares. Recommended for unbalanced designs or when interactions are present. Standard in many other statistical packages.",
    )


# =============================================================================
# Parameter Definitions
# =============================================================================


# --- Core Analysis Parameters ---

response_column_param = knext.ColumnParameter(
    label="Dependent Variable",
    description="Numeric column containing the outcome you want to analyze.",
    column_filter=is_numeric,
)

factor_columns_param = knext.MultiColumnParameter(
    label="Factor Variables",
    description="Select one or more categorical grouping variables to test their effect on the outcome.",
    column_filter=is_categorical,
)

include_interactions_param = knext.BoolParameter(
    label="Include Interaction Terms",
    description="""Test whether the effect of one factor depends on another factor.

• Checked: Include interaction effects (e.g., A + B + A:B)

• Unchecked: Main effects only (e.g., A + B)""",
    default_value=True,
    is_advanced=True,
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
    is_advanced=True,
)

anova_type_param = knext.EnumParameter(
    label="Sum of Squares Type",
    description="Method used to partition variance between factors. Defaults to Type III.",
    enum=AnovaType,
    default_value=AnovaType.TYPE_III.name,
    is_advanced=True,
)

# --- Output Format Parameter ---

advanced_output_param = knext.BoolParameter(
    label="Compute advanced statistics",
    description=(
        "Controls the level of detail in the ANOVA Results table.\n\n"
        "• Basic: Factor, F-Statistic, P-Value, Conclusion.\n\n"
        "• Advanced: Full variance decomposition — Sum of Squares, DF, Mean Square, Partial Eta Squared, Conclusion."
    ),
    default_value=False,
    is_advanced=True,
)
