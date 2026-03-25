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
        "Marginal sum of squares. Tests each factor's unique contribution. "
        "Recommended for unbalanced designs or when interactions are present.",
    )


# =============================================================================
# Parameter Definitions
# =============================================================================


# --- Core Analysis Parameters ---

response_column_param = knext.ColumnParameter(
    label="Response Variable (Dependent)",
    description=(
        "Select the continuous numeric column to analyze (dependent variable). "
        "This is the outcome you want to explain using the factor variables."
    ),
    column_filter=is_numeric,
)

factor_columns_param = knext.MultiColumnParameter(
    label="Factor Variables (Independent)",
    description=(
        "Select one or more categorical grouping variables (independent variables). "
        "These are the factors whose effects on the response variable will be tested. "
        "String and categorical columns work best. Numeric columns with few unique values "
        "will be treated as categorical."
    ),
)

include_interactions_param = knext.BoolParameter(
    label="Include Interaction Terms",
    description=(
        "Include interactions between factors in the model.\n\n"
        "• Checked: Factorial model with interactions (e.g., A + B + A:B)\n"
        "• Unchecked: Main effects only (e.g., A + B)\n\n"
        "Interactions test whether the effect of one factor depends on the level of another."
    ),
    default_value=True,
)

max_interaction_order_param = knext.IntParameter(
    label="Maximum Interaction Order",
    description=(
        "Highest-order interaction to include in the model.\n\n"
        "• 2: Up to 2-way interactions (A:B) - recommended default\n"
        "• 3: Up to 3-way interactions (A:B:C)\n"
        "• 4: Up to 4-way interactions (A:B:C:D)\n\n"
        "Higher-order interactions increase model complexity exponentially "
        "and are rarely meaningful in practice."
    ),
    default_value=2,
    min_value=2,
    max_value=4,
)

anova_type_param = knext.EnumParameter(
    label="Sum of Squares Type",
    description=(
        "Method for calculating sum of squares:\n\n"
        "• Type I: Sequential (order-dependent)\n"
        "• Type II: Partial (best for balanced designs without interactions)\n"
        "• Type III: Marginal (handles interactions and unbalanced data)\n\n"
        "Type II is the default. Use Type III if your design is unbalanced "
        "or if interactions are important."
    ),
    enum=AnovaType,
    default_value=AnovaType.TYPE_II.name,
)

alpha_param = knext.DoubleParameter(
    label="Significance Level (α)",
    description=(
        "Threshold for determining statistical significance. "
        "Factors with p-value ≤ α are considered significant. "
        "Common values: 0.05 (5%), 0.01 (1%), 0.10 (10%)."
    ),
    default_value=0.05,
    min_value=0.001,
    max_value=0.5,
)


# --- Output Format Parameter ---

advanced_output_param = knext.BoolParameter(
    label="Advanced Output",
    description=(
        "Choose the level of detail in the ANOVA output table.\n\n"
        "• Unchecked (Basic): Factor, F-Statistic, P-Value, Conclusion\n"
        "• Checked (Advanced): Source, Sum_Sq, DF, F, P-Value, Coefficient, "
        "Std Error, T-Value, P-Value, Conclusion\n\n"
        "Advanced output includes coefficient details merged with ANOVA statistics."
    ),
    default_value=False,
)
