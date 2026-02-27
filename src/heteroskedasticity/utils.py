"""
Utility functions and parameters for heteroskedasticity tests.
"""

from typing import List

import knime.extension as knext
import pandas as pd


def is_numeric(col: knext.Column) -> bool:
    """Helper function to filter for numeric columns."""
    return col.ktype in (knext.double(), knext.int32(), knext.int64())


def detect_categorical_columns(df: pd.DataFrame, column_names: List[str]) -> List[str]:
    """Detect categorical columns for dummy encoding.

    Any column that is not numeric (including pandas StringDtype from KNIME)
    is treated as categorical.
    """
    categorical_cols: List[str] = []

    for col_name in column_names:
        if col_name in df.columns and not pd.api.types.is_numeric_dtype(df[col_name]):
            categorical_cols.append(col_name)

    return categorical_cols


def format_p_value(p):
    """Format p-values for readable KNIME output (avoid scientific notation)."""
    # Handle potential nulls or '?' from KNIME
    if pd.isna(p) or p == "?":
        return "?"

    # Threshold for very small values
    if p < 0.001:
        return "< 0.001"

    # Standard rounding (helps distinguish p=0.049 vs p=0.051)
    return f"{p:.4f}"


# Test type enumeration
class TestType(knext.EnumParameterOptions):
    BREUSCH_PAGAN = (
        "Breusch-Pagan",
        "Standard test for most situations. Tests whether error variance is related to the predictors.",
    )
    WHITE = (
        "White",
        "More general test that can detect complex patterns in error variance.",
    )
    GOLDFELD_QUANDT = (
        "Goldfeld-Quandt",
        "Compares variance between low and high values of a selected sort variable.",
    )


# Individual parameters (not in a group)
test_type_param = knext.EnumParameter(
    label="Description",
    description=(
        "The Heteroskedasticity Tests node checks whether your regression errors have constant variance "
        "(homoskedasticity) or changing variance (heteroskedasticity) — a key assumption in regression analysis. "
        "You can choose between three well-known methods:\n\n"
        "• Breusch-Pagan: Standard test for most situations.\n"
        "• White: More general test for complex patterns.\n"
        "• Goldfeld-Quandt: Compares variance across two groups after sorting by a selected variable."
    ),
    enum=TestType,
    default_value=TestType.BREUSCH_PAGAN.name,
)

target_column_param = knext.ColumnParameter(
    label="Target Variable (y)",
    description="Numeric dependent variable you want to predict.",
    column_filter=is_numeric,
)

predictor_columns_param = knext.MultiColumnParameter(
    label="Predictor Variables (X)",
    description=("One or more predictor variables (independent variables). Categorical columns are automatically converted to dummy variables."),
)

alpha_param = knext.DoubleParameter(
    label="Significance Level (α)",
    description="Significance level for the heteroskedasticity test (default: 0.05)",
    default_value=0.05,
    min_value=0.01,
    max_value=0.20,
)

gq_sort_variable_param = knext.ColumnParameter(
    label="Sort Variable (Goldfeld-Quandt)",
    description="Numeric predictor used to sort the data before splitting into two comparison groups.",
    column_filter=is_numeric,
)

gq_split_fraction_param = knext.DoubleParameter(
    label="Split Fraction (Goldfeld-Quandt)",
    description="Proportion of data used in each comparison group (default: 0.5).",
    default_value=0.5,
    min_value=0.2,
    max_value=0.8,
)
