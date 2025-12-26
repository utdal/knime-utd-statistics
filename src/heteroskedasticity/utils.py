"""
Utility functions and parameters for heteroskedasticity testing and OLS regression.

This module provides parameter definitions, enumerations, and helper functions
for the Heteroskedasticity Node in KNIME.
"""

import knime.extension as knext
import pandas as pd
from typing import List


def is_numeric(col: knext.Column) -> bool:
    """
    Helper function to filter for numeric columns.

    Args:
        col: KNIME column object

    Returns:
        True if column is numeric (int or float), False otherwise
    """
    return col.ktype in (knext.double(), knext.int32(), knext.int64())


def detect_categorical_columns(df: pd.DataFrame, column_names: List[str]) -> List[str]:
    """
    Automatically detect which columns should be treated as categorical.

    Categorical columns are identified by their pandas dtype:
    - object (strings)
    - category
    - bool

    Args:
        df: Input pandas DataFrame
        column_names: List of column names to check

    Returns:
        List of column names that are categorical

    Example:
        >>> df = pd.DataFrame({'age': [25, 30], 'dept': ['Sales', 'IT']})
        >>> detect_categorical_columns(df, ['age', 'dept'])
        ['dept']
    """
    categorical_dtypes = ["object", "category", "bool"]
    categorical_cols = []

    for col_name in column_names:
        if col_name in df.columns:
            if df[col_name].dtype.name in categorical_dtypes:
                categorical_cols.append(col_name)

    return categorical_cols


def format_p_value(p):
    """
    Format p-value to avoid scientific notation (e.g., E-22) in output.

    This function ensures p-values are displayed in a human-readable format:
    - Very small values (< 0.001) are shown as "< 0.001"
    - Other values are rounded to 4 decimal places
    - Handles missing or invalid values gracefully

    Args:
        p: P-value (float, or potentially NaN or '?')

    Returns:
        Formatted string representation of the p-value

    Example:
        >>> format_p_value(0.0456)
        '0.0456'
        >>> format_p_value(1.23e-22)
        '< 0.001'
        >>> format_p_value(None)
        '?'
    """
    # Handle potential nulls or '?' from KNIME
    if pd.isna(p) or p == "?":
        return "?"

    # Threshold for extremely small values
    if p < 0.001:
        return "< 0.001"

    # Rounding for meaningful values
    # Using 4 decimal places is standard to catch p=0.049 vs p=0.051
    else:
        return f"{p:.4f}"


# Test type enumeration
class TestType(knext.EnumParameterOptions):
    """
    Enumeration of available heteroskedasticity tests.

    Each test checks whether the variance of regression errors is constant
    (homoskedastic) or changes systematically (heteroskedastic).
    """

    BREUSCH_PAGAN = (
        "Breusch-Pagan",
        "Tests if error variance is related to the predictor variables. Most commonly used test, assumes normally distributed errors.",
    )
    WHITE = ("White", "General test that doesn't assume normal distribution. Detects more complex patterns but is computationally intensive.")
    GOLDFELD_QUANDT = ("Goldfeld-Quandt", "Compares variance between two subgroups of data. Requires specifying a sort variable and split fraction.")


# Test type parameter
test_type_param = knext.EnumParameter(
    label="Heteroskedasticity Test",
    description=(
        "Select which test to use for detecting heteroskedasticity:\n\n"
        "• Breusch-Pagan: Standard test for most situations\n"
        "• White: More robust but slower\n"
        "• Goldfeld-Quandt: Compares variance between groups"
    ),
    enum=TestType,
    default_value=TestType.BREUSCH_PAGAN.name,
)


# Target column parameter
target_column_param = knext.ColumnParameter(
    label="Target Variable (y)",
    description=("Select the numeric column you want to predict. This is the dependent variable in your regression model."),
    column_filter=is_numeric,
)


# Predictor columns parameter
predictor_columns_param = knext.MultiColumnParameter(
    label="Predictor Variables (X)",
    description=(
        "Select one or more columns to use as predictors (independent variables). "
        "Can include numeric and categorical columns. "
        "Categorical columns will be automatically converted to dummy variables."
    ),
)


# Significance level parameter
alpha_param = knext.DoubleParameter(
    label="Significance Level (α)",
    description=(
        "Threshold for determining statistical significance (default: 0.05). "
        "If the test p-value is below this threshold, heteroskedasticity is detected."
    ),
    default_value=0.05,
    min_value=0.01,
    max_value=0.20,
)


# Goldfeld-Quandt sort variable parameter
gq_sort_variable_param = knext.ColumnParameter(
    label="Sort Variable (Goldfeld-Quandt)",
    description=(
        "Select which predictor variable to sort the data by before splitting. "
        "The test will compare variance between low and high values of this variable. "
        "Only used when Goldfeld-Quandt test is selected."
    ),
    column_filter=is_numeric,
)


# Goldfeld-Quandt split fraction parameter
gq_split_fraction_param = knext.DoubleParameter(
    label="Split Fraction (Goldfeld-Quandt)",
    description=(
        "Proportion of data to use in each comparison group (default: 0.5). "
        "For example, 0.5 means the bottom 50% and top 50% are compared, "
        "with the middle observations omitted. "
        "Only used when Goldfeld-Quandt test is selected."
    ),
    default_value=0.5,
    min_value=0.2,
    max_value=0.8,
)
