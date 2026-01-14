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
    categorical_dtypes = ["object", "category", "bool"]
    categorical_cols = []

    for col_name in column_names:
        if col_name in df.columns:
            if df[col_name].dtype.name in categorical_dtypes:
                categorical_cols.append(col_name)

    return categorical_cols


def format_p_value(p):
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
    BREUSCH_PAGAN = (
        "Breusch-Pagan",
        "Tests if error variance is related to the predictor variables. Most commonly used test, assumes normally distributed errors.",
    )
    WHITE = ("White", "General test that doesn't assume normal distribution. Detects more complex patterns but is computationally intensive.")
    GOLDFELD_QUANDT = ("Goldfeld-Quandt", "Compares variance between two subgroups of data. Requires specifying a sort variable and split fraction.")


# Test type parameter
test_type_param = knext.EnumParameter(
    label="Description",
    description=(
        "The Heteroskedasticity Tests node checks whether your regression errors have constant variance (homoskedastic) or changing variance (heteroskedastic) - a key assumption in regression analysis. "
        "When this assumption is violated, your p-values and confidence intervals become unreliable. Choose between three well-established methods:\n\n"
        "• Breusch-Pagan: Standard test for most situations. Tests if error variance is related to your predictor variables.\n\n"
        "• White: More general test that doesn't require normally distributed errors. Good for detecting complex patterns.\n\n"
        "• Goldfeld-Quandt: Compares variance between two groups of your data. Useful when you suspect variance changes with a specific variable."
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
