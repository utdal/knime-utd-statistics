"""
Utility functions and parameters for normality tests.
"""

import knime.extension as knext


def is_numeric(col: knext.Column) -> bool:
    """Helper function to filter for numeric columns."""
    return col.ktype in (knext.double(), knext.int32(), knext.int64())


# Test type enumeration
class TestType(knext.EnumParameterOptions):
    ANDERSON_DARLING = (
        "Anderson-Darling",
        "Sensitive to deviations in the center and tails of the distribution.",
    )
    CRAMER = (
        "Cramer-von Mises",
        "Focuses on the overall shape and fit to the normal curve.",
    )


# Individual parameters (not in a group)
test_type_param = knext.EnumParameter(
    label="Description",
    description="The Statistical Normality Tests node checks whether your data follows a normal (bell-shaped) distribution - a key assumption in many statistical analyses. You can choose between two well-known methods:",
    enum=TestType,
    default_value=TestType.ANDERSON_DARLING.name,
)

input_column_param = knext.ColumnParameter(
    label="Data Column",
    description="Numeric column to test for normality distribution.",
    column_filter=is_numeric,
)

alpha_param = knext.DoubleParameter(
    label="Significance Level (α)",
    description="Significance level for the normality test (default: 0.05)",
    default_value=0.05,
    min_value=0.01,
    max_value=0.999,
)