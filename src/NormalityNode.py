"""
Simplified Normality Tests Node for KNIME.

This module provides a single KNIME node that allows users to choose between
Anderson-Darling (statsmodels) and Cramer-von Mises (scipy) normality tests
with minimal parameters and automatic data validation.
"""

import knime.extension as knext
import numpy as np
import pandas as pd
from .normalityTests import run_ad_test, run_cramer_test


# Create normality tests category
normality_category = knext.category(
    path="/community",
    level_id="normality_tests",
    name="Normality Tests",
    description="Statistical normality testing nodes",
    icon="./icons/icon.png",
)


def _is_numeric(col: knext.Column) -> bool:
    """Helper function to filter for numeric columns."""
    return col.ktype in (knext.double(), knext.int32(), knext.int64())


@knext.parameter_group("Normality Test Settings")
class _NormalityParams:
    """Parameter group for unified normality testing node."""

    # Test type selection
    class _TestType(knext.EnumParameterOptions):
        ANDERSON_DARLING = (
            "Anderson-Darling",
            "Anderson-Darling normality test using statsmodels",
        )
        CRAMER = (
            "Cramer-von Mises",
            "Cramer-von Mises normality test using scipy",
        )

    test_type = knext.EnumParameter(
        label="Test Type",
        description="Choose the normality test to perform",
        enum=_TestType,
        default_value=_TestType.ANDERSON_DARLING.name,
    )

    # Column selection
    input_column = knext.ColumnParameter(
        label="Data column",
        description="Numeric column to test for normality.",
        column_filter=_is_numeric,
    )


@knext.node(
    name="Normality Tests",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="./icons/icon.png",
    category=normality_category,
)
@knext.input_table(
    name="Input data", description="Table containing the numeric column to test."
)
@knext.output_table(
    name="Results",
    description="Normality test results with statistical decision.",
)
class NormalityTestsNode:
    """
    Simplified normality testing node supporting Anderson-Darling and Cramer-von Mises tests.

    Features:
    • Choose between Anderson-Darling (statsmodels) and Cramer-von Mises (scipy) tests
    • Simple interface with minimal parameters
    • Automatic data validation (rejects data with nulls)
    • Fixed significance level (α = 0.05)
    """

    params = _NormalityParams()

    def _validate_input_data(self, df, col_name):
        """
        Internal validation - not exposed as user parameters.
        Validates input data and returns clean numpy array.
        """
        # Column existence check
        if col_name is None:
            raise ValueError("No column selected. Please configure the node and select a numeric data column.")
        
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in input data.")
        
        # Extract column data
        data = df[col_name]
        
        # Numeric type check  
        if not pd.api.types.is_numeric_dtype(data):
            raise ValueError(f"Column '{col_name}' must be numeric (int/float). Found: {data.dtype}")
        
        # Null check - REJECT if any nulls present
        if data.isnull().any():
            null_count = data.isnull().sum()
            raise ValueError(f"Column '{col_name}' contains {null_count} null/missing values. Normality tests require complete data with no missing values.")
        
        # Constant data check
        if data.nunique() == 1:
            raise ValueError(f"Column '{col_name}' contains only constant values ({data.iloc[0]}). Normality tests cannot be performed on constant data.")
        
        return data.values  # Return as numpy array

    def configure(self, cfg_ctx, input_spec):
        """Configure the node's output table schema."""
        # Simple single output table schema
        results_cols = [
            knext.Column(knext.string(), "Test"),
            knext.Column(knext.string(), "Column Tested"),
            knext.Column(knext.int32(), "Sample Size (n)"),
            knext.Column(knext.double(), "Test Statistic"),
            knext.Column(knext.double(), "P-Value"),
            knext.Column(knext.string(), "Statistical Decision"),
        ]

        results_schema = knext.Schema.from_columns(results_cols)
        return results_schema

    def execute(self, exec_ctx, input_table):
        """Execute the selected normality test."""
        df = input_table.to_pandas()
        col_name = self.params.input_column

        # Validate input data (will raise error if validation fails)
        data = self._validate_input_data(df, col_name)

        # Execute the selected test
        if self.params.test_type == _NormalityParams._TestType.ANDERSON_DARLING.name:
            result = run_ad_test(data)
        else:  # Cramer-von Mises test
            result = run_cramer_test(data)

        # Format results into KNIME table
        results_df = pd.DataFrame([{
            "Test": result["test"],
            "Column Tested": col_name,
            "Sample Size (n)": np.int32(result["n"]),
            "Test Statistic": result["statistic"],
            "P-Value": result["p_value"],
            "Statistical Decision": result["decision"],
        }])

        return knext.Table.from_pandas(results_df)


