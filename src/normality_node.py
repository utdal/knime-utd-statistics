"""
Simplified Normality Tests Node for KNIME.

This module provides a single KNIME node that allows users to choose between
Anderson-Darling (statsmodels) and Cramer-von Mises (scipy) normality tests
with minimal parameters and automatic data validation.
"""

import knime.extension as knext
import numpy as np
import pandas as pd
from . import utd_category
from .normality_tests import run_ad_test, run_cramer_test
from .normality_tests.utils import test_type_param, input_column_param, alpha_param, TestType


@knext.node(
    name="Statistical Normality Tests",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="./icons/bell_curve.png",
    category=utd_category,
)
@knext.input_table(name="Input data", description="Table containing the numeric column to test.")
@knext.output_table(
    name="Results",
    description="Normality test results with statistical decision.",
)
class NormalityTestsNode:
    """
    Tests whether your data follows a normal (bell-shaped) distribution using Anderson-Darling
    or Cramer-von Mises methods. Normality is a key assumption in many statistical analyses.

    This node performs statistical tests to determine if your data follows a normal distribution,
    which is required for many parametric statistical procedures.
    """

    test_type = test_type_param
    input_column = input_column_param
    alpha = alpha_param

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

        # Null check - REJECT if any nulls present
        if data.isnull().any():
            null_count = data.isnull().sum()
            raise ValueError(
                f"Column '{col_name}' contains {null_count} null/missing values. Normality tests require complete data with no missing values."
            )

        # Constant data check
        if data.nunique() == 1:
            raise ValueError(
                f"Column '{col_name}' contains only constant values ({data.iloc[0]}). Normality tests cannot be performed on constant data."
            )

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
        col_name = self.input_column

        # Validate input data (will raise error if validation fails)
        data = self._validate_input_data(df, col_name)

        # Execute the selected test
        if self.test_type == TestType.ANDERSON_DARLING.name:
            result = run_ad_test(data, alpha=self.alpha)
        else:  # Cramer-von Mises test
            result = run_cramer_test(data, alpha=self.alpha)

        # Format results into KNIME table
        results_df = pd.DataFrame(
            [
                {
                    "Test": result["test"],
                    "Column Tested": col_name,
                    "Sample Size (n)": np.int32(result["n"]),
                    "Test Statistic": result["statistic"],
                    "P-Value": result["p_value"],
                    "Statistical Decision": result["decision"],
                }
            ]
        )

        return knext.Table.from_pandas(results_df)
