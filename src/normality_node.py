"""
Simplified Normality Tests Node for KNIME.

This module provides a single KNIME node that allows users to choose between
Anderson-Darling (statsmodels) and Cramer-von Mises (scipy) normality tests
with minimal parameters and automatic data validation.
"""

import knime.extension as knext
import numpy as np
import pandas as pd
from .normality_tests import run_ad_test, run_cramer_test
from .normality_tests.utils import test_type_param, input_columns_param, alpha_param, TestType


# UTD statistical analysis category
utd_category = knext.category(
    path="/community",
    level_id="utd_development",
    name="University of Texas at Dallas Development",
    description="Statistical analysis tools developed by the University of Texas at Dallas",
    icon="./icons/utd.png",
)


@knext.node(
    name="Statistical Normality Tests",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="./icons/curve.jpg",
    category=utd_category,
)
@knext.input_table(name="Input data", description="Table containing the numeric column to test.")
@knext.output_table(
    name="Results",
    description="Normality test results with statistical decision.",
)
class NormalityTestsNode:
    """Tests whether your data follows a normal (bell-shaped) distribution using Anderson-Darling or Cramer-von Mises methods.

    This node performs statistical tests to determine if your data follows a normal distribution,
    which is a key assumption required for many parametric statistical procedures.
    """

    test_type = test_type_param
    input_columns = input_columns_param
    alpha = alpha_param

    def _validate_input_data(self, df, col_name):
        """
        Internal validation - not exposed as user parameters.
        Validates input data and returns clean numpy array.
        """
        # Column existence check
        if col_name is None:
            raise ValueError("Column name is None.")

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
        results_cols = [
            knext.Column(knext.string(), "Column Tested"),
            knext.Column(knext.string(), "Test Method"),
            knext.Column(knext.int32(), "Sample Size (n)"),
            knext.Column(knext.double(), "Test Statistic"),
            knext.Column(knext.double(), "P-Value"),
            knext.Column(knext.string(), "Statistical Decision"),
        ]

        results_schema = knext.Schema.from_columns(results_cols)
        return results_schema

    def execute(self, exec_ctx, input_table):
        """Execute the selected normality test on all selected columns."""
        df = input_table.to_pandas()
        selected_columns = self.input_columns

        # Validate that at least one column is selected
        if not selected_columns or len(selected_columns) == 0:
            raise ValueError("No columns selected. Please select at least one numeric column to test.")

        # Determine test method name
        test_method_name = "Anderson-Darling" if self.test_type == TestType.ANDERSON_DARLING.name else "Cramer-von Mises"

        results = []

        # Process each column
        for col_name in selected_columns:
            try:
                data = self._validate_input_data(df, col_name)

                if self.test_type == TestType.ANDERSON_DARLING.name:
                    result = run_ad_test(data, alpha=self.alpha)
                else:  # Cramer-von Mises test
                    result = run_cramer_test(data, alpha=self.alpha)

                results.append(
                    {
                        "Column Tested": col_name,
                        "Test Method": test_method_name,
                        "Sample Size (n)": np.int32(result["n"]),
                        "Test Statistic": result["statistic"],
                        "P-Value": result["p_value"],
                        "Statistical Decision": result["decision"],
                    }
                )

            except ValueError as e:
                exec_ctx.set_warning(f"Column '{col_name}' skipped: {str(e)}")
                results.append(
                    {
                        "Column Tested": col_name,
                        "Test Method": test_method_name,
                        "Sample Size (n)": np.int32(0),
                        "Test Statistic": np.nan,
                        "P-Value": np.nan,
                        "Statistical Decision": f"Skipped - {str(e)}",
                    }
                )

        if not results:
            raise ValueError("No columns could be tested. All selected columns failed validation.")

        return knext.Table.from_pandas(pd.DataFrame(results))
