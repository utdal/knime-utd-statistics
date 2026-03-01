"""
One-Way MANOVA Node for KNIME.

This module provides a KNIME node for performing one-way Multivariate Analysis
of Variance (MANOVA) with Pillai's Trace test statistic and Box's M assumption
check.  Supports basic and advanced output modes via a checkbox toggle.
"""

import knime.extension as knext
import pandas as pd
from .manova import (
    run_manova,
    format_basic_results,
    format_advanced_results,
    compute_box_m,
    validate_manova_data,
    dependent_columns_param,
    group_column_param,
    alpha_param,
    advanced_stats_param,
    is_string,
)


# UTD statistical analysis category
utd_category = knext.category(
    path="/community",
    level_id="utd_development",
    name="University of Texas at Dallas Development",
    description="Statistical analysis tools developed by the University of Texas at Dallas",
    icon="./icons/utd.png",
)


@knext.node(
    name="One-Way MANOVA",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="./icons/MANOVA3.png",
    category=utd_category,
)
@knext.input_table(
    name="Input Data",
    description="Data table with multiple numeric dependent variables and one categorical grouping variable.",
)
@knext.output_table(
    name="Multivariate Results",
    description="MANOVA results using Pillai's Trace.",
)
@knext.output_table(
    name="Reliability Report",
    description="Box's M test result for equality of covariance matrices across groups.",
)
class ManovaNode:
    """
    Performs one-way Multivariate Analysis of Variance (MANOVA) using Pillai's Trace.

    MANOVA tests whether group means differ across multiple dependent variables
    simultaneously.

    This node uses Pillai's Trace as the primary test statistic due to its
    robustness to violations of assumptions.  Box's M test is provided as a
    reliability check for the equality-of-covariance-matrices assumption.
    """

    dependent_columns = dependent_columns_param
    group_column = group_column_param
    alpha = alpha_param
    advanced_stats = advanced_stats_param

    def configure(self, cfg_ctx, input_spec):
        """Configure the node's two output table schemas."""

        # Auto-preselect grouping column if not already selected
        if self.group_column is None:
            string_cols = [col.name for col in input_spec if is_string(col)]
            if string_cols:
                self.group_column = string_cols[-1]

        # Validate that grouping column is selected
        if self.group_column is None:
            raise knext.InvalidParametersError("No grouping variable selected. Please select a categorical column.")

        # ---------------------------------------------------------------
        # Port 1 – Multivariate Results (schema depends on checkbox)
        # ---------------------------------------------------------------
        if self.advanced_stats:
            results_cols = [
                knext.Column(knext.string(), "Source"),
                knext.Column(knext.string(), "Test Stat"),
                knext.Column(knext.double(), "Value"),
                knext.Column(knext.double(), "Numerator Df"),
                knext.Column(knext.double(), "Denominator Df"),
                knext.Column(knext.double(), "F-Value"),
                knext.Column(knext.double(), "P-Value"),
            ]
        else:
            results_cols = [
                knext.Column(knext.string(), "Factor"),
                knext.Column(knext.double(), "Pillai's P-Val"),
                knext.Column(knext.string(), "Conclusion"),
            ]

        # ---------------------------------------------------------------
        # Port 2 – Reliability Report (Box's M)
        # ---------------------------------------------------------------
        reliability_cols = [
            knext.Column(knext.string(), "Test"),
            knext.Column(knext.double(), "Statistic"),
            knext.Column(knext.double(), "Chi-Square Approx"),
            knext.Column(knext.double(), "Degrees of Freedom"),
            knext.Column(knext.double(), "P-Value"),
            knext.Column(knext.string(), "Status"),
        ]

        results_schema = knext.Schema.from_columns(results_cols)
        reliability_schema = knext.Schema.from_columns(reliability_cols)

        return results_schema, reliability_schema

    def execute(self, exec_ctx, input_table):
        """Execute the MANOVA analysis."""

        df = input_table.to_pandas()

        # Resolve selected columns
        dep_vars = list(self.dependent_columns)
        group_col = self.group_column

        # ------------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------------
        if not dep_vars:
            raise ValueError("No dependent variables selected. Please configure the node and select at least 2 numeric columns.")

        if group_col is None:
            raise ValueError("No grouping variable selected. Please configure the node and select a categorical column.")

        # Validate and clean data (NaN handling, constant-variance, etc.)
        df_clean = validate_manova_data(df, dep_vars, group_col)

        # ------------------------------------------------------------------
        # Port 1 – MANOVA via Pillai's Trace
        # ------------------------------------------------------------------
        manova_result = run_manova(df_clean, dep_vars, group_col, alpha=self.alpha)

        if self.advanced_stats:
            results_df = format_advanced_results(manova_result)
        else:
            results_df = format_basic_results(manova_result)

        # ------------------------------------------------------------------
        # Port 2 – Box's M reliability check
        # ------------------------------------------------------------------
        box_m_result = compute_box_m(df_clean, dep_vars, group_col)

        status = "Pass" if box_m_result["p_value"] > 0.001 else "Warning"

        reliability_df = pd.DataFrame(
            {
                "Test": ["Box's M"],
                "Statistic": [box_m_result["statistic"]],
                "Chi-Square Approx": [box_m_result["chi2_approx"]],
                "Degrees of Freedom": [box_m_result["df"]],
                "P-Value": [box_m_result["p_value"]],
                "Status": [status],
            }
        )

        # ------------------------------------------------------------------
        # Return KNIME tables
        # ------------------------------------------------------------------
        results_table = knext.Table.from_pandas(results_df)
        reliability_table = knext.Table.from_pandas(reliability_df)

        return results_table, reliability_table
