"""
Post-Hoc Multiple Comparisons Node for KNIME.

This module provides a KNIME node for performing post-hoc multiple comparison tests
following significant ANOVA results. Supports Tukey HSD and Holm-Bonferroni methods
with comprehensive validation and dual input ports.
"""

import knime.extension as knext
import numpy as np
import pandas as pd
from .post_hoc import (
    run_one_way_anova,
    validate_anova_data,
    run_tukey_test,
    run_bonferroni_test,
    format_tukey_results_for_knime,
    format_bonferroni_results_for_knime,
    test_type_param,
    data_column_param,
    group_column_param,
    alpha_param,
    PostHocTestType,
)


# Create post-hoc tests category (same as normality tests)
post_hoc_category = knext.category(
    path="/community",
    level_id="utd_development",
    name="University of Texas at Dallas Development",
    description="Statistical Post-Hoc Multiple Comparison Testing Node",
    icon="./icons/utd.png",
)


@knext.node(
    name="Post-Hoc Multiple Comparisons",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="./icons/post_hoc.png",
    category=post_hoc_category,
)
@knext.input_table(name="Data", description="Data table with numeric dependent variable and categorical grouping variable.")
@knext.output_table(
    name="ANOVA Summary",
    description="Overall ANOVA test results.",
)
@knext.output_table(
    name="Pairwise Details",
    description="Pairwise post-hoc comparison results (conditional on ANOVA significance).",
)
class PostHocTestsNode:
    test_type = test_type_param
    data_column = data_column_param
    group_column = group_column_param
    alpha = alpha_param

    def _validate_and_prepare_data(self, df, data_col, group_col):
        # High-level: validate that required columns exist and that the
        # data meet minimal assumptions for ANOVA; return (data_array, groups_array).
        # Check column existence
        if data_col is None:
            raise ValueError("No dependent variable selected. Please configure the node and select a numeric data column.")

        if group_col is None:
            raise ValueError("No grouping variable selected. Please configure the node and select a categorical grouping column.")

        missing_cols = []
        if data_col not in df.columns:
            missing_cols.append(f"dependent variable '{data_col}'")
        if group_col not in df.columns:
            missing_cols.append(f"grouping variable '{group_col}'")

        if missing_cols:
            raise ValueError(f"Columns not found in data: {', '.join(missing_cols)}")

        # Extract data
        data = df[data_col].values
        groups = df[group_col].values

        # Validate data requirements for ANOVA
        try:
            validate_anova_data(data, groups)
            return data, groups
        except ValueError as e:
            raise ValueError(f"Data validation failed: {str(e)}")

    def configure(self, cfg_ctx, input_spec):
        """Configure the node's two output table schemas."""
        # Output Port 1: ANOVA Summary
        # Columns: Tested Variable, Grouping Variable, Significance Level, ANOVA p-Value, Overall Conclusion
        anova_summary_cols = [
            knext.Column(knext.string(), "Tested Variable"),
            knext.Column(knext.string(), "Grouping Variable"),
            knext.Column(knext.double(), "Significance Level"),
            knext.Column(knext.double(), "ANOVA p-Value"),
            knext.Column(knext.string(), "Overall Conclusion"),
        ]

        # Output Port 2: Pairwise Details
        # Columns: Comparison, Post-Hoc Method, Mean Difference, Corrected p-Value, Difference is Significant?
        pairwise_details_cols = [
            knext.Column(knext.string(), "Comparison"),
            knext.Column(knext.string(), "Post-Hoc Method"),
            knext.Column(knext.double(), "Mean Difference"),
            knext.Column(knext.double(), "Corrected p-Value"),
            knext.Column(knext.string(), "Difference is Significant?"),
        ]

        anova_summary_schema = knext.Schema.from_columns(anova_summary_cols)
        pairwise_details_schema = knext.Schema.from_columns(pairwise_details_cols)

        return anova_summary_schema, pairwise_details_schema

    def execute(self, exec_ctx, input_table):
        """Execute the integrated ANOVA and post-hoc analysis."""
        # Convert input table to pandas DataFrame
        df = input_table.to_pandas()

        # Step 1: Validate and prepare data
        data, groups = self._validate_and_prepare_data(df, self.data_column, self.group_column)

        # Step 2: Run ANOVA analysis
        anova_results = run_one_way_anova(data, groups, alpha=self.alpha)

        # Step 3: Create Table 1 - ANOVA Summary (single row)
        anova_p_value = anova_results["summary"]["p_value"]
        is_significant = anova_p_value <= self.alpha

        if is_significant:
            overall_conclusion = "Significant Difference Found"
        else:
            overall_conclusion = "No Difference Found"

        table_1_anova = pd.DataFrame(
            {
                "Tested Variable": [self.data_column],
                "Grouping Variable": [self.group_column],
                "Significance Level": [self.alpha],
                "ANOVA p-Value": [anova_p_value],
                "Overall Conclusion": [overall_conclusion],
            }
        )

        # Step 4: Create Table 2 - Pairwise Details (conditional on ANOVA significance)
        if is_significant:
            # ANOVA IS significant - run post-hoc tests

            # Validate data for post-hoc (additional checks beyond ANOVA)
            unique_groups = np.unique(groups)
            if len(unique_groups) < 3:
                raise ValueError(
                    f"Post-hoc tests require at least 3 groups, found {len(unique_groups)}. "
                    f"With only 2 groups, the ANOVA F-test is equivalent to a t-test."
                )

            # Run selected post-hoc test
            if self.test_type == PostHocTestType.TUKEY_HSD.name:
                test_results = run_tukey_test(data, groups, alpha=self.alpha)
                pairwise_df, _ = format_tukey_results_for_knime(test_results)
                post_hoc_method = "Tukey HSD"
            else:  # Holm-Bonferroni
                test_results = run_bonferroni_test(data, groups, alpha=self.alpha)
                pairwise_df, _ = format_bonferroni_results_for_knime(test_results)
                post_hoc_method = "Holm-Bonferroni"

            # Extract required columns and format
            table_2_pairwise = pd.DataFrame(
                {
                    "Comparison": pairwise_df["Comparison"],
                    "Post-Hoc Method": [post_hoc_method] * len(pairwise_df),
                    "Mean Difference": pairwise_df["Mean Difference"],
                    "Corrected p-Value": pairwise_df.get("Corrected P-Value", pairwise_df.get("P-Value")),
                    "Difference is Significant?": pairwise_df.apply(
                        lambda row: "Yes"
                        if pd.to_numeric(row.get("Corrected P-Value", row.get("P-Value", np.nan)), errors="coerce") <= self.alpha
                        else "No",
                        axis=1,
                    ),
                }
            )
        else:
            # ANOVA IS NOT significant - create single fallback row
            table_2_pairwise = pd.DataFrame(
                {
                    "Comparison": [f"ANOVA not significant (p = {anova_p_value:.3f}). Comparisons were skipped."],
                    "Post-Hoc Method": ["N/A"],
                    "Mean Difference": [np.nan],
                    "Corrected p-Value": [np.nan],
                    "Difference is Significant?": ["N/A"],
                }
            )

        # Convert to KNIME tables
        anova_table = knext.Table.from_pandas(table_1_anova)
        pairwise_table = knext.Table.from_pandas(table_2_pairwise)

        # Return tables in same order as configure() and decorators
        return anova_table, pairwise_table
