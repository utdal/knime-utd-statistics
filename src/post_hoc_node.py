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
    format_anova_results_for_knime,
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
    name="ANOVA Results",
    description="ANOVA summary results (overall F-test).",
)
@knext.output_table(
    name="Pairwise Comparisons",
    description="Pairwise post-hoc comparison results (Tukey or Holm-Bonferroni).",
)
@knext.output_table(
    name="Group Summary",
    description="Descriptive statistics per group (N, mean, std dev).",
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
        """Configure the node's single output table schema."""
        # High-level: declare the three output table schemas (ANOVA, Pairwise,
        # Group Summary) that the node will return in `execute()`.
        # Build three separate schemas matching the tables we'll return in execute()
        anova_cols = [
            knext.Column(knext.string(), "Result Type"),
            knext.Column(knext.string(), "Test Column"),
            knext.Column(knext.string(), "Source"),
            knext.Column(knext.double(), "Sum of Squares"),
            knext.Column(knext.double(), "df"),
            knext.Column(knext.double(), "Mean Square"),
            knext.Column(knext.double(), "F"),
            knext.Column(knext.double(), "ANOVA P-Value"),
            knext.Column(knext.string(), "Test Method"),
            knext.Column(knext.double(), "Significance Level"),
            knext.Column(knext.string(), "Significant"),
        ]

        pairwise_cols = [
            knext.Column(knext.string(), "Result Type"),
            knext.Column(knext.string(), "Test Column"),
            knext.Column(knext.string(), "Comparison"),
            knext.Column(knext.string(), "Group 1"),
            knext.Column(knext.string(), "Group 2"),
            knext.Column(knext.double(), "Mean Difference"),
            knext.Column(knext.double(), "Lower CI"),
            knext.Column(knext.double(), "Upper CI"),
            knext.Column(knext.double(), "P-Value"),
            knext.Column(knext.double(), "Corrected P-Value"),
            knext.Column(knext.string(), "Reject H0"),
            knext.Column(knext.string(), "Test Method"),
            knext.Column(knext.double(), "Significance Level"),
            knext.Column(knext.string(), "Significant"),
        ]

        group_summary_cols = [
            knext.Column(knext.string(), "Result Type"),
            knext.Column(knext.string(), "Test Column"),
            knext.Column(knext.string(), "Group"),
            knext.Column(knext.double(), "N"),
            knext.Column(knext.double(), "Group Mean"),
            knext.Column(knext.double(), "Group Std Dev"),
            knext.Column(knext.string(), "Test Method"),
            knext.Column(knext.double(), "Significance Level"),
        ]

        anova_schema = knext.Schema.from_columns(anova_cols)
        pairwise_schema = knext.Schema.from_columns(pairwise_cols)
        group_schema = knext.Schema.from_columns(group_summary_cols)

        # Return schemas in the same order as the output_table decorators
        return anova_schema, pairwise_schema, group_schema

    def execute(self, exec_ctx, input_table):
        """Execute the integrated ANOVA and post-hoc analysis."""
        # High-level: run ANOVA, optionally run post-hoc (Tukey/Holm-Bonferroni),
        # assemble a unified results list, coerce types to the configured schema,
        # split into three tables, and return KNIME tables.

        # Convert input table to pandas DataFrame
        df = input_table.to_pandas()

        # Step 1: Validate and prepare data
        data, groups = self._validate_and_prepare_data(df, self.data_column, self.group_column)

        # Step 2: Run ANOVA analysis
        anova_results = run_one_way_anova(data, groups, alpha=self.alpha)

        # Step 3: Initialize results list for unified table
        all_results = []

        # Add ANOVA results to unified table
        anova_df = format_anova_results_for_knime(anova_results, data_column_name=self.data_column, group_column_name=self.group_column)

        row_id_counter = 0  # Initialize sequential row ID counter

        for _, row in anova_df.iterrows():
            result_row = {
                "Result Type": "ANOVA",
                "Row ID": row_id_counter,  # Use sequential counter instead of idx
                "Test Column": row["Test Column"],
                "Source": row["Source"],
                "Sum of Squares": row["Sum of Squares"],
                "df": row["df"],
                "Mean Square": row["Mean Square"],
                "F": row["F"],
                "ANOVA P-Value": row["p-value"],
                "Test Method": row["Test Method"],
                "Significance Level": row["Significance Level"],
                "Significant": row["Significant"],
                # Pairwise columns as NaN
                "Comparison": np.nan,
                "Group 1": np.nan,
                "Group 2": np.nan,
                "Mean Difference": np.nan,
                "Lower CI": np.nan,
                "Upper CI": np.nan,
                "P-Value": np.nan,
                "T-Statistic": np.nan,
                "Raw P-Value": np.nan,
                "Corrected P-Value": np.nan,
                "Reject H0": np.nan,
                # Group summary columns as NaN
                "Group": np.nan,
                "N": np.nan,
                "Group Mean": np.nan,
                "Group Std Dev": np.nan,
            }
            all_results.append(result_row)
            row_id_counter += 1  # Increment counter

        # Step 4: Check ANOVA significance and add post-hoc/summary results
        if anova_results["significant"]:
            # ANOVA is significant - proceed with post-hoc tests

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
                pairwise_df, group_summary_df = format_tukey_results_for_knime(test_results)
            else:  # Holm-Bonferroni
                test_results = run_bonferroni_test(data, groups, alpha=self.alpha)
                pairwise_df, group_summary_df = format_bonferroni_results_for_knime(test_results)

            # Add pairwise comparison results
            for _, row in pairwise_df.iterrows():
                # Determine corrected p-value and rejection decision
                corrected_p = row.get("Corrected P-Value", np.nan)
                raw_p = row.get("Raw P-Value", row.get("P-Value", np.nan))
                # Decision logic: prefer corrected p-value when available, then formatter's reject, then raw p-value
                reject_val = row.get("Reject H0", None)
                reject_bool = False
                # coerce corrected and raw p to numeric if possible
                try:
                    corrected_num = pd.to_numeric(corrected_p, errors="coerce")
                except Exception:
                    corrected_num = np.nan
                try:
                    raw_num = pd.to_numeric(raw_p, errors="coerce")
                except Exception:
                    raw_num = np.nan

                if not pd.isna(corrected_num):
                    reject_bool = bool(corrected_num <= float(self.alpha))
                elif reject_val is not None and not (isinstance(reject_val, float) and np.isnan(reject_val)):
                    # use provided reject value if formatter supplied it
                    try:
                        reject_bool = bool(reject_val)
                    except Exception:
                        reject_bool = str(reject_val).lower() in ("true", "1", "yes")
                elif not pd.isna(raw_num):
                    # fallback to raw p-value
                    reject_bool = bool(raw_num <= float(self.alpha))
                else:
                    reject_bool = False

                result_row = {
                    "Result Type": "Pairwise Comparison",
                    "Row ID": row_id_counter,  # Use sequential counter instead of idx
                    "Test Column": self.data_column,
                    "ANOVA P-Value": anova_results["summary"]["p_value"],
                    "Comparison": row["Comparison"],
                    "Group 1": row["Group 1"],
                    "Group 2": row["Group 2"],
                    "Mean Difference": row["Mean Difference"],
                    "Lower CI": row["Lower CI"],
                    "Upper CI": row["Upper CI"],
                    "P-Value": raw_p,
                    "T-Statistic": row.get("T-Statistic", np.nan),
                    "Raw P-Value": row.get("Raw P-Value", np.nan),
                    "Corrected P-Value": corrected_p,
                    "Reject H0": str(reject_bool),
                    "Test Method": row.get("Test Method", ""),
                    "Significance Level": row.get("Significance Level", self.alpha),
                    "Significant": str(reject_bool),
                    # ANOVA columns as NaN (except already filled)
                    "Source": np.nan,
                    "Sum of Squares": np.nan,
                    "df": np.nan,
                    "Mean Square": np.nan,
                    "F": np.nan,
                    # Group summary columns as NaN
                    "Group": np.nan,
                    "N": np.nan,
                    "Group Mean": np.nan,
                    "Group Std Dev": np.nan,
                }
                all_results.append(result_row)
                row_id_counter += 1  # Increment counter

            # Add group summary results
            for _, row in group_summary_df.iterrows():
                result_row = {
                    "Result Type": "Group Summary",
                    "Row ID": row_id_counter,  # Use sequential counter instead of idx
                    "Test Column": self.data_column,
                    "ANOVA P-Value": anova_results["summary"]["p_value"],
                    "Group": row["Group"],
                    "N": row["N"],
                    "Group Mean": row["Mean"],
                    "Group Std Dev": row["Std Dev"],
                    "Test Method": row["Test Method"],
                    "Significance Level": self.alpha,
                    "Significant": str(anova_results["significant"]),
                    # ANOVA columns as NaN
                    "Source": np.nan,
                    "Sum of Squares": np.nan,
                    "df": np.nan,
                    "Mean Square": np.nan,
                    "F": np.nan,
                    # Pairwise columns as NaN
                    "Comparison": np.nan,
                    "Group 1": np.nan,
                    "Group 2": np.nan,
                    "Mean Difference": np.nan,
                    "Lower CI": np.nan,
                    "Upper CI": np.nan,
                    "P-Value": np.nan,
                    "T-Statistic": np.nan,
                    "Raw P-Value": np.nan,
                    "Corrected P-Value": np.nan,
                    "Reject H0": np.nan,
                }
                all_results.append(result_row)
                row_id_counter += 1  # Increment counter

        else:
            # ANOVA is not significant - add basic group summary only
            empty_message = f"ANOVA not significant (p = {anova_results['summary']['p_value']:.4f}). Post-hoc tests not performed."

            unique_groups = np.unique(groups)
            for group in unique_groups:
                group_data = data[groups == group]
                result_row = {
                    "Result Type": "Group Summary",
                    "Row ID": row_id_counter,  # Use sequential counter instead of idx
                    "Test Column": self.data_column,
                    "ANOVA P-Value": anova_results["summary"]["p_value"],
                    "Group": group,
                    "N": len(group_data),
                    "Group Mean": np.mean(group_data),
                    "Group Std Dev": np.std(group_data, ddof=1),
                    "Test Method": empty_message,
                    "Significance Level": self.alpha,
                    "Significant": str(anova_results["significant"]),
                    # ANOVA columns as NaN
                    "Source": np.nan,
                    "Sum of Squares": np.nan,
                    "df": np.nan,
                    "Mean Square": np.nan,
                    "F": np.nan,
                    # Pairwise columns as NaN
                    "Comparison": np.nan,
                    "Group 1": np.nan,
                    "Group 2": np.nan,
                    "Mean Difference": np.nan,
                    "Lower CI": np.nan,
                    "Upper CI": np.nan,
                    "P-Value": np.nan,
                    "T-Statistic": np.nan,
                    "Raw P-Value": np.nan,
                    "Corrected P-Value": np.nan,
                    "Reject H0": np.nan,
                }
                all_results.append(result_row)
                row_id_counter += 1  # Increment counter

        # Create unified results DataFrame
        results_df = pd.DataFrame(all_results)

        # Enforce the exact column ordering and presence as declared in configure()
        configured_cols = [
            "Result Type",
            "Row ID",
            "Test Column",
            "Source",
            "Sum of Squares",
            "df",
            "Mean Square",
            "F",
            "ANOVA P-Value",
            "Comparison",
            "Group 1",
            "Group 2",
            "Mean Difference",
            "Lower CI",
            "Upper CI",
            "P-Value",
            "T-Statistic",
            "Raw P-Value",
            "Corrected P-Value",
            "Reject H0",
            "Group",
            "N",
            "Group Mean",
            "Group Std Dev",
            "Test Method",
            "Significance Level",
            "Significant",
        ]

        # Add any missing configured columns (use NaN) then reindex to exact order
        for col in configured_cols:
            if col not in results_df.columns:
                results_df[col] = np.nan
        results_df = results_df[configured_cols]

        # Coerce dtypes to match configure(): Row ID -> int32, numeric columns -> float64, string columns -> str
        # Row ID: fill NaN with 0 then convert to int32
        results_df["Row ID"] = results_df["Row ID"].fillna(0).astype("int32")

        # Numeric columns expected as double in configure()
        numeric_cols = [
            "Sum of Squares",
            "df",
            "Mean Square",
            "F",
            "ANOVA P-Value",
            "Mean Difference",
            "Lower CI",
            "Upper CI",
            "P-Value",
            "T-Statistic",
            "Raw P-Value",
            "Corrected P-Value",
            "N",
            "Group Mean",
            "Group Std Dev",
            "Significance Level",
        ]
        for col in numeric_cols:
            if col in results_df.columns:
                # coerce non-numeric values to NaN, then ensure float dtype
                results_df[col] = pd.to_numeric(results_df[col], errors="coerce").astype(float)

        # String columns: ensure no numeric or NaN types remain; convert to string and clean 'nan'/'None'
        string_cols = ["Result Type", "Test Column", "Source", "Comparison", "Group 1", "Group 2", "Reject H0", "Group", "Test Method", "Significant"]
        for col in string_cols:
            if col in results_df.columns:
                # Replace numpy NaN with empty string and cast to str
                results_df[col] = results_df[col].where(results_df[col].notnull(), "")
                results_df[col] = results_df[col].astype(str)
                results_df[col] = results_df[col].replace({"nan": "", "None": ""})

        # Split unified results into three cleaned DataFrames
        anova_df, pairwise_df, group_summary_df = split_posthoc_results(results_df)

        # Helper to prepare each DataFrame to match the configure() schema and KNIME expectations
        def _prepare_and_convert(df, schema_cols, numeric_cols, string_cols):
            # Ensure all expected columns exist
            for c in schema_cols:
                if c not in df.columns:
                    df[c] = np.nan
            # Reorder to schema
            df = df[schema_cols].copy()

            # Numeric coercion
            for c in numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

            # String coercion
            for c in string_cols:
                if c in df.columns:
                    df[c] = df[c].where(df[c].notnull(), "")
                    df[c] = df[c].astype(str)
                    df[c] = df[c].replace({"nan": "", "None": ""})

            # Reset index for cleanliness
            df = df.reset_index(drop=True)
            return knext.Table.from_pandas(df)

        # Define schema column names and dtype groupings for each output (match configure())
        anova_schema_cols = [
            "Result Type",
            "Test Column",
            "Source",
            "Sum of Squares",
            "df",
            "Mean Square",
            "F",
            "ANOVA P-Value",
            "Test Method",
            "Significance Level",
            "Significant",
        ]
        anova_numeric = ["Sum of Squares", "df", "Mean Square", "F", "ANOVA P-Value", "Significance Level"]
        anova_string = ["Result Type", "Test Column", "Source", "Test Method", "Significant"]

        pairwise_schema_cols = [
            "Result Type",
            "Test Column",
            "Comparison",
            "Group 1",
            "Group 2",
            "Mean Difference",
            "Lower CI",
            "Upper CI",
            "P-Value",
            "Corrected P-Value",
            "Reject H0",
            "Test Method",
            "Significance Level",
            "Significant",
        ]
        pairwise_numeric = ["Mean Difference", "Lower CI", "Upper CI", "P-Value", "Corrected P-Value", "Significance Level"]
        pairwise_string = ["Result Type", "Test Column", "Comparison", "Group 1", "Group 2", "Reject H0", "Test Method", "Significant"]

        group_schema_cols = ["Result Type", "Test Column", "Group", "N", "Group Mean", "Group Std Dev", "Test Method", "Significance Level"]
        group_numeric = ["N", "Group Mean", "Group Std Dev", "Significance Level"]
        group_string = ["Result Type", "Test Column", "Group", "Test Method"]

        anova_table = _prepare_and_convert(anova_df, anova_schema_cols, anova_numeric, anova_string)
        pairwise_table = _prepare_and_convert(pairwise_df, pairwise_schema_cols, pairwise_numeric, pairwise_string)
        group_table = _prepare_and_convert(group_summary_df, group_schema_cols, group_numeric, group_string)

        # Return tables in same order as configure() and decorators
        return anova_table, pairwise_table, group_table


def split_posthoc_results(results_df: pd.DataFrame):
    # High-level: split the unified results DataFrame (created in execute)
    # into three separate DataFrames (ANOVA, Pairwise, Group Summary),
    # preserving important metadata columns and coercing types.

    if "Result Type" not in results_df.columns:
        raise ValueError("Input DataFrame must contain a 'Result Type' column")

    # Work on a copy to avoid mutating caller data
    df = results_df.copy()

    # Drop Row ID if present
    if "Row ID" in df.columns:
        df = df.drop(columns=["Row ID"])

    # Define target columns for each output table
    anova_cols = [
        "Result Type",
        "Test Column",
        "Source",
        "Sum of Squares",
        "df",
        "Mean Square",
        "F",
        "ANOVA P-Value",
        "Test Method",
        "Significance Level",
        "Significant",
    ]

    pairwise_cols = [
        "Result Type",
        "Test Column",
        "Comparison",
        "Group 1",
        "Group 2",
        "Mean Difference",
        "Lower CI",
        "Upper CI",
        "P-Value",
        "Corrected P-Value",
        "Reject H0",
        "Test Method",
        "Significance Level",
        "Significant",
    ]

    group_summary_cols = ["Result Type", "Test Column", "Group", "N", "Group Mean", "Group Std Dev", "Test Method", "Significance Level"]

    # Helper to safely select and ensure columns exist (fill with NaN if missing)
    def _select_and_fill(row_filter, cols):
        sub = df.loc[row_filter, :].copy()
        for c in cols:
            if c not in sub.columns:
                sub[c] = np.nan
        # keep only the requested columns in the requested order
        sub = sub[cols]
        # reset index for cleanliness
        sub = sub.reset_index(drop=True)
        return sub

    anova_df = _select_and_fill(df["Result Type"] == "ANOVA", anova_cols)
    pairwise_df = _select_and_fill(df["Result Type"] == "Pairwise Comparison", pairwise_cols)
    group_summary_df = _select_and_fill(df["Result Type"] == "Group Summary", group_summary_cols)

    # Coerce numeric columns to numeric dtypes where appropriate
    numeric_map = {
        "Sum of Squares": float,
        "df": float,
        "Mean Square": float,
        "F": float,
        "ANOVA P-Value": float,
        "Mean Difference": float,
        "Lower CI": float,
        "Upper CI": float,
        "P-Value": float,
        "Corrected P-Value": float,
        "N": float,
        "Group Mean": float,
        "Group Std Dev": float,
        "Significance Level": float,
    }

    for col, dtype in numeric_map.items():
        for tbl in (anova_df, pairwise_df, group_summary_df):
            if col in tbl.columns:
                tbl[col] = pd.to_numeric(tbl[col], errors="coerce").astype(float)

    # Convert boolean-like string fields to boolean where possible
    def _coerce_bool_series(s: pd.Series):
        if s.dtype == object or s.dtype == str:
            # Map common string representations to booleans; preserve empty string -> NaN
            mapped = s.replace({"True": True, "False": False, "true": True, "false": False, "": None})
            # If after mapping values are booleans or None, return that dtype
            if mapped.dropna().isin([True, False]).all():
                return mapped.where(mapped.notnull(), None).astype("object")
        return s

    # Apply boolean coercion to 'Significant' and 'Reject H0' if present
    for tbl in (anova_df, pairwise_df, group_summary_df):
        for bcol in ("Significant", "Reject H0"):
            if bcol in tbl.columns:
                tbl[bcol] = _coerce_bool_series(tbl[bcol])

    return anova_df, pairwise_df, group_summary_df
