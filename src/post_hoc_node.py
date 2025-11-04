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
    validate_group_data,
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
@knext.input_table(
    name="Data", 
    description="Data table with numeric dependent variable and categorical grouping variable."
)
@knext.output_table(
    name="Results",
    description="Comprehensive post-hoc analysis results including ANOVA, pairwise comparisons, and group statistics.",
)
class PostHocTestsNode:
    """
    Post-hoc multiple comparison testing node with integrated ANOVA analysis.
    
    This node performs:
    1. One-way ANOVA analysis on the input data
    2. Post-hoc pairwise comparisons (if ANOVA is significant)
    3. Returns ANOVA results, pairwise comparisons, and group summary statistics
    
    Post-hoc tests are only performed when ANOVA shows significant group differences (p ≤ α).
    """

    test_type = test_type_param
    data_column = data_column_param
    group_column = group_column_param
    alpha = alpha_param

    def _validate_and_prepare_data(self, df, data_col, group_col):
        """
        Validate and prepare data for ANOVA and post-hoc analysis.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data table
        data_col : str
            Name of dependent variable column
        group_col : str
            Name of grouping variable column
            
        Returns:
        --------
        tuple
            (validated_data, validated_groups)
            
        Raises:
        -------
        ValueError
            If data validation fails
        """
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
        
        # Unified results table schema combining ANOVA, pairwise, and summary information
        results_cols = [
            # Result type identifier
            knext.Column(knext.string(), "Result Type"),
            knext.Column(knext.int32(), "Row ID"),
            
            # ANOVA-specific columns
            knext.Column(knext.string(), "Test Column"),
            knext.Column(knext.string(), "Source"),
            knext.Column(knext.double(), "Sum of Squares"),
            knext.Column(knext.double(), "df"),  # Changed to double to allow NaN
            knext.Column(knext.double(), "Mean Square"),
            knext.Column(knext.double(), "F"),
            knext.Column(knext.double(), "ANOVA P-Value"),
            
            # Pairwise comparison columns
            knext.Column(knext.string(), "Comparison"),
            knext.Column(knext.string(), "Group 1"),
            knext.Column(knext.string(), "Group 2"),
            knext.Column(knext.double(), "Mean Difference"),
            knext.Column(knext.double(), "Lower CI"),
            knext.Column(knext.double(), "Upper CI"),
            knext.Column(knext.double(), "P-Value"),
            knext.Column(knext.double(), "T-Statistic"),
            knext.Column(knext.double(), "Raw P-Value"),
            knext.Column(knext.double(), "Corrected P-Value"),
            knext.Column(knext.string(), "Reject H0"),
            
            # Group summary columns
            knext.Column(knext.string(), "Group"),
            knext.Column(knext.double(), "N"),  # Changed to double to allow NaN
            knext.Column(knext.double(), "Group Mean"),
            knext.Column(knext.double(), "Group Std Dev"),
            
            # Common columns
            knext.Column(knext.string(), "Test Method"),
            knext.Column(knext.double(), "Significance Level"),
            knext.Column(knext.string(), "Significant"),
        ]
        
        results_schema = knext.Schema.from_columns(results_cols)
        return results_schema

    def execute(self, exec_ctx, input_table):
        """Execute the integrated ANOVA and post-hoc analysis."""
        
        # Convert input table to pandas DataFrame
        df = input_table.to_pandas()
        
        # Step 1: Validate and prepare data
        data, groups = self._validate_and_prepare_data(
            df, self.data_column, self.group_column
        )
        
        # Step 2: Run ANOVA analysis
        anova_results = run_one_way_anova(data, groups, alpha=self.alpha)
        
        # Step 3: Initialize results list for unified table
        all_results = []
        
        # Add ANOVA results to unified table
        anova_df = format_anova_results_for_knime(
            anova_results, 
            data_column_name=self.data_column,
            group_column_name=self.group_column
        )
        
        row_id_counter = 0  # Initialize sequential row ID counter
        
        for _, row in anova_df.iterrows():
            result_row = {
                'Result Type': 'ANOVA',
                'Row ID': row_id_counter,  # Use sequential counter instead of idx
                'Test Column': row['Test Column'],
                'Source': row['Source'],
                'Sum of Squares': row['Sum of Squares'],
                'df': row['df'],
                'Mean Square': row['Mean Square'],
                'F': row['F'],
                'ANOVA P-Value': row['p-value'],
                'Test Method': row['Test Method'],
                'Significance Level': row['Significance Level'],
                'Significant': row['Significant'],
                # Pairwise columns as NaN
                'Comparison': np.nan,
                'Group 1': np.nan,
                'Group 2': np.nan,
                'Mean Difference': np.nan,
                'Lower CI': np.nan,
                'Upper CI': np.nan,
                'P-Value': np.nan,
                'T-Statistic': np.nan,
                'Raw P-Value': np.nan,
                'Corrected P-Value': np.nan,
                'Reject H0': np.nan,
                # Group summary columns as NaN
                'Group': np.nan,
                'N': np.nan,
                'Group Mean': np.nan,
                'Group Std Dev': np.nan,
            }
            all_results.append(result_row)
            row_id_counter += 1  # Increment counter
        
        # Step 4: Check ANOVA significance and add post-hoc/summary results
        if anova_results['significant']:
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
                result_row = {
                    'Result Type': 'Pairwise Comparison',
                    'Row ID': row_id_counter,  # Use sequential counter instead of idx
                    'Test Column': self.data_column,
                    'ANOVA P-Value': anova_results['summary']['p_value'],
                    'Comparison': row['Comparison'],
                    'Group 1': row['Group 1'],
                    'Group 2': row['Group 2'],
                    'Mean Difference': row['Mean Difference'],
                    'Lower CI': row['Lower CI'],
                    'Upper CI': row['Upper CI'],
                    'P-Value': row['P-Value'],
                    'T-Statistic': row.get('T-Statistic', np.nan),
                    'Raw P-Value': row.get('Raw P-Value', np.nan),
                    'Corrected P-Value': row.get('Corrected P-Value', np.nan),
                    'Reject H0': row['Reject H0'],
                    'Test Method': row['Test Method'],
                    'Significance Level': row['Significance Level'],
                    'Significant': str(anova_results['significant']),
                    # ANOVA columns as NaN (except already filled)
                    'Source': np.nan,
                    'Sum of Squares': np.nan,
                    'df': np.nan,
                    'Mean Square': np.nan,
                    'F': np.nan,
                    # Group summary columns as NaN
                    'Group': np.nan,
                    'N': np.nan,
                    'Group Mean': np.nan,
                    'Group Std Dev': np.nan,
                }
                all_results.append(result_row)
                row_id_counter += 1  # Increment counter
            
            # Add group summary results
            for _, row in group_summary_df.iterrows():
                result_row = {
                    'Result Type': 'Group Summary',
                    'Row ID': row_id_counter,  # Use sequential counter instead of idx
                    'Test Column': self.data_column,
                    'ANOVA P-Value': anova_results['summary']['p_value'],
                    'Group': row['Group'],
                    'N': row['N'],
                    'Group Mean': row['Mean'],
                    'Group Std Dev': row['Std Dev'],
                    'Test Method': row['Test Method'],
                    'Significance Level': self.alpha,
                    'Significant': str(anova_results['significant']),
                    # ANOVA columns as NaN
                    'Source': np.nan,
                    'Sum of Squares': np.nan,
                    'df': np.nan,
                    'Mean Square': np.nan,
                    'F': np.nan,
                    # Pairwise columns as NaN
                    'Comparison': np.nan,
                    'Group 1': np.nan,
                    'Group 2': np.nan,
                    'Mean Difference': np.nan,
                    'Lower CI': np.nan,
                    'Upper CI': np.nan,
                    'P-Value': np.nan,
                    'T-Statistic': np.nan,
                    'Raw P-Value': np.nan,
                    'Corrected P-Value': np.nan,
                    'Reject H0': np.nan,
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
                    'Result Type': 'Group Summary',
                    'Row ID': row_id_counter,  # Use sequential counter instead of idx
                    'Test Column': self.data_column,
                    'ANOVA P-Value': anova_results['summary']['p_value'],
                    'Group': group,
                    'N': len(group_data),
                    'Group Mean': np.mean(group_data),
                    'Group Std Dev': np.std(group_data, ddof=1),
                    'Test Method': empty_message,
                    'Significance Level': self.alpha,
                    'Significant': str(anova_results['significant']),
                    # ANOVA columns as NaN
                    'Source': np.nan,
                    'Sum of Squares': np.nan,
                    'df': np.nan,
                    'Mean Square': np.nan,
                    'F': np.nan,
                    # Pairwise columns as NaN
                    'Comparison': np.nan,
                    'Group 1': np.nan,
                    'Group 2': np.nan,
                    'Mean Difference': np.nan,
                    'Lower CI': np.nan,
                    'Upper CI': np.nan,
                    'P-Value': np.nan,
                    'T-Statistic': np.nan,
                    'Raw P-Value': np.nan,
                    'Corrected P-Value': np.nan,
                    'Reject H0': np.nan,
                }
                all_results.append(result_row)
                row_id_counter += 1  # Increment counter
        
        # Create unified results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Ensure proper data types for KNIME to match configure() schema
        results_df['Row ID'] = results_df['Row ID'].astype('int32')
        
        # Ensure numeric columns are float64 (which KNIME double() expects)
        numeric_cols = ['Sum of Squares', 'df', 'Mean Square', 'F', 'ANOVA P-Value',
                       'Mean Difference', 'Lower CI', 'Upper CI', 'P-Value', 
                       'T-Statistic', 'Raw P-Value', 'Corrected P-Value', 
                       'N', 'Group Mean', 'Group Std Dev', 'Significance Level']
        for col in numeric_cols:
            results_df[col] = results_df[col].astype(float)
        
        # Ensure string columns handle NaN properly
        # First replace np.nan with None, then fill None with empty string, then convert to string
        string_cols = ['Result Type', 'Test Column', 'Source', 'Comparison', 
                      'Group 1', 'Group 2', 'Reject H0', 'Group', 
                      'Test Method', 'Significant']
        for col in string_cols:
            # Replace any np.nan with None first
            results_df[col] = results_df[col].replace({np.nan: None})
            # Then fill None with empty string and convert to string
            results_df[col] = results_df[col].fillna('').astype(str)
            # Clean up any 'nan' or 'None' strings that might have slipped through
            results_df[col] = results_df[col].replace({'nan': '', 'None': ''})
        
        # Convert to KNIME table
        results_table = knext.Table.from_pandas(results_df)
        
        return results_table