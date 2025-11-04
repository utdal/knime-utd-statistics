"""
Tukey HSD (Honest Significant Difference) test implementation.

This module provides Tukey's HSD test for multiple pairwise comparisons
using statsmodels, with support for unbalanced designs (Tukey-Kramer).
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools


def run_tukey_test(data, groups, alpha=0.05):
    """
    Perform Tukey HSD test for multiple pairwise comparisons.
    
    Parameters:
    -----------
    data : array-like
        Numeric dependent variable values
    groups : array-like
        Group assignment labels
    alpha : float, default=0.05
        Significance level for the test
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'test': str, test name
        - 'alpha': float, significance level used
        - 'n_comparisons': int, number of pairwise comparisons
        - 'results': pd.DataFrame, pairwise comparison results
        - 'summary': dict, test summary statistics
    """
    
    # Convert inputs to numpy arrays for consistency
    data = np.asarray(data)
    groups = np.asarray(groups)
    
    # Remove any missing values
    mask = ~(pd.isna(data) | pd.isna(groups))
    data_clean = data[mask]
    groups_clean = groups[mask]
    
    # Perform Tukey HSD test using statsmodels
    tukey_result = pairwise_tukeyhsd(data_clean, groups_clean, alpha=alpha)
    
    # Extract results into a structured format using simpler approach
    # Get the comparison names directly from the tukey result
    import itertools
    
    # Create all pairwise combinations
    group_pairs = list(itertools.combinations(tukey_result.groupsunique, 2))
    
    results_df = pd.DataFrame({
        'Group 1': [pair[0] for pair in group_pairs],
        'Group 2': [pair[1] for pair in group_pairs],
        'Mean Difference': tukey_result.meandiffs,
        'Lower CI': tukey_result.confint[:, 0],
        'Upper CI': tukey_result.confint[:, 1],
        'P-Value': tukey_result.pvalues,
        'Reject H0': tukey_result.reject,
    })
    
    # Add formatted comparison names
    results_df['Comparison'] = results_df['Group 1'].astype(str) + ' vs ' + results_df['Group 2'].astype(str)
    
    # Calculate summary statistics
    n_groups = len(np.unique(groups_clean))
    n_comparisons = len(results_df)
    n_significant = results_df['Reject H0'].sum()
    
    # Get group means and sample sizes
    group_summary = pd.DataFrame({
        'Group': tukey_result.groupsunique,
        'N': [np.sum(groups_clean == group) for group in tukey_result.groupsunique],
        'Mean': [np.mean(data_clean[groups_clean == group]) for group in tukey_result.groupsunique],
        'Std Dev': [np.std(data_clean[groups_clean == group], ddof=1) for group in tukey_result.groupsunique],
    })
    
    summary = {
        'n_groups': n_groups,
        'n_comparisons': n_comparisons,
        'n_significant': int(n_significant),
        'family_wise_error_rate': alpha,
        'method': 'Tukey HSD (Honest Significant Difference)',
        'group_summary': group_summary,
    }
    
    # Reorder columns for better readability
    results_df = results_df[[
        'Comparison', 'Group 1', 'Group 2', 'Mean Difference', 
        'Lower CI', 'Upper CI', 'P-Value', 'Reject H0'
    ]]
    
    return {
        'test': 'Tukey HSD',
        'alpha': alpha,
        'n_comparisons': n_comparisons,
        'results': results_df,
        'summary': summary,
    }


def format_tukey_results_for_knime(tukey_output):
    """
    Format Tukey test results for KNIME output tables.
    
    Parameters:
    -----------
    tukey_output : dict
        Output from run_tukey_test function
        
    Returns:
    --------
    tuple
        (pairwise_results_df, group_summary_df) formatted for KNIME
    """
    
    # Format pairwise comparisons table
    pairwise_df = tukey_output['results'].copy()
    
    # Add test method and alpha for context
    pairwise_df['Test Method'] = tukey_output['test']
    pairwise_df['Significance Level'] = tukey_output['alpha']
    
    # Add Bonferroni-specific columns as NaN for schema consistency
    pairwise_df['T-Statistic'] = np.nan
    pairwise_df['Raw P-Value'] = np.nan
    pairwise_df['Corrected P-Value'] = np.nan
    
    # Reorder columns to match expected schema
    pairwise_df = pairwise_df[[
        'Comparison', 'Group 1', 'Group 2', 'Mean Difference', 
        'Lower CI', 'Upper CI', 'P-Value', 'T-Statistic', 
        'Raw P-Value', 'Corrected P-Value', 'Reject H0', 
        'Test Method', 'Significance Level'
    ]]
    
    # Ensure proper data types for KNIME
    pairwise_df['Mean Difference'] = pairwise_df['Mean Difference'].astype(float)
    pairwise_df['Lower CI'] = pairwise_df['Lower CI'].astype(float)
    pairwise_df['Upper CI'] = pairwise_df['Upper CI'].astype(float)
    pairwise_df['P-Value'] = pairwise_df['P-Value'].astype(float)
    pairwise_df['T-Statistic'] = pairwise_df['T-Statistic'].astype(float)
    pairwise_df['Raw P-Value'] = pairwise_df['Raw P-Value'].astype(float)
    pairwise_df['Corrected P-Value'] = pairwise_df['Corrected P-Value'].astype(float)
    pairwise_df['Reject H0'] = pairwise_df['Reject H0'].astype(str)  # KNIME prefers string for boolean
    
    # Format group summary table
    group_summary_df = tukey_output['summary']['group_summary'].copy()
    group_summary_df['Test Method'] = tukey_output['test']
    group_summary_df['N'] = group_summary_df['N'].astype(float)  # Keep as float to allow NaN values
    group_summary_df['Mean'] = group_summary_df['Mean'].astype(float)
    group_summary_df['Std Dev'] = group_summary_df['Std Dev'].astype(float)
    
    return pairwise_df, group_summary_df