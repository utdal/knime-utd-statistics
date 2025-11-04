"""
Holm-Bonferroni sequential correction implementation.

This module provides Holm-Bonferroni correction for multiple pairwise 
comparisons using t-tests with sequential p-value adjustment via 
statsmodels.stats.multitest.multipletests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import itertools


def run_bonferroni_test(data, groups, alpha=0.05):
    """
    Perform pairwise t-tests with Holm-Bonferroni correction.
    
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
    
    # Get unique groups and create all pairwise combinations
    unique_groups = np.unique(groups_clean)
    n_groups = len(unique_groups)
    
    # Prepare data structures for results
    comparisons = []
    raw_pvalues = []
    test_statistics = []
    mean_differences = []
    group1_means = []
    group2_means = []
    group1_ns = []
    group2_ns = []
    
    # Perform all pairwise t-tests
    for i, group1 in enumerate(unique_groups):
        for j, group2 in enumerate(unique_groups):
            if i < j:  # Only do each comparison once
                # Extract data for each group
                data1 = data_clean[groups_clean == group1]
                data2 = data_clean[groups_clean == group2]
                
                # Perform independent t-test (assuming unequal variances)
                t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
                
                # Store results
                comparisons.append(f"{group1} vs {group2}")
                raw_pvalues.append(p_val)
                test_statistics.append(t_stat)
                mean_differences.append(np.mean(data1) - np.mean(data2))
                group1_means.append(np.mean(data1))
                group2_means.append(np.mean(data2))
                group1_ns.append(len(data1))
                group2_ns.append(len(data2))
    
    # Apply Holm-Bonferroni correction
    rejected, corrected_pvalues, alpha_sidak, alpha_bonf = multipletests(
        raw_pvalues, 
        alpha=alpha, 
        method='holm'
    )
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Comparison': comparisons,
        'Group 1': [comp.split(' vs ')[0] for comp in comparisons],
        'Group 2': [comp.split(' vs ')[1] for comp in comparisons],
        'Mean Difference': mean_differences,
        'T-Statistic': test_statistics,
        'Raw P-Value': raw_pvalues,
        'Corrected P-Value': corrected_pvalues,
        'Reject H0': rejected,
        'Group 1 Mean': group1_means,
        'Group 2 Mean': group2_means,
        'Group 1 N': group1_ns,
        'Group 2 N': group2_ns,
    })
    
    # Calculate confidence intervals (approximate)
    # Using pooled standard error for CI calculation
    cis_lower = []
    cis_upper = []
    
    for idx, row in results_df.iterrows():
        group1 = row['Group 1']
        group2 = row['Group 2']
        
        data1 = data_clean[groups_clean == group1]
        data2 = data_clean[groups_clean == group2]
        
        # Calculate pooled standard error
        n1, n2 = len(data1), len(data2)
        s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        
        # Welch's t-test degrees of freedom
        se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
        df = ((s1**2 / n1) + (s2**2 / n2))**2 / ((s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1))
        
        # Critical value for confidence interval (using corrected alpha)
        corrected_alpha_per_test = row['Corrected P-Value']
        if corrected_alpha_per_test < 1.0:  # Avoid issues with p-values >= 1
            try:
                t_crit = stats.t.ppf(1 - corrected_alpha_per_test/2, df)
                margin_error = t_crit * se
                
                ci_lower = row['Mean Difference'] - margin_error
                ci_upper = row['Mean Difference'] + margin_error
            except:
                # Fallback if calculation fails
                ci_lower = np.nan
                ci_upper = np.nan
        else:
            ci_lower = np.nan
            ci_upper = np.nan
            
        cis_lower.append(ci_lower)
        cis_upper.append(ci_upper)
    
    results_df['Lower CI'] = cis_lower
    results_df['Upper CI'] = cis_upper
    
    # Calculate summary statistics
    n_comparisons = len(results_df)
    n_significant = results_df['Reject H0'].sum()
    
    # Get group summary statistics
    group_summary = pd.DataFrame({
        'Group': unique_groups,
        'N': [np.sum(groups_clean == group) for group in unique_groups],
        'Mean': [np.mean(data_clean[groups_clean == group]) for group in unique_groups],
        'Std Dev': [np.std(data_clean[groups_clean == group], ddof=1) for group in unique_groups],
    })
    
    summary = {
        'n_groups': n_groups,
        'n_comparisons': n_comparisons,
        'n_significant': int(n_significant),
        'family_wise_error_rate': alpha,
        'method': 'Holm-Bonferroni Sequential Correction',
        'correction_method': 'holm',
        'group_summary': group_summary,
    }
    
    # Reorder columns for better readability
    results_df = results_df[[
        'Comparison', 'Group 1', 'Group 2', 'Mean Difference', 
        'Lower CI', 'Upper CI', 'T-Statistic', 'Raw P-Value', 
        'Corrected P-Value', 'Reject H0'
    ]]
    
    return {
        'test': 'Holm-Bonferroni',
        'alpha': alpha,
        'n_comparisons': n_comparisons,
        'results': results_df,
        'summary': summary,
    }


def format_bonferroni_results_for_knime(bonferroni_output):
    """
    Format Holm-Bonferroni test results for KNIME output tables.
    
    Parameters:
    -----------
    bonferroni_output : dict
        Output from run_bonferroni_test function
        
    Returns:
    --------
    tuple
        (pairwise_results_df, group_summary_df) formatted for KNIME
    """
    
    # Format pairwise comparisons table
    pairwise_df = bonferroni_output['results'].copy()
    
    # Add unified P-Value column (use corrected p-value as primary)
    pairwise_df['P-Value'] = pairwise_df['Corrected P-Value']
    
    # Add test method and alpha for context
    pairwise_df['Test Method'] = bonferroni_output['test']
    pairwise_df['Significance Level'] = bonferroni_output['alpha']
    
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
    group_summary_df = bonferroni_output['summary']['group_summary'].copy()
    group_summary_df['Test Method'] = bonferroni_output['test']
    group_summary_df['N'] = group_summary_df['N'].astype(float)  # Keep as float to allow NaN values
    group_summary_df['Mean'] = group_summary_df['Mean'].astype(float)
    group_summary_df['Std Dev'] = group_summary_df['Std Dev'].astype(float)
    
    return pairwise_df, group_summary_df