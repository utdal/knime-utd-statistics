"""
Heteroskedasticity test implementations.

What is Heteroskedasticity?
----------------------------
Heteroskedasticity means the variance (spread) of regression errors is not constant.

Example: When predicting house prices
- For cheap houses ($100k), prediction errors might be ±$5k
- For expensive houses ($1M), prediction errors might be ±$100k
- The error variance is increasing with house value → heteroskedastic!

Why does it matter?
-------------------
OLS regression assumes homoskedasticity (constant error variance).
When this assumption is violated:
- Standard errors are wrong
- P-values are unreliable
- Confidence intervals are inaccurate
- Your conclusions may be invalid

These tests help detect heteroskedasticity so you can:
1. Use robust standard errors
2. Transform your variables
3. Use a different modeling approach

Available Tests
---------------
1. Breusch-Pagan: Tests if error variance correlates with predictors (most common)
2. White: More general test, doesn't assume specific patterns (robust)
3. Goldfeld-Quandt: Compares variance between sorted subgroups (simple, intuitive)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, het_goldfeldquandt
from typing import Dict


def run_breusch_pagan_test(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X: pd.DataFrame,
    alpha: float = 0.05
) -> Dict:
    """
    Run the Breusch-Pagan test for heteroskedasticity.
    
    What it does:
    -------------
    Tests if the variance of residuals is related to the predictor variables.
    It assumes residuals are normally distributed.
    
    How it works:
    -------------
    1. Fit the original regression
    2. Square the residuals
    3. Regress squared residuals on the original predictors
    4. Test if any predictor significantly explains the squared residuals
    
    When to use:
    ------------
    - Standard choice for most situations
    - When you suspect variance increases/decreases with predictors
    - Assumes normally distributed errors
    
    Interpretation:
    ---------------
    - High p-value (> α): Homoskedastic (good! Constant variance)
    - Low p-value (< α): Heteroskedastic (problem! Non-constant variance)
    
    Args:
        model: Fitted OLS model
        X: DataFrame of predictor variables (without constant)
        alpha: Significance level for decision (default 0.05)
        
    Returns:
        Dictionary with:
        - test: Name of test
        - statistic: LM test statistic
        - p_value: P-value for the test
        - decision: "Homoskedastic" or "Heteroskedastic"
        - interpretation: Plain English explanation
    """
    # Add constant to match model structure
    X_with_const = sm.add_constant(X, has_constant='add')
    
    # Run Breusch-Pagan test
    # Returns: (lm_statistic, lm_pvalue, f_statistic, f_pvalue)
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(model.resid, X_with_const)
    
    # Make decision based on p-value
    is_heteroskedastic = lm_pvalue < alpha
    
    if is_heteroskedastic:
        decision = "Heteroskedastic"
        interpretation = (
            f"The test detected heteroskedasticity (p-value = {lm_pvalue:.4f} < {alpha}). "
            "The variance of residuals is not constant across predictor values. "
            "Consider using robust standard errors or transforming your variables."
        )
    else:
        decision = "Homoskedastic"
        interpretation = (
            f"No significant heteroskedasticity detected (p-value = {lm_pvalue:.4f} ≥ {alpha}). "
            "The assumption of constant error variance appears to be satisfied."
        )
    
    return {
        'test': 'Breusch-Pagan',
        'statistic': float(lm_stat),
        'p_value': float(lm_pvalue),
        'decision': decision,
        'interpretation': interpretation
    }


def run_white_test(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X: pd.DataFrame,
    alpha: float = 0.05
) -> Dict:
    """
    Run White's test for heteroskedasticity.
    
    What it does:
    -------------
    A general test that doesn't assume a specific form of heteroskedasticity.
    It doesn't require normally distributed errors.
    
    How it works:
    -------------
    1. Fit the original regression
    2. Square the residuals
    3. Regress squared residuals on predictors, their squares, and cross-products
    4. Test if this expanded model explains the squared residuals
    
    When to use:
    ------------
    - When you want a robust, general test
    - When you don't want to assume normal distribution
    - When patterns of heteroskedasticity might be complex
    
    Trade-off:
    ----------
    More powerful but uses more degrees of freedom (can be problematic with
    many predictors or small sample sizes).
    
    Interpretation:
    ---------------
    - High p-value (> α): Homoskedastic (good! Constant variance)
    - Low p-value (< α): Heteroskedastic (problem! Non-constant variance)
    
    Args:
        model: Fitted OLS model
        X: DataFrame of predictor variables (without constant)
        alpha: Significance level for decision (default 0.05)
        
    Returns:
        Dictionary with:
        - test: Name of test
        - statistic: LM test statistic
        - p_value: P-value for the test
        - decision: "Homoskedastic" or "Heteroskedastic"
        - interpretation: Plain English explanation
    """
    # Add constant to match model structure
    X_with_const = sm.add_constant(X, has_constant='add')
    
    # Run White's test
    # Returns: (lm_statistic, lm_pvalue, f_statistic, f_pvalue)
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(model.resid, X_with_const)
    
    # Make decision based on p-value
    is_heteroskedastic = lm_pvalue < alpha
    
    if is_heteroskedastic:
        decision = "Heteroskedastic"
        interpretation = (
            f"White's test detected heteroskedasticity (p-value = {lm_pvalue:.4f} < {alpha}). "
            "The variance of residuals is not constant. This test is robust to non-normality "
            "and detects complex patterns. Consider using robust standard errors."
        )
    else:
        decision = "Homoskedastic"
        interpretation = (
            f"No significant heteroskedasticity detected (p-value = {lm_pvalue:.4f} ≥ {alpha}). "
            "The assumption of constant error variance appears to be satisfied."
        )
    
    return {
        'test': 'White',
        'statistic': float(lm_stat),
        'p_value': float(lm_pvalue),
        'decision': decision,
        'interpretation': interpretation
    }


def run_goldfeld_quandt_test(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X: pd.DataFrame,
    y: pd.Series,
    sort_variable: str,
    split_fraction: float = 0.5,
    alpha: float = 0.05
) -> Dict:
    """
    Run the Goldfeld-Quandt test for heteroskedasticity.
    
    What it does:
    -------------
    Compares the variance of residuals between two subgroups of your data.
    It sorts data by a chosen variable and compares low vs. high values.
    
    How it works:
    -------------
    1. Sort the data by the chosen variable (e.g., income)
    2. Split into bottom X% and top X% (default X=50)
    3. Fit separate regressions on each group
    4. Compare the variance of residuals between groups
    5. Use an F-test to see if variances are significantly different
    
    When to use:
    ------------
    - When you suspect variance changes with a specific variable
    - When you want an intuitive, easy-to-understand test
    - When you have a moderate to large sample size
    
    Example:
    --------
    If testing income → spending:
    - Sort by income
    - Compare variance in bottom 50% of earners vs. top 50%
    - If high earners have much more variable spending → heteroskedastic!
    
    Interpretation:
    ---------------
    - High p-value (> α): Homoskedastic (similar variance in both groups)
    - Low p-value (< α): Heteroskedastic (variance differs between groups)
    
    Args:
        model: Fitted OLS model (used for reference, test refits on subgroups)
        X: DataFrame of predictor variables (without constant)
        y: Series of target variable
        sort_variable: Which variable to sort by before splitting
        split_fraction: Proportion to use in each group (default 0.5 = 50%)
        alpha: Significance level for decision (default 0.05)
        
    Returns:
        Dictionary with:
        - test: Name of test
        - statistic: F-statistic comparing variances
        - p_value: P-value for the test
        - decision: "Homoskedastic" or "Heteroskedastic"
        - interpretation: Plain English explanation
        - sort_variable: Which variable was used for sorting
        
    Raises:
        ValueError: If sort_variable not in predictors
    """
    # Validate sort variable
    if sort_variable not in X.columns:
        raise ValueError(
            f"Sort variable '{sort_variable}' not found in predictors. "
            f"Available predictors: {X.columns.tolist()}"
        )
    
    # Add constant
    X_with_const = sm.add_constant(X, has_constant='add')
    
    # Get the index of the sort variable (after adding constant, so +1)
    sort_idx = list(X.columns).index(sort_variable) + 1  # +1 because const is at index 0
    
    # Run Goldfeld-Quandt test
    # Returns: (F_statistic, p_value, ordering)
    # The test automatically sorts by the specified column
    f_stat, gq_pvalue, ordering = het_goldfeldquandt(
        y, 
        X_with_const, 
        idx=sort_idx,
        split=split_fraction
    )
    
    # Make decision based on p-value
    is_heteroskedastic = gq_pvalue < alpha
    
    # Calculate how many observations in each group
    n_total = len(y)
    n_per_group = int(n_total * split_fraction / 2)
    n_middle_omitted = n_total - (2 * n_per_group)
    
    if is_heteroskedastic:
        decision = "Heteroskedastic"
        interpretation = (
            f"The Goldfeld-Quandt test detected heteroskedasticity (p-value = {gq_pvalue:.4f} < {alpha}). "
            f"Data was sorted by '{sort_variable}'. "
            f"The variance differs significantly between the bottom {n_per_group} and top {n_per_group} observations "
            f"({n_middle_omitted} middle observations were omitted). "
            "Consider using robust standard errors or variable transformations."
        )
    else:
        decision = "Homoskedastic"
        interpretation = (
            f"No significant heteroskedasticity detected (p-value = {gq_pvalue:.4f} ≥ {alpha}). "
            f"The variance is similar between low and high values of '{sort_variable}'. "
            "The assumption of constant error variance appears satisfied."
        )
    
    return {
        'test': 'Goldfeld-Quandt',
        'statistic': float(f_stat),
        'p_value': float(gq_pvalue),
        'decision': decision,
        'interpretation': interpretation,
        'sort_variable': sort_variable,
        'split_fraction': split_fraction
    }
