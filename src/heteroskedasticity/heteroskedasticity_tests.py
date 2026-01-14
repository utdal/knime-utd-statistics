import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, het_goldfeldquandt
from typing import Dict


def run_breusch_pagan_test(model: sm.regression.linear_model.RegressionResultsWrapper, X: pd.DataFrame, alpha: float = 0.05) -> Dict:
    # Add constant to match model structure
    X_with_const = sm.add_constant(X, has_constant="add")

    # Run Breusch-Pagan test
    # Returns: (lm_statistic, lm_pvalue, f_statistic, f_pvalue)
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(model.resid, X_with_const)

    # Make decision based on p-value
    is_heteroskedastic = lm_pvalue < alpha

    if is_heteroskedastic:
        interpretation = (
            f"The test detected heteroskedasticity (p-value = {lm_pvalue:.4f} < {alpha}). "
            "The variance of residuals is not constant across predictor values. "
            "Consider using robust standard errors or transforming your variables."
        )
    else:
        interpretation = (
            f"No significant heteroskedasticity detected (p-value = {lm_pvalue:.4f} ≥ {alpha}). "
            "The assumption of constant error variance appears to be satisfied."
        )

    return {
        "test": "Breusch-Pagan",
        "statistic": float(lm_stat),
        "p_value": float(lm_pvalue),
        "is_heteroskedastic": "True" if is_heteroskedastic else "False",
        "interpretation": interpretation,
    }


def run_white_test(model: sm.regression.linear_model.RegressionResultsWrapper, X: pd.DataFrame, alpha: float = 0.05) -> Dict:
    # Add constant to match model structure
    X_with_const = sm.add_constant(X, has_constant="add")

    # Run White's test
    # Returns: (lm_statistic, lm_pvalue, f_statistic, f_pvalue)
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(model.resid, X_with_const)

    # Make decision based on p-value
    is_heteroskedastic = lm_pvalue < alpha

    if is_heteroskedastic:
        interpretation = (
            f"White's test detected heteroskedasticity (p-value = {lm_pvalue:.4f} < {alpha}). "
            "The variance of residuals is not constant. This test is robust to non-normality "
            "and detects complex patterns. Consider using robust standard errors."
        )
    else:
        interpretation = (
            f"No significant heteroskedasticity detected (p-value = {lm_pvalue:.4f} ≥ {alpha}). "
            "The assumption of constant error variance appears to be satisfied."
        )

    return {
        "test": "White",
        "statistic": float(lm_stat),
        "p_value": float(lm_pvalue),
        "is_heteroskedastic": "True" if is_heteroskedastic else "False",
        "interpretation": interpretation,
    }


def run_goldfeld_quandt_test(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X: pd.DataFrame,
    y: pd.Series,
    sort_variable: str,
    split_fraction: float = 0.5,
    alpha: float = 0.05,
) -> Dict:
    # Validate sort variable
    if sort_variable not in X.columns:
        raise ValueError(f"Sort variable '{sort_variable}' not found in predictors. Available predictors: {X.columns.tolist()}")

    # Add constant
    X_with_const = sm.add_constant(X, has_constant="add")

    # Get the index of the sort variable (after adding constant, so +1)
    sort_idx = list(X.columns).index(sort_variable) + 1  # +1 because const is at index 0

    # Run Goldfeld-Quandt test
    # Returns: (F_statistic, p_value, ordering)
    # The test automatically sorts by the specified column
    f_stat, gq_pvalue, ordering = het_goldfeldquandt(y, X_with_const, idx=sort_idx, split=split_fraction)

    # Make decision based on p-value
    is_heteroskedastic = gq_pvalue < alpha

    # Calculate how many observations in each group
    n_total = len(y)
    n_per_group = int(n_total * split_fraction / 2)
    n_middle_omitted = n_total - (2 * n_per_group)

    if is_heteroskedastic:
        interpretation = (
            f"The Goldfeld-Quandt test detected heteroskedasticity (p-value = {gq_pvalue:.4f} < {alpha}). "
            f"Data was sorted by '{sort_variable}'. "
            f"The variance differs significantly between the bottom {n_per_group} and top {n_per_group} observations "
            f"({n_middle_omitted} middle observations were omitted). "
            "Consider using robust standard errors or variable transformations."
        )
    else:
        interpretation = (
            f"No significant heteroskedasticity detected (p-value = {gq_pvalue:.4f} ≥ {alpha}). "
            f"The variance is similar between low and high values of '{sort_variable}'. "
            "The assumption of constant error variance appears satisfied."
        )

    return {
        "test": "Goldfeld-Quandt",
        "statistic": float(f_stat),
        "p_value": float(gq_pvalue),
        "is_heteroskedastic": "True" if is_heteroskedastic else "False",
        "interpretation": interpretation,
        "sort_variable": sort_variable,
        "split_fraction": split_fraction,
    }
