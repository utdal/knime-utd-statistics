"""
Anderson-Darling normality test computational core.

Simple implementation using statsmodels.
"""

from statsmodels.stats.diagnostic import normal_ad


def run_ad_test(data, alpha=0.05):
    """
    Simple Anderson-Darling test using statsmodels.
    Parameters:
    -----------
    data : array-like
        Numeric data to test for normality
    alpha : float
        Significance level for the test (default: 0.05)
    Returns:
    --------
    dict
        Dictionary with test results
    """
    statistic, p_value = normal_ad(data)
    decision = "Reject normality" if p_value <= alpha else "Do not reject normality"
    return {"test": "Anderson-Darling", "statistic": statistic, "p_value": p_value, "decision": decision, "n": len(data)}
