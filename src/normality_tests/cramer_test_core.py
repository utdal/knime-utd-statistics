"""
Cramer-von Mises normality test computational core.

Simple implementation using scipy.
"""

from scipy.stats import cramervonmises, norm


def run_cramer_test(data, alpha=0.05):
    """
    Simple Cramer-von Mises test using scipy.

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
    result = cramervonmises(data, norm.cdf)
    decision = "Reject normality" if result.pvalue <= alpha else "Do not reject normality"

    return {"test": "Cramer-von Mises", "statistic": result.statistic, "p_value": result.pvalue, "decision": decision, "n": len(data)}
