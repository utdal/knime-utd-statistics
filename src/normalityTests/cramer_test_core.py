"""
Cramer-von Mises normality test computational core.

Simple implementation using scipy.
"""

import numpy as np
from scipy.stats import cramervonmises, norm


def run_cramer_test(data):
    """
    Simple Cramer-von Mises test using scipy.
    
    Parameters:
    -----------
    data : array-like
        Numeric data to test for normality
        
    Returns:
    --------
    dict
        Dictionary with test results
    """
    result = cramervonmises(data, norm.cdf)
    decision = "Reject normality" if result.pvalue <= 0.05 else "Do not reject normality"
    
    return {
        "test": "Cramer-von Mises", 
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "decision": decision,
        "n": len(data)
    }
