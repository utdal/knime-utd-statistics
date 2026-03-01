"""
Box's M test for equality of covariance matrices.

This module implements Box's M test with a chi-square approximation,
the standard multivariate assumption check for MANOVA.  A significant
result (typically *p* < 0.001) indicates that the covariance matrices
differ across groups, violating the homogeneity-of-covariance assumption.
"""

import numpy as np
from scipy import stats


def compute_box_m(df, dep_vars, group_col):
    """
    Compute Box's M test for equality of covariance matrices.

    Box's M tests the null hypothesis that the population covariance
    matrices of the dependent variables are equal across all groups.

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame (no missing values) containing the dependent
        variables and grouping column.
    dep_vars : list of str
        Names of the numeric dependent variable columns.
    group_col : str
        Name of the categorical grouping variable column.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``statistic`` : float – Box's M statistic, rounded to 6 decimals.
        - ``chi2_approx`` : float – Chi-square approximation, rounded to 6.
        - ``df`` : float – Degrees of freedom for the chi-square test.
        - ``p_value`` : float – P-value from the chi-square distribution,
          rounded to 8 decimals.
    """

    unique_groups = df[group_col].unique()
    g = len(unique_groups)  # number of groups
    p = len(dep_vars)  # number of dependent variables

    # ------------------------------------------------------------------
    # Per-group covariance matrices and sample sizes
    # ------------------------------------------------------------------
    n_list = []
    S_list = []

    for grp in unique_groups:
        grp_data = df.loc[df[group_col] == grp, dep_vars].values
        n_i = len(grp_data)
        n_list.append(n_i)
        S_list.append(np.cov(grp_data.T, ddof=1))

    n_arr = np.array(n_list, dtype=float)
    N = np.sum(n_arr)  # total sample size

    # ------------------------------------------------------------------
    # Pooled (within-group) covariance matrix
    # ------------------------------------------------------------------
    S_pooled = np.zeros((p, p))
    for i in range(g):
        S_pooled += (n_arr[i] - 1) * S_list[i]
    S_pooled /= N - g

    # ------------------------------------------------------------------
    # Box's M statistic
    # M = (N - g) * ln|S_pooled| - Σ (n_i - 1) * ln|S_i|
    # slogdet is used instead of log(det(...)) to avoid silent underflow
    # to 0 or floating-point negatives that produce -inf / nan.
    # ------------------------------------------------------------------
    _, ln_det_pooled = np.linalg.slogdet(S_pooled)

    M = (N - g) * ln_det_pooled
    for i in range(g):
        _, ln_det_i = np.linalg.slogdet(S_list[i])
        M -= (n_arr[i] - 1) * ln_det_i

    # ------------------------------------------------------------------
    # Chi-square approximation via correction factor c₁
    # c₁ = [(2p² + 3p − 1) / (6(p+1)(g−1))] · [Σ 1/(n_i−1) − 1/(N−g)]
    # ------------------------------------------------------------------
    sum_inv = np.sum(1.0 / (n_arr - 1)) - 1.0 / (N - g)
    c1 = ((2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))) * sum_inv

    df_chi2 = p * (p + 1) * (g - 1) / 2.0
    chi2_approx = M * (1 - c1)
    p_value = 1.0 - stats.chi2.cdf(chi2_approx, df_chi2)

    return {
        "statistic": round(float(M), 6),
        "chi2_approx": round(float(chi2_approx), 6),
        "df": float(df_chi2),
        "p_value": round(float(p_value), 8),
    }
