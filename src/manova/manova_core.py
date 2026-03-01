"""
One-way MANOVA implementation using statsmodels.

This module provides one-way MANOVA (Multivariate Analysis of Variance) with
Pillai's Trace as the primary test statistic, formatted for seamless
integration with the KNIME table schema.
"""

import warnings

import pandas as pd
from statsmodels.multivariate.manova import MANOVA


def run_manova(df, dep_vars, group_col, alpha=0.05):
    """
    Perform one-way MANOVA using statsmodels with Pillai's Trace.

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame (no missing values) with the dependent variables
        and the grouping column.
    dep_vars : list of str
        Names of the numeric dependent variable columns.
    group_col : str
        Name of the categorical grouping variable column.
    alpha : float, default 0.05
        Significance level for the test.

    Returns
    -------
    dict
        Dictionary containing:

        - ``factor`` : str – name of the grouping variable.
        - ``pillai_value`` : float – Pillai's Trace value (rounded to 6).
        - ``num_df`` : float – numerator degrees of freedom.
        - ``den_df`` : float – denominator degrees of freedom.
        - ``f_value`` : float – approximate F-value (rounded to 6).
        - ``p_value`` : float – p-value (rounded to 8).
        - ``significant`` : bool – whether the test is significant at *alpha*.
        - ``alpha`` : float – significance level used.
    """

    # Build formula:  y1 + y2 + … ~ C(group)
    dep_formula = " + ".join(dep_vars)
    formula = f"{dep_formula} ~ C({group_col})"

    # KNIME string columns arrive as pandas StringDtype ("string[python]").
    # patsy (used internally by statsmodels' formula API) calls
    # np.issubdtype() on the dtype, which does not recognise StringDtype
    # and raises "Cannot interpret 'string[python]' as a data type".
    # Casting to plain object dtype resolves this.
    df = df.copy()
    if hasattr(df[group_col], "dtype") and pd.api.types.is_string_dtype(df[group_col]):
        df[group_col] = df[group_col].astype(object)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maov = MANOVA.from_formula(formula, data=df)
        result = maov.mv_test()

    # Extract Pillai's Trace for the grouping factor
    factor_key = f"C({group_col})"
    stat_df = result.results[factor_key]["stat"]
    pillai_row = stat_df.loc["Pillai's trace"]

    pillai_value = round(float(pillai_row["Value"]), 6)
    num_df = float(pillai_row["Num DF"])
    den_df = float(pillai_row["Den DF"])
    f_value = round(float(pillai_row["F Value"]), 6)
    p_value = round(float(pillai_row["Pr > F"]), 8)
    is_significant = p_value <= alpha

    return {
        "factor": group_col,
        "pillai_value": pillai_value,
        "num_df": num_df,
        "den_df": den_df,
        "f_value": f_value,
        "p_value": p_value,
        "significant": is_significant,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def format_basic_results(manova_result):
    """
    Format MANOVA results for the **Basic** (Simple) output view.

    Columns: Factor, Pillai's P-Val, Conclusion.
    """

    conclusion = "Significant" if manova_result["significant"] else "Not Significant"

    return pd.DataFrame(
        {
            "Factor": [manova_result["factor"]],
            "Pillai's P-Val": [manova_result["p_value"]],
            "Conclusion": [conclusion],
        }
    )


def format_advanced_results(manova_result):
    """
    Format MANOVA results for the **Advanced** (Technical) output view.

    Columns: Source, Test Stat, Value, Numerator Df, Denominator Df,
    F-Value, P-Value.
    """

    return pd.DataFrame(
        {
            "Source": [manova_result["factor"]],
            "Test Stat": ["Pillai's Trace"],
            "Value": [manova_result["pillai_value"]],
            "Numerator Df": [manova_result["num_df"]],
            "Denominator Df": [manova_result["den_df"]],
            "F-Value": [manova_result["f_value"]],
            "P-Value": [manova_result["p_value"]],
        }
    )
