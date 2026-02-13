"""
Factorial ANOVA Core Computation Module.

Contains the main computation logic for factorial (N-way) ANOVA analysis
using statsmodels OLS and formula-based model specification.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations
from typing import List, Dict, Any, Optional

from .utils import format_p_value


# =============================================================================
# Data Preparation
# =============================================================================


def prepare_anova_data(
    df: pd.DataFrame,
    response_col: str,
    factor_cols: List[str],
) -> tuple:
    """
    Prepare and validate data for factorial ANOVA.

    Always drops rows with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    response_col : str
        Name of the response (dependent) variable column
    factor_cols : List[str]
        Names of factor (independent) variable columns

    Returns
    -------
    tuple
        (cleaned_df, warnings_list) - Cleaned dataframe and any warnings generated

    Raises
    ------
    ValueError
        If columns not found, insufficient data, or invalid factor levels
    """
    warnings = []

    # Validate response column exists
    if response_col not in df.columns:
        raise ValueError(f"Response column '{response_col}' not found in input data.")

    # Validate factor columns exist
    missing_factors = [col for col in factor_cols if col not in df.columns]
    if missing_factors:
        raise ValueError(f"Factor columns not found: {missing_factors}")

    # Select relevant columns
    relevant_columns = [response_col] + list(factor_cols)
    work_df = df[relevant_columns].copy()

    # Always drop rows with missing values
    missing_count = work_df.isnull().sum().sum()
    if missing_count > 0:
        original_len = len(work_df)
        work_df = work_df.dropna()
        dropped_count = original_len - len(work_df)
        if dropped_count > 0:
            warnings.append(f"Dropped {dropped_count} rows containing missing values ({missing_count} total missing cells).")

    # Check minimum data requirement
    if len(work_df) < 3:
        raise ValueError(f"Insufficient data for ANOVA. Only {len(work_df)} complete rows available. At minimum, 3 observations are required.")

    # Validate response is numeric
    if not pd.api.types.is_numeric_dtype(work_df[response_col]):
        raise ValueError(f"Response column '{response_col}' must be numeric. Found type: {work_df[response_col].dtype}")

    # Convert factor columns to categorical and validate
    for col in factor_cols:
        n_unique = work_df[col].nunique()

        # Validate factor has at least 2 levels
        if n_unique < 2:
            raise ValueError(f"Factor '{col}' has only {n_unique} level(s). Each factor must have at least 2 levels for ANOVA.")

        work_df[col] = pd.Categorical(work_df[col])

    return work_df, warnings


# =============================================================================
# Formula Construction
# =============================================================================


def build_anova_formula(
    response_col: str,
    factor_cols: List[str],
    include_interactions: bool,
    max_order: int,
) -> str:
    """
    Build patsy-style formula for factorial ANOVA.

    Parameters
    ----------
    response_col : str
        Name of the response variable
    factor_cols : List[str]
        Names of factor variables
    include_interactions : bool
        Whether to include interaction terms
    max_order : int
        Maximum interaction order (2 = 2-way, 3 = 3-way, etc.)

    Returns
    -------
    str
        Formula string for statsmodels OLS (e.g., "Y ~ C(A) + C(B) + C(A):C(B)")

    Examples
    --------
    >>> build_anova_formula("Y", ["A", "B"], False, 2)
    'Y ~ C(A) + C(B)'

    >>> build_anova_formula("Y", ["A", "B"], True, 2)
    'Y ~ C(A) * C(B)'

    >>> build_anova_formula("Y", ["A", "B", "C"], True, 2)
    'Y ~ C(A) + C(B) + C(C) + C(A):C(B) + C(A):C(C) + C(B):C(C)'
    """
    if not include_interactions:
        # Main effects only
        terms = [f"C({col})" for col in factor_cols]
        return f"{response_col} ~ {' + '.join(terms)}"

    if max_order >= len(factor_cols):
        # Full factorial (all interactions up to n-way)
        terms = [f"C({col})" for col in factor_cols]
        return f"{response_col} ~ {' * '.join(terms)}"

    # Custom interaction depth (e.g., max_order=2 for 2-way only)
    terms = [f"C({col})" for col in factor_cols]  # Main effects

    for order in range(2, max_order + 1):
        for combo in combinations(factor_cols, order):
            interaction_term = ":".join([f"C({c})" for c in combo])
            terms.append(interaction_term)

    return f"{response_col} ~ {' + '.join(terms)}"


# =============================================================================
# Design Balance Check
# =============================================================================


def check_design_balance(df: pd.DataFrame, factor_cols: List[str]) -> bool:
    """
    Check if the experimental design is balanced (equal cell counts).

    Parameters
    ----------
    df : pd.DataFrame
        Data with factor columns
    factor_cols : List[str]
        Names of factor columns

    Returns
    -------
    bool
        True if design is balanced (all cells have equal counts), False otherwise
    """
    counts = df.groupby(list(factor_cols)).size()
    return counts.nunique() == 1


# =============================================================================
# Output Formatting
# =============================================================================


def format_basic_anova_table(anova_table: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Format ANOVA table for basic output port.

    Parameters
    ----------
    anova_table : pd.DataFrame
        Raw ANOVA table from statsmodels
    alpha : float
        Significance level for conclusions

    Returns
    -------
    pd.DataFrame
        Formatted table with columns: Factor, F-Statistic, P-Value, Conclusion
    """
    rows = []
    for idx, row in anova_table.iterrows():
        # Determine conclusion
        if idx == "Residual":
            conclusion = "Unexplained"
        elif pd.notna(row["PR(>F)"]) and row["PR(>F)"] <= alpha:
            conclusion = "Significant"
        elif pd.notna(row["PR(>F)"]):
            conclusion = "Not Significant"
        else:
            conclusion = "Unexplained"

        rows.append(
            {
                "Factor": str(idx),
                "F-Statistic": row["F"] if pd.notna(row["F"]) else np.nan,
                "P-Value": format_p_value(row["PR(>F)"]),
                "Conclusion": conclusion,
            }
        )

    return pd.DataFrame(rows)


def format_advanced_anova_table(anova_table: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Format ANOVA table for advanced output port with full statistics.

    Parameters
    ----------
    anova_table : pd.DataFrame
        Raw ANOVA table from statsmodels
    alpha : float
        Significance level for conclusions

    Returns
    -------
    pd.DataFrame
        Formatted table with columns: Source, Sum_Sq, DF, Mean_Sq, F-Statistic, P-Value, Conclusion
    """
    rows = []
    for idx, row in anova_table.iterrows():
        # Calculate mean square
        mean_sq = row["sum_sq"] / row["df"] if row["df"] > 0 else np.nan

        # Determine conclusion
        if idx == "Residual":
            conclusion = "Unexplained"
        elif pd.notna(row["PR(>F)"]) and row["PR(>F)"] <= alpha:
            conclusion = "Significant"
        elif pd.notna(row["PR(>F)"]):
            conclusion = "Not Significant"
        else:
            conclusion = "Unexplained"

        rows.append(
            {
                "Source": str(idx),
                "Sum_Sq": row["sum_sq"],
                "DF": int(row["df"]) if pd.notna(row["df"]) else 0,
                "Mean_Sq": mean_sq,
                "F-Statistic": row["F"] if pd.notna(row["F"]) else np.nan,
                "P-Value": format_p_value(row["PR(>F)"]),
                "Conclusion": conclusion,
            }
        )

    return pd.DataFrame(rows)


def format_coefficient_table(model) -> pd.DataFrame:
    """
    Format model coefficients for output.

    Parameters
    ----------
    model : statsmodels RegressionResultsWrapper
        Fitted OLS model

    Returns
    -------
    pd.DataFrame
        Formatted table with coefficient details
    """
    conf_int = model.conf_int()

    return pd.DataFrame(
        {
            "Term": model.params.index.tolist(),
            "Coefficient": model.params.values,
            "Std_Error": model.bse.values,
            "P-Value": [format_p_value(p) for p in model.pvalues.values],
            "CI_Lower": conf_int[0].values,
            "CI_Upper": conf_int[1].values,
        }
    )


def format_residual_table(
    model,
    response_col: str,
    y_values: pd.Series,
) -> pd.DataFrame:
    """
    Format predictions and residuals for output.

    Parameters
    ----------
    model : statsmodels RegressionResultsWrapper
        Fitted OLS model
    response_col : str
        Name of the response column (for labeling)
    y_values : pd.Series
        Original response values

    Returns
    -------
    pd.DataFrame
        Table with response, predicted, and residual values
    """
    return pd.DataFrame(
        {
            response_col: y_values.values,
            "Predicted": model.fittedvalues.values,
            "Residual": model.resid.values,
        }
    )


# =============================================================================
# Main ANOVA Function
# =============================================================================


def run_factorial_anova(
    df: pd.DataFrame,
    response_col: str,
    factor_cols: List[str],
    include_interactions: bool,
    max_interaction_order: int,
    anova_type: str,
    alpha: float,
) -> Dict[str, Any]:
    """
    Perform factorial (N-way) ANOVA using statsmodels OLS.

    Missing values are automatically dropped. Coefficients and residuals
    are always included in the output.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    response_col : str
        Name of the response (dependent) variable column
    factor_cols : List[str]
        Names of factor (independent) variable columns
    include_interactions : bool
        Whether to include interaction terms
    max_interaction_order : int
        Maximum interaction order (2, 3, or 4)
    anova_type : str
        ANOVA type ("TYPE_I", "TYPE_II", or "TYPE_III")
    alpha : float
        Significance level for statistical decisions

    Returns
    -------
    dict
        Dictionary containing:
        - "basic_table": pd.DataFrame - Basic ANOVA summary
        - "advanced_table": pd.DataFrame - Detailed ANOVA table
        - "coefficient_table": pd.DataFrame - Model coefficients (always included)
        - "residual_table": pd.DataFrame - Predictions and residuals (always included)
        - "n_obs": int - Number of observations used
        - "formula": str - Model formula used
        - "warnings": List[str] - Any warnings generated
        - "is_balanced": bool - Whether design is balanced

    Raises
    ------
    ValueError
        If data validation fails or model cannot be fit
    """
    # Step 1: Prepare and validate data (always drops missing values)
    work_df, data_warnings = prepare_anova_data(df, response_col, factor_cols)
    warnings = data_warnings.copy()

    n_obs = len(work_df)

    # Step 2: Check design balance and generate warnings
    is_balanced = check_design_balance(work_df, factor_cols)

    if not is_balanced and anova_type == "TYPE_II" and include_interactions:
        warnings.append(
            "⚠️ Unbalanced design detected with Type II SS and interactions enabled. "
            "Consider using Type III SS for accurate interaction effects in unbalanced data."
        )

    # Step 3: Build formula
    formula = build_anova_formula(response_col, factor_cols, include_interactions, max_interaction_order)

    # Step 4: Fit OLS model
    try:
        model = ols(formula, data=work_df).fit()
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Failed to fit ANOVA model. This often happens when:\n"
            "1. Factors are perfectly correlated\n"
            "2. You have more factor levels than observations\n"
            "3. A factor has constant/identical values across groups\n"
            f"Technical error: {str(e)}"
        )
    except Exception as e:
        raise ValueError(f"Unexpected error fitting ANOVA model: {str(e)}")

    # Step 5: Generate ANOVA table
    type_mapping = {"TYPE_I": 1, "TYPE_II": 2, "TYPE_III": 3}
    anova_type_num = type_mapping.get(anova_type, 2)

    anova_table = sm.stats.anova_lm(model, typ=anova_type_num)

    # Step 6: Format output tables
    basic_table = format_basic_anova_table(anova_table, alpha)
    advanced_table = format_advanced_anova_table(anova_table, alpha)

    # Step 7: Always generate coefficient table
    coefficient_table = format_coefficient_table(model)

    # Step 8: Always generate residual table
    y_values = work_df[response_col]
    residual_table = format_residual_table(model, response_col, y_values)

    # Step 9: Check for small sample size warning
    n_params = len(model.params)
    if n_obs < n_params + 20:
        warnings.append(
            f"⚠️ Small sample size: n={n_obs} observations with {n_params} parameters. Recommended: n > {n_params + 20}. Results may be unreliable."
        )

    # Step 10: High-order interaction warning
    if include_interactions and max_interaction_order >= 3 and len(factor_cols) >= 3:
        warnings.append("⚠️ High-order interactions (3-way+) can produce large, difficult-to-interpret models.")

    return {
        "basic_table": basic_table,
        "advanced_table": advanced_table,
        "coefficient_table": coefficient_table,
        "residual_table": residual_table,
        "n_obs": n_obs,
        "formula": formula,
        "warnings": warnings,
        "is_balanced": is_balanced,
    }
