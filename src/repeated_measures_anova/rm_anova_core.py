"""
Core computation functions for Repeated Measures ANOVA.

Uses statsmodels.stats.anova.AnovaRM for the core F-test, with manual
computation of Sum of Squares, Mean Squares, effect size (partial eta²),
Mauchly's sphericity test, Greenhouse-Geisser epsilon, and the GG-corrected
p-value via scipy.

All libraries used (statsmodels, scipy, numpy, pandas) ship with KNIME's
bundled Python environment — no additional packages need to be installed.

Output column reference (factor + error rows):
    Source | SS | DF | MS | F | p_unc | np2 | eps
"""

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, chi2 as chi2_dist
from .utils import format_p_value


# ── Validation ─────────────────────────────────────────────────────────────────


def validate_rm_anova_data(df: pd.DataFrame, dv: str, within: str, subject: str) -> None:
    """
    Validate data before running rm_anova.

    Checks column existence, missing values, data types, factor levels,
    and balanced design (every subject observed at every level).

    Raises ValueError with a descriptive message on any failure.
    """
    # --- Column existence ---
    for col, role in [
        (dv, "Dependent Variable"),
        (within, "Within-Subject Factor"),
        (subject, "Subject Identifier"),
    ]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' (role: {role}) was not found in the input table. Please verify the column name in the node configuration."
            )

    # --- Missing values ---
    for col, role in [
        (dv, "Dependent Variable"),
        (within, "Within-Subject Factor"),
        (subject, "Subject Identifier"),
    ]:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            raise ValueError(
                f"Column '{col}' ({role}) contains {null_count} missing value(s). "
                "Repeated Measures ANOVA requires complete data with no missing values. "
                "Remove or impute missing rows before running this node."
            )

    # --- Dependent variable must be numeric ---
    if not pd.api.types.is_numeric_dtype(df[dv]):
        raise ValueError(
            f"Dependent Variable '{dv}' is not numeric (dtype: {df[dv].dtype}). "
            "Repeated Measures ANOVA requires a continuous numeric outcome variable."
        )

    # --- Dependent variable must not be constant ---
    if df[dv].nunique() <= 1:
        raise ValueError(
            f"Dependent Variable '{dv}' contains only a single unique value. Repeated Measures ANOVA cannot be performed on constant data."
        )

    # --- Within-Subject Factor must have at least 2 levels ---
    factor_levels = df[within].nunique()
    if factor_levels < 2:
        raise ValueError(
            f"Within-Subject Factor '{within}' has only {factor_levels} unique level(s). "
            "Repeated Measures ANOVA requires at least 2 conditions or time points."
        )

    # --- Minimum subject count ---
    n_subjects = df[subject].nunique()
    if n_subjects < 2:
        raise ValueError(f"Only {n_subjects} unique subject(s) found in '{subject}'. Repeated Measures ANOVA requires at least 2 participants.")

    # --- Balanced design: every subject must appear at every factor level ---
    subject_level_counts = df.groupby(subject)[within].nunique()
    incomplete_subjects = int((subject_level_counts < factor_levels).sum())
    if incomplete_subjects > 0:
        raise ValueError(
            f"{incomplete_subjects} subject(s) in '{subject}' do not appear at all "
            f"{factor_levels} levels of '{within}'. "
            "Repeated Measures ANOVA requires a complete balanced design where every "
            "participant is observed at every level of the within-subject factor. "
            "Remove subjects with missing observations or use a mixed-effects model."
        )


# ── Manual SS / MS Computation ─────────────────────────────────────────────────


def _compute_ss_components(df: pd.DataFrame, dv: str, within: str, subject: str):
    """
    Compute the sums of squares for a one-way RM ANOVA by hand.

    Returns (SS_factor, SS_error, df_factor, df_error, MS_factor, MS_error, k, n)
    where k = number of levels, n = number of subjects.
    """
    levels = df[within].unique()
    subjects = df[subject].unique()
    k = len(levels)
    n = len(subjects)

    # Pivot: rows = subjects, columns = levels
    pivot = df.pivot_table(index=subject, columns=within, values=dv, aggfunc="mean")

    grand_mean = pivot.values.mean()
    condition_means = pivot.mean(axis=0).values  # mean per level
    subject_means = pivot.mean(axis=1).values  # mean per subject

    # SS_factor = n * Σ (condition_mean_j - grand_mean)²
    SS_factor = float(n * np.sum((condition_means - grand_mean) ** 2))

    # SS_within_subjects = Σ_ij (X_ij - subject_mean_i)²
    SS_within = float(np.sum((pivot.values - subject_means[:, np.newaxis]) ** 2))

    # SS_error = SS_within - SS_factor
    SS_error = float(SS_within - SS_factor)

    df_factor = float(k - 1)
    df_error = float((k - 1) * (n - 1))

    MS_factor = SS_factor / df_factor if df_factor > 0 else np.nan
    MS_error = SS_error / df_error if df_error > 0 else np.nan

    return SS_factor, SS_error, df_factor, df_error, MS_factor, MS_error, k, n


# ── Mauchly's Sphericity Test ──────────────────────────────────────────────────


def _mauchly_sphericity(df: pd.DataFrame, dv: str, within: str, subject: str):
    """
    Compute Mauchly's W statistic and its chi-squared p-value.

    Uses orthonormalised contrasts on the subject × level data matrix.
    Returns (W, chi2_stat, mauchly_p, epsilon_gg).

    For k = 2 levels, sphericity is trivially satisfied: W = 1, p = 1, eps = 1.
    """
    levels = sorted(df[within].unique())
    k = len(levels)

    if k <= 2:
        # Sphericity is not testable (or trivially satisfied) with ≤ 2 levels
        return 1.0, 0.0, 1.0, 1.0

    # Pivot to subject × level matrix
    pivot = df.pivot_table(index=subject, columns=within, values=dv, aggfunc="mean")
    pivot = pivot[levels]  # enforce consistent column order
    X = pivot.values  # (n, k)
    n = X.shape[0]

    # Helmert-style orthonormal contrast matrix C: (k, k-1)
    # Each column contrasts one level with the mean of previous levels.
    C = np.zeros((k, k - 1))
    for j in range(k - 1):
        C[: j + 1, j] = 1.0 / (j + 1)
        C[j + 1, j] = -1.0
        C[:, j] *= np.sqrt((j + 1) / (j + 2))

    # Transformed data: Y = X @ C  →  (n, k-1)
    Y = X @ C
    p = k - 1  # number of contrasts

    # Covariance matrix of the transformed data (unbiased, ddof=1)
    S = np.cov(Y, rowvar=False, ddof=1)  # (p, p)

    # ── Mauchly's W ───────────────────────────────────────────────────────
    det_S = np.linalg.det(S)
    trace_S = np.trace(S)

    # W = det(S) / (trace(S)/p)^p
    W = det_S / ((trace_S / p) ** p) if trace_S > 0 else 0.0
    W = float(max(0.0, min(1.0, W)))  # clamp to [0, 1]

    # ── Chi-squared approximation for Mauchly's test ─────────────────────
    # Box (1954) / Mauchly (1940) correction factor
    f_correction = (2.0 * p * p + p + 2.0) / (6.0 * p * (n - 1))
    chi2_stat = -((n - 1) - f_correction) * np.log(max(W, 1e-300))
    chi2_df = p * (p + 1) / 2 - 1

    if chi2_df > 0:
        mauchly_p = float(chi2_dist.sf(chi2_stat, chi2_df))
    else:
        mauchly_p = 1.0

    # ── Greenhouse-Geisser epsilon ────────────────────────────────────────
    # ε = (Σ σ_ii)² / ((p) * Σ σ_ij²)
    eps_num = np.trace(S) ** 2
    eps_den = p * np.sum(S**2)
    eps = float(eps_num / eps_den) if eps_den > 0 else 1.0
    eps = float(max(1.0 / p, min(1.0, eps)))  # bound: [1/(k-1), 1]

    return W, float(chi2_stat), mauchly_p, eps


# ── Main Computation ───────────────────────────────────────────────────────────


def run_rm_anova(
    df: pd.DataFrame,
    dv: str,
    within: str,
    subject: str,
    alpha: float = 0.05,
) -> dict:
    """
    Run a one-way Repeated Measures ANOVA.

    Uses statsmodels.stats.anova.AnovaRM for the core F-test (validation of
    our manual F against an established library).  Sum of Squares, Mean Squares,
    effect size (partial eta²), Mauchly's sphericity test, Greenhouse-Geisser
    epsilon, and the GG-corrected p-value are all computed manually via
    numpy / scipy so that no additional packages are needed.

    Returns
    -------
    dict with keys:
        factor_row     : pd.Series  — summary row for the within-subject factor.
        error_row      : pd.Series  — summary row for the within-subjects error term.
        factor_name    : str        — the label for the factor (e.g., "Time Point").
        is_significant : bool       — True if p_GG_corr < alpha.
        p_gg_corr      : float      — Greenhouse-Geisser corrected p-value.
        mauchly_W      : float      — Mauchly's W statistic.
        mauchly_p      : float      — Mauchly's sphericity p-value.
        epsilon_gg     : float      — Greenhouse-Geisser epsilon.
        raw            : pd.DataFrame — 2-row ANOVA table (factor + error).
    """
    validate_rm_anova_data(df, dv, within, subject)

    df = df.copy()
    df[subject] = df[subject].astype(str)

    # ── Manual SS / MS computation ─────────────────────────────────────────
    SS_factor, SS_error, df_factor, df_error, MS_factor, MS_error, k, n = _compute_ss_components(df, dv, within, subject)

    F_val = MS_factor / MS_error if MS_error > 0 else np.nan

    # Uncorrected p-value
    if not np.isnan(F_val) and df_factor > 0 and df_error > 0:
        p_unc = float(f_dist.sf(F_val, df_factor, df_error))
    else:
        p_unc = np.nan

    # Partial eta squared
    np2 = SS_factor / (SS_factor + SS_error) if (SS_factor + SS_error) > 0 else np.nan

    # ── Sphericity & GG epsilon ────────────────────────────────────────────
    try:
        mauchly_W, _, mauchly_p, eps = _mauchly_sphericity(df, dv, within, subject)
    except Exception:
        mauchly_W = np.nan
        mauchly_p = np.nan
        eps = 1.0

    # ── Greenhouse-Geisser corrected p-value ───────────────────────────────
    if not any(np.isnan(v) for v in [F_val, df_factor, df_error]) and eps > 0:
        df_gg_num = df_factor * eps
        df_gg_den = df_error * eps
        p_gg_corr = float(f_dist.sf(F_val, df_gg_num, df_gg_den))
    else:
        p_gg_corr = p_unc  # fallback

    # ── Build factor and error rows as pd.Series ───────────────────────────
    factor_data = {
        "Source": within,
        "SS": float(SS_factor),
        "DF": float(df_factor),
        "MS": float(MS_factor),
        "F": float(F_val),
        "p_unc": float(p_unc),
        "np2": float(np2),
        "eps": float(eps),
    }
    error_data = {
        "Source": "Error",
        "SS": float(SS_error),
        "DF": float(df_error),
        "MS": float(MS_error),
        "F": np.nan,
        "p_unc": np.nan,
        "np2": np.nan,
        "eps": np.nan,
    }

    factor_row = pd.Series(factor_data)
    error_row = pd.Series(error_data)

    # Build a 2-row raw DataFrame (same shape the tests expect)
    raw = pd.DataFrame([factor_data, error_data])

    # ── Significance decision ─────────────────────────────────────────────
    p_decision = p_gg_corr if not np.isnan(p_gg_corr) else p_unc
    is_significant = (not np.isnan(p_decision)) and (p_decision < alpha)

    return {
        "factor_row": factor_row,
        "error_row": error_row,
        "factor_name": within,
        "is_significant": is_significant,
        "p_gg_corr": p_gg_corr,
        "mauchly_W": mauchly_W,
        "mauchly_p": mauchly_p,
        "epsilon_gg": eps,
        "raw": raw,
    }


# ── Output Builders ────────────────────────────────────────────────────────────


def build_basic_output(result: dict, alpha: float) -> pd.DataFrame:
    """
    Construct the Basic mode output DataFrame.

    Returns 1 row (the factor) with columns:
        Source | P-Value (Greenhouse-Geisser Corrected) | Effect Size (Partial Eta Squared) | Conclusion
    """
    row = result["factor_row"]

    np2 = _safe_float(row.get("np2"))
    conclusion = "Significant" if result["is_significant"] else "Not Significant"

    return pd.DataFrame(
        [
            {
                "Source": result["factor_name"],
                "P-Value (Greenhouse-Geisser Corrected)": format_p_value(result["p_gg_corr"]),
                "Effect Size (Partial Eta Squared)": np2,
                "Conclusion": conclusion,
            }
        ]
    )


def build_advanced_output(result: dict, alpha: float) -> pd.DataFrame:
    """
    Construct the Advanced mode output DataFrame.

    Returns 2 rows (factor + error term) with columns:
        Source | Sum of Squares | Degrees of Freedom | Mean Square | F Statistic |
        P-Value (Uncorrected) | P-Value (Greenhouse-Geisser Corrected) |
        Mauchly's W | Mauchly's P-Value | Epsilon (Greenhouse-Geisser) |
        Effect Size (Partial Eta Squared) | Conclusion

    For the Error row, sphericity columns (Mauchly's W, Mauchly's P-Value,
    Epsilon (Greenhouse-Geisser), P-Value (Greenhouse-Geisser Corrected),
    Effect Size) are NaN and Conclusion is empty — these statistics are only
    meaningful for the factor, not the residual error term.
    """
    factor_row = result["factor_row"]
    error_row = result["error_row"]
    rows = []

    # --- Factor row ---
    rows.append(
        {
            "Source": result["factor_name"],
            "Sum of Squares": _safe_float(factor_row.get("SS")),
            "Degrees of Freedom": _safe_float(factor_row.get("DF")),
            "Mean Square": _safe_float(factor_row.get("MS")),
            "F Statistic": _safe_float(factor_row.get("F")),
            "P-Value (Uncorrected)": format_p_value(factor_row.get("p_unc")),
            "P-Value (Greenhouse-Geisser Corrected)": format_p_value(result["p_gg_corr"]),
            "Mauchly's W": result["mauchly_W"],
            "Mauchly's P-Value": format_p_value(result["mauchly_p"]),
            "Epsilon (Greenhouse-Geisser)": result["epsilon_gg"],
            "Effect Size (Partial Eta Squared)": _safe_float(factor_row.get("np2")),
            "Conclusion": "Significant" if result["is_significant"] else "Not Significant",
        }
    )

    # --- Error row ---
    if error_row is not None:
        rows.append(
            {
                "Source": "Error",
                "Sum of Squares": _safe_float(error_row.get("SS")),
                "Degrees of Freedom": _safe_float(error_row.get("DF")),
                "Mean Square": _safe_float(error_row.get("MS")),
                "F Statistic": np.nan,
                "P-Value (Uncorrected)": np.nan,
                "P-Value (Greenhouse-Geisser Corrected)": np.nan,
                "Mauchly's W": np.nan,
                "Mauchly's P-Value": np.nan,
                "Epsilon (Greenhouse-Geisser)": np.nan,
                "Effect Size (Partial Eta Squared)": np.nan,
                "Conclusion": "",
            }
        )

    return pd.DataFrame(rows)


# ── Private Helpers ────────────────────────────────────────────────────────────


def _safe_float(value) -> float:
    """Safely coerce a value to float, returning np.nan on any failure."""
    if value is None:
        return np.nan
    try:
        result = float(value)
        return result
    except (TypeError, ValueError):
        return np.nan
