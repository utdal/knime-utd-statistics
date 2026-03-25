"""
Utility functions and parameters for Repeated Measures ANOVA.
"""

import knime.extension as knext


# ── Column Type Filters ────────────────────────────────────────────────────────


def is_numeric(col: knext.Column) -> bool:
    """Filter for numeric columns. The dependent variable must be continuous."""
    return col.ktype in (knext.double(), knext.int32(), knext.int64())


def is_nominal(col: knext.Column) -> bool:
    """Filter for nominal/string columns. The Within-Subject Factor must be categorical."""
    return col.ktype == knext.string()


def is_subject_id(col: knext.Column) -> bool:
    """Filter for subject identifier columns. Accepts both string and integer IDs."""
    return col.ktype in (knext.string(), knext.int32(), knext.int64())


# ── Helper Functions ───────────────────────────────────────────────────────────


def format_p_value(p) -> float:
    """Format p-values by rounding to 4 decimal places, returning a double (float)."""
    import numpy as np

    if p is None:
        return np.nan
    try:
        val = float(p)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(val):
        return np.nan
    return round(val, 4)


# ── Parameters ────────────────────────────────────────────────────────────────

dv_column_param = knext.ColumnParameter(
    label="Dependent Variable",
    description=("The column containing the values you want to compare across conditions"),
    column_filter=is_numeric,
)

within_factor_param = knext.ColumnParameter(
    label="Within-Subject Factor",
    description=("The column that identifies which condition or time point each row belongs to."),
    column_filter=is_nominal,
)

subject_id_param = knext.ColumnParameter(
    label="Subject Identifier",
    description=("The column that uniquely identifies each participant. "),
    column_filter=is_subject_id,
)


# ── Shared Parameters ──────────────────────────────────────────────────────────

alpha_param = knext.DoubleParameter(
    label="Significance Level (\u03b1)",
    description=("The threshold used to decide whether a result is statistically significant. The default of 0.05."),
    default_value=0.05,
    min_value=0.001,
    max_value=0.999,
)

advanced_output_param = knext.BoolParameter(
    label="Compute advanced statistics",
    description=(
        "When checked, shows the full statistical breakdown, including "
        "all intermediate calculations and assumption check results."
    ),
    default_value=False,
    is_advanced=True,
)
