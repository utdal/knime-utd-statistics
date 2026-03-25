"""
Generic utilities shared by multiple nodes.
"""

import knime.extension as knext


# =============================================================================
# Helper Functions
# =============================================================================


def is_numeric(col: knext.Column) -> bool:
    """Filter for numeric columns (double, int32, int64)."""
    return col.ktype in (knext.double(), knext.int32(), knext.int64())


def is_categorical(col: knext.Column) -> bool:
    """Filter for categorical columns (string, boolean)."""
    return col.ktype in (knext.string(), knext.bool_())


def format_p_value(p):
    """Return a p-value as a full-precision float.

    Returns ``float('nan')`` for missing or unrecognised values so the
    P-Value column stays ``knext.double()`` and is usable by downstream
    KNIME nodes.
    """
    import pandas as pd

    if pd.isna(p):
        return float("nan")
    return float(p)


# =============================================================================
# Parameter Definitions
# =============================================================================

alpha_param = knext.DoubleParameter(
    label="Significance Level (α)",
    description="Factors with p-value ≤ α are considered significant (default: 0.05).",
    default_value=0.05,
    min_value=0.001,
    max_value=0.5,
)
