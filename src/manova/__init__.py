"""
MANOVA (Multivariate Analysis of Variance) package.

This package provides one-way MANOVA analysis with Pillai's Trace
and Box's M test for covariance-matrix equality.
"""

from .manova_core import run_manova, format_basic_results, format_advanced_results
from .box_m import compute_box_m
from .utils import (
    validate_manova_data,
    dependent_columns_param,
    group_column_param,
    alpha_param,
    advanced_stats_param,
    is_numeric,
    is_string,
    format_p_value,
)

__all__ = [
    # Core MANOVA
    "run_manova",
    "format_basic_results",
    "format_advanced_results",
    # Box's M
    "compute_box_m",
    # Validation
    "validate_manova_data",
    # Parameters
    "dependent_columns_param",
    "group_column_param",
    "alpha_param",
    "advanced_stats_param",
    # Helpers
    "is_numeric",
    "is_string",
    "format_p_value",
]
