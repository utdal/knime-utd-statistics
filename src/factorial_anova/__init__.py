"""
Factorial ANOVA Core Module.

Main computation function and utilities for factorial ANOVA analysis.
"""

from .factorial_anova_core import run_factorial_anova
from .utils import (
    # Enums
    AnovaType,
    # Parameters
    response_column_param,
    factor_columns_param,
    include_interactions_param,
    max_interaction_order_param,
    anova_type_param,
    alpha_param,
    advanced_output_param,
    # Helpers
    is_numeric,
    format_p_value,
)

__all__ = [
    "run_factorial_anova",
    "AnovaType",
    "response_column_param",
    "factor_columns_param",
    "include_interactions_param",
    "max_interaction_order_param",
    "anova_type_param",
    "alpha_param",
    "advanced_output_param",
    "is_numeric",
    "format_p_value",
]
