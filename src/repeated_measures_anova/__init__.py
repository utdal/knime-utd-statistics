"""
Repeated Measures ANOVA package.

Provides one-way repeated measures ANOVA via statsmodels with support for:
  - Long Format only
  - Greenhouse-Geisser sphericity correction (manual computation via scipy)
  - Basic output (executive summary) and Advanced output (technical validation)
"""

from .rm_anova_core import (
    run_rm_anova,
    validate_rm_anova_data,
    build_basic_output,
    build_advanced_output,
)

from .utils import (
    dv_column_param,
    within_factor_param,
    subject_id_param,
    alpha_param,
    advanced_output_param,
    is_numeric,
    is_nominal,
    is_subject_id,
    format_p_value,
)

__all__ = [
    # Core computation
    "run_rm_anova",
    "validate_rm_anova_data",
    "build_basic_output",
    "build_advanced_output",
    # Parameters
    "dv_column_param",
    "within_factor_param",
    "subject_id_param",
    "alpha_param",
    "advanced_output_param",
    # Utilities
    "is_numeric",
    "is_nominal",
    "is_subject_id",
    "format_p_value",
]
