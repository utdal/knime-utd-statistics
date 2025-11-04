"""
Post-hoc multiple comparison tests package.

This package provides ANOVA analysis with Tukey HSD and Holm-Bonferroni correction methods
for pairwise comparisons following significant ANOVA results.
"""

from .anova import run_one_way_anova, format_anova_results_for_knime, validate_anova_data
from .tukey_core import run_tukey_test, format_tukey_results_for_knime
from .bonferroni_core import run_bonferroni_test, format_bonferroni_results_for_knime
from .utils import (
    validate_group_data,
    PostHocTestType,
    test_type_param,
    data_column_param,
    group_column_param,
    alpha_param,
)

__all__ = [
    'run_one_way_anova',
    'format_anova_results_for_knime',
    'validate_anova_data',
    'run_tukey_test',
    'run_bonferroni_test',
    'format_tukey_results_for_knime',
    'format_bonferroni_results_for_knime',
    'validate_group_data',
    'PostHocTestType',
    'test_type_param',
    'data_column_param',
    'group_column_param',
    'alpha_param',
]