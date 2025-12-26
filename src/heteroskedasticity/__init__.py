"""
Heteroskedasticity Testing Package for KNIME.

This package provides tools for detecting heteroskedasticity in OLS regression models.

Key Components:
---------------
- utils: Parameter definitions and helper functions
- regression_core: OLS model fitting and data preparation
- heteroskedasticity_tests: Statistical tests for heteroskedasticity

Exported Functions:
-------------------
- run_breusch_pagan_test: Standard heteroskedasticity test
- run_white_test: Robust general heteroskedasticity test
- run_goldfeld_quandt_test: Test based on variance comparison between groups

Exported Classes and Parameters:
---------------------------------
- TestType: Enumeration of available tests
- test_type_param, target_column_param, predictor_columns_param: KNIME parameters
- alpha_param, gq_sort_variable_param, gq_split_fraction_param: Additional parameters
"""

from .heteroskedasticity_tests import run_breusch_pagan_test, run_white_test, run_goldfeld_quandt_test

from .utils import (
    TestType,
    test_type_param,
    target_column_param,
    predictor_columns_param,
    alpha_param,
    gq_sort_variable_param,
    gq_split_fraction_param,
    is_numeric,
    detect_categorical_columns,
    format_p_value,
)

from .regression_core import prepare_data, fit_ols_model, generate_predictions_residuals, extract_model_summary

__all__ = [
    # Test functions
    "run_breusch_pagan_test",
    "run_white_test",
    "run_goldfeld_quandt_test",
    # Parameters
    "TestType",
    "test_type_param",
    "target_column_param",
    "predictor_columns_param",
    "alpha_param",
    "gq_sort_variable_param",
    "gq_split_fraction_param",
    # Utilities
    "is_numeric",
    "detect_categorical_columns",
    "format_p_value",
    # Regression functions
    "prepare_data",
    "fit_ols_model",
    "generate_predictions_residuals",
    "extract_model_summary",
]
