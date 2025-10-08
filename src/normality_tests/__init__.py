"""
Normality test computational cores.

This package contains pure Python implementations of various normality tests,
separated from KNIME node UI logic for better maintainability and testability.
"""

from .anderson_darling_core import run_ad_test
from .cramer_test_core import run_cramer_test

__all__ = ["run_ad_test", "run_cramer_test"]
