"""
Normality tests module.
"""

from .anderson_darling_core import run_ad_test
from .cramer_test_core import run_cramer_test

__all__ = ["run_ad_test", "run_cramer_test"]
