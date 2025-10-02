"""
KNIME Extension entry point.

This module imports and registers the unified normality testing node.
"""

from .NormalityNode import NormalityTestsNode

__all__ = ["NormalityTestsNode"]
