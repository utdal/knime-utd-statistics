"""
KNIME Extension entry point.

This module imports and registers the unified normality testing node.
"""

from .normality_node import NormalityTestsNode

__all__ = ["NormalityTestsNode"]
