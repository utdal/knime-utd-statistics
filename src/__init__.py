import knime.extension as knext

# Shared category for all UTD statistical nodes
# MUST be defined before importing node modules that reference it
utd_category = knext.category(
    path="/community",
    level_id="utd_development",
    name="University of Texas at Dallas Development",
    description="Statistical analysis tools developed by the University of Texas at Dallas",
    icon="./icons/utd.png",
)

# Import nodes after category is defined to avoid circular import
from .normality_node import NormalityTestsNode
from .post_hoc_node import PostHocTestsNode

__all__ = ["NormalityTestsNode", "PostHocTestsNode", "utd_category"]
