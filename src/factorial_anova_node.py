"""
Factorial ANOVA Node for KNIME.

This module provides a KNIME node for testing how multiple categorical factors
affect a continuous outcome variable. Supports interaction effects, multiple
sum of squares methods, and flexible output formats.
"""

import knime.extension as knext
import numpy as np
import pandas as pd

from .factorial_anova import (
    # Core function
    run_factorial_anova,
    # Parameters
    response_column_param,
    factor_columns_param,
    include_interactions_param,
    max_interaction_order_param,
    anova_type_param,
    alpha_param,
    advanced_output_param,
)


# =============================================================================
# Category Definition
# =============================================================================

factorial_anova_category = knext.category(
    path="/community",
    level_id="utd_development",
    name="University of Texas at Dallas Development",
    description="Statistical analysis tools developed by the University of Texas at Dallas",
    icon="./icons/utd.png",
)


# =============================================================================
# Node Definition
# =============================================================================


@knext.node(
    name="Factorial ANOVA",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="./icons/factorial.jpg",
    category=factorial_anova_category,
)
@knext.input_table(
    name="Input Data",
    description="Table with numeric response variable and categorical factor columns.",
)
@knext.output_table(
    name="ANOVA Results",
    description="Statistical test results showing which factors significantly affect the outcome.",
)
@knext.output_table(
    name="Model Coefficients",
    description="Detailed coefficient estimates showing the effect size of each factor level.",
)
class FactorialAnovaNode:
    """Tests whether multiple categorical factors (and their interactions) have a significant effect on a continuous outcome variable.

    This node performs factorial (N-way) ANOVA to analyze how different grouping factors affect a numeric outcome. It can detect both individual factor effects and interaction effects (when the impact of one factor depends on another factor).
    """

    # --- Core Parameters ---
    response_column = response_column_param
    factor_columns = factor_columns_param
    include_interactions = include_interactions_param

    # Conditional parameter: only show when interactions are enabled
    max_interaction_order = max_interaction_order_param.rule(
        knext.OneOf(include_interactions, [True]),
        knext.Effect.SHOW,
    )

    anova_type = anova_type_param
    alpha = alpha_param

    # --- Output Format ---
    advanced_output = advanced_output_param

    def configure(self, cfg_ctx, input_spec):
        """Configure the node's two output table schemas."""
        if self.advanced_output:
            # Advanced ANOVA Table - variance decomposition and effect sizes only
            anova_schema = knext.Schema.from_columns(
                [
                    knext.Column(knext.string(), "Source"),
                    knext.Column(knext.double(), "Sum Sq"),
                    knext.Column(knext.double(), "Mean Sq"),
                    knext.Column(knext.int64(), "DF"),
                    knext.Column(knext.double(), "F-Statistic"),
                    knext.Column(knext.string(), "P-Value"),
                    knext.Column(knext.double(), "Partial Eta Squared"),
                    knext.Column(knext.string(), "Conclusion"),
                ]
            )
        else:
            # Basic ANOVA Summary
            anova_schema = knext.Schema.from_columns(
                [
                    knext.Column(knext.string(), "Factor"),
                    knext.Column(knext.double(), "F-Statistic"),
                    knext.Column(knext.string(), "P-Value"),
                    knext.Column(knext.string(), "Conclusion"),
                ]
            )

        # Port 2: Model Coefficients (always the same schema)
        coef_schema = knext.Schema.from_columns(
            [
                knext.Column(knext.string(), "Term"),
                knext.Column(knext.double(), "Coefficient"),
                knext.Column(knext.double(), "Std Error"),
                knext.Column(knext.string(), "P-Value"),
                knext.Column(knext.double(), "CI Lower"),
                knext.Column(knext.double(), "CI Upper"),
            ]
        )

        return anova_schema, coef_schema

    def execute(self, exec_ctx, input_table):
        """Execute the factorial ANOVA analysis."""
        # Convert input to pandas
        df = input_table.to_pandas()

        # Validate factor columns selection
        if not self.factor_columns or len(self.factor_columns) == 0:
            raise ValueError("No factor columns selected. Please select at least one categorical factor variable in the node configuration.")

        # Validate response column selection
        if self.response_column is None:
            raise ValueError("No response column selected. Please select a numeric response variable in the node configuration.")

        # Check for overlap between response and factors
        if self.response_column in self.factor_columns:
            raise ValueError(
                f"Response column '{self.response_column}' cannot also be a factor variable. "
                "Please select different columns for response and factors."
            )

        # Run factorial ANOVA
        result = run_factorial_anova(
            df=df,
            response_col=self.response_column,
            factor_cols=list(self.factor_columns),
            include_interactions=self.include_interactions,
            max_interaction_order=self.max_interaction_order if self.include_interactions else 1,
            anova_type=self.anova_type,
            alpha=self.alpha,
        )

        # Propagate warnings to KNIME console
        if "warnings" in result and result["warnings"]:
            for warning_msg in result["warnings"]:
                exec_ctx.set_warning(warning_msg)

        # Format output based on advanced_output setting
        if self.advanced_output:
            anova_df = self._format_advanced_anova_table(result["advanced_table"])
        else:
            anova_df = result["basic_table"]

        # Prepare output tables
        anova_table = knext.Table.from_pandas(anova_df)

        coef_table = knext.Table.from_pandas(result["coefficient_table"])

        return anova_table, coef_table

    def _format_advanced_anova_table(self, advanced_df: pd.DataFrame) -> pd.DataFrame:
        """Format advanced ANOVA table adding Partial_Eta_Squared effect size."""
        # Get residual SS for partial eta-squared denominator
        residual_rows = advanced_df[advanced_df["Source"] == "Residual"]["Sum Sq"].values
        ss_residual = residual_rows[0] if len(residual_rows) > 0 else np.nan

        rows = []
        for _, row in advanced_df.iterrows():
            source = row["Source"]
            sum_sq = row["Sum Sq"]

            # Partial eta-squared: SS_effect / (SS_effect + SS_residual)
            if source != "Residual" and not np.isnan(ss_residual):
                partial_eta_sq = sum_sq / (sum_sq + ss_residual)
            else:
                partial_eta_sq = np.nan

            rows.append(
                {
                    "Source": source,
                    "Sum Sq": sum_sq,
                    "Mean Sq": row["Mean Sq"],
                    "DF": row["DF"],
                    "F-Statistic": row["F-Statistic"],
                    "P-Value": row["P-Value"],
                    "Partial Eta Squared": partial_eta_sq,
                    "Conclusion": row["Conclusion"],
                }
            )

        return pd.DataFrame(rows)
