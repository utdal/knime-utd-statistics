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
            # Advanced ANOVA Table with coefficient info
            anova_schema = knext.Schema.from_columns([
                knext.Column(knext.string(), "Source"),
                knext.Column(knext.double(), "Sum_Sq"),
                knext.Column(knext.int64(), "DF"),
                knext.Column(knext.double(), "F"),
                knext.Column(knext.string(), "PR(>F)"),
                knext.Column(knext.double(), "Coefficient"),
                knext.Column(knext.double(), "Std_Error"),
                knext.Column(knext.double(), "T-Value"),
                knext.Column(knext.string(), "P-Value"),
                knext.Column(knext.string(), "Conclusion"),
            ])
        else:
            # Basic ANOVA Summary
            anova_schema = knext.Schema.from_columns([
                knext.Column(knext.string(), "Factor"),
                knext.Column(knext.double(), "F-Statistic"),
                knext.Column(knext.string(), "P-Value"),
                knext.Column(knext.string(), "Conclusion"),
            ])
        
        # Port 1: Model Coefficients (always the same schema)
        coef_schema = knext.Schema.from_columns([
            knext.Column(knext.string(), "Term"),
            knext.Column(knext.double(), "Coefficient"),
            knext.Column(knext.double(), "Std_Error"),
            knext.Column(knext.string(), "P-Value"),
            knext.Column(knext.double(), "CI_Lower"),
            knext.Column(knext.double(), "CI_Upper"),
        ])
        
        return anova_schema, coef_schema

    def execute(self, exec_ctx, input_table):
        """Execute the factorial ANOVA analysis."""
        # Convert input to pandas
        df = input_table.to_pandas()
        
        # Validate factor columns selection
        if not self.factor_columns or len(self.factor_columns) == 0:
            raise ValueError(
                "No factor columns selected. Please select at least one categorical "
                "factor variable in the node configuration."
            )
        
        # Validate response column selection
        if self.response_column is None:
            raise ValueError(
                "No response column selected. Please select a numeric response "
                "variable in the node configuration."
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
            anova_df = self._format_advanced_with_coefficients(
                result["advanced_table"],
                result["coefficient_table"],
                self.alpha,
            )
        else:
            anova_df = result["basic_table"]
        
        # Prepare output tables
        anova_table = knext.Table.from_pandas(anova_df)
        coef_table = knext.Table.from_pandas(result["coefficient_table"])
        
        return anova_table, coef_table

    def _format_advanced_with_coefficients(
        self,
        advanced_df: pd.DataFrame,
        coef_df: pd.DataFrame,
        alpha: float,
    ) -> pd.DataFrame:
        """Combine ANOVA statistics with coefficient details for advanced output."""
        # Build coefficient lookup (excluding Intercept for matching)
        coef_lookup = {}
        for _, row in coef_df.iterrows():
            term = row["Term"]
            # Clean term name for matching (remove C() wrapper if present)
            clean_term = term.replace("C(", "").replace(")[T.", ":").replace("]", "")
            coef_lookup[clean_term] = row
            coef_lookup[term] = row
        
        rows = []
        for _, row in advanced_df.iterrows():
            source = row["Source"]
            
            # Try to find matching coefficient
            coef_row = None
            # Try exact match first
            if source in coef_lookup:
                coef_row = coef_lookup[source]
            else:
                # Try partial match for interaction terms
                for key in coef_lookup:
                    if source in key or key in source:
                        coef_row = coef_lookup[key]
                        break
            
            # Extract coefficient values if found
            if coef_row is not None and source != "Residual":
                coefficient = coef_row["Coefficient"]
                std_error = coef_row["Std_Error"]
                # Calculate T-value from coefficient / std_error
                t_value = coefficient / std_error if std_error != 0 else np.nan
                p_value = coef_row["P-Value"]
            else:
                coefficient = np.nan
                std_error = np.nan
                t_value = np.nan
                p_value = "NaN"
            
            rows.append({
                "Source": source,
                "Sum_Sq": row["Sum_Sq"],
                "DF": row["DF"],
                "F": row["F-Statistic"],
                "PR(>F)": row["P-Value"],
                "Coefficient": coefficient,
                "Std_Error": std_error,
                "T-Value": t_value,
                "P-Value": p_value,
                "Conclusion": row["Conclusion"],
            })
        
        return pd.DataFrame(rows)
