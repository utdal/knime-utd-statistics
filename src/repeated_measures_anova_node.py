"""
Repeated Measures ANOVA Node for KNIME.

Performs a one-way Repeated Measures ANOVA using statsmodels with Greenhouse-Geisser
sphericity correction. Accepts Long Format input data only, and toggles between a
Basic executive-summary output and a full Advanced technical-validation output.
"""

import knime.extension as knext
import pandas as pd

from .repeated_measures_anova import (
    # Computation
    run_rm_anova,
    build_basic_output,
    build_advanced_output,
    # Parameters
    dv_column_param,
    within_factor_param,
    subject_id_param,
    alpha_param,
    advanced_output_param,
    # Utilities
    is_numeric,
    is_nominal,
    is_subject_id,
)


# ── Category ───────────────────────────────────────────────────────────────────

utd_category = knext.category(
    path="/community",
    level_id="utd_development",
    name="University of Texas at Dallas Development",
    description="Statistical analysis tools developed by the University of Texas at Dallas",
    icon="./icons/utd.png",
)


# ── Node Definition ────────────────────────────────────────────────────────────


@knext.node(
    name="Repeated Measures ANOVA",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="./icons/rm_anova.png",
    category=utd_category,
)
@knext.input_table(
    name="Input Data",
    description=(
        "Your data in long format: one row per measurement, with separate columns for "
        "the measured value, the condition/time point, and the participant ID."
    ),
)
@knext.output_table(
    name="RM ANOVA Results",
    description=(
        "The ANOVA results table.\n\n"
        "Basic mode: A single summary row showing the corrected p-value, "
        "effect size, and conclusion.\n\n"
        "Advanced mode: Two rows with the complete statistical breakdown "
        "for detailed inspection."
    ),
)
class RepeatedMeasuresAnovaNode:
    """Performs repeated measures ANOVA to test for mean differences in subjects across multiple conditions/time points.

    Repeated Measures ANOVA tests whether the same participants respond differently across multiple conditions or time points.
    Because the same people are measured more than once, the test accounts for individual differences and focuses on whether the changes across conditions are meaningful.
    Input must be in long format.

    """

    # ── Parameters ─────────────────────────────────────────────────────────────

    dv_column = dv_column_param
    within_factor = within_factor_param
    subject_id = subject_id_param

    alpha = alpha_param
    advanced_output = advanced_output_param

    # ── Configure ──────────────────────────────────────────────────────────────

    def configure(self, cfg_ctx, input_spec):
        """
        Validate column selections and return the output table schema.

        The schema switches between Basic (4 columns) and Advanced (12 columns)
        based on the advanced_output toggle, so downstream nodes always receive
        a predictable structure.
        """
        if self.dv_column is not None:
            try:
                dv_col = input_spec[self.dv_column]
                if not is_numeric(dv_col):
                    raise knext.InvalidParametersError(
                        f"Dependent Variable '{self.dv_column}' must be numeric "
                        f"(current type: {dv_col.ktype}). "
                        "Repeated Measures ANOVA requires a continuous numeric outcome."
                    )
            except KeyError:
                pass

        if self.within_factor is not None:
            try:
                wf_col = input_spec[self.within_factor]
                if not is_nominal(wf_col):
                    raise knext.InvalidParametersError(
                        f"Within-Subject Factor '{self.within_factor}' must be a string/categorical column (current type: {wf_col.ktype})."
                    )
            except KeyError:
                pass

        if self.subject_id is not None:
            try:
                sid_col = input_spec[self.subject_id]
                if not is_subject_id(sid_col):
                    raise knext.InvalidParametersError(
                        f"Subject Identifier '{self.subject_id}' must be a string or integer column (current type: {sid_col.ktype})."
                    )
            except KeyError:
                pass

        # Output schema — switches on the advanced_output toggle
        if self.advanced_output:
            output_cols = [
                knext.Column(knext.string(), "Source"),
                knext.Column(knext.double(), "Sum of Squares"),
                knext.Column(knext.double(), "Degrees of Freedom"),
                knext.Column(knext.double(), "Mean Square"),
                knext.Column(knext.double(), "F Statistic"),
                knext.Column(knext.double(), "P-Value (Uncorrected)"),
                knext.Column(knext.double(), "P-Value (Greenhouse-Geisser Corrected)"),
                knext.Column(knext.double(), "Mauchly's W"),
                knext.Column(knext.double(), "Mauchly's P-Value"),
                knext.Column(knext.double(), "Epsilon (Greenhouse-Geisser)"),
                knext.Column(knext.double(), "Effect Size (Partial Eta Squared)"),
                knext.Column(knext.string(), "Conclusion"),
            ]
        else:
            output_cols = [
                knext.Column(knext.string(), "Source"),
                knext.Column(knext.double(), "P-Value (Greenhouse-Geisser Corrected)"),
                knext.Column(knext.double(), "Effect Size (Partial Eta Squared)"),
                knext.Column(knext.string(), "Conclusion"),
            ]

        return knext.Schema.from_columns(output_cols)

    # ── Execute ────────────────────────────────────────────────────────────────

    def execute(self, exec_ctx, input_table):
        """
        Execute the Repeated Measures ANOVA.

        Steps:
          1. Validate column assignments.
          2. Run RM ANOVA with Greenhouse-Geisser correction.
          3. Return a Basic or Advanced output DataFrame based on the toggle.
        """
        df = input_table.to_pandas()

        # ── Step 1: Resolve columns ─────────────────────────────────────────

        self._validate_long_params(df)
        dv = self.dv_column
        within = self.within_factor
        subject = self.subject_id

        # ── Step 2: Run RM ANOVA ────────────────────────────────────────────

        result = run_rm_anova(df, dv=dv, within=within, subject=subject, alpha=self.alpha)

        # ── Step 3: Build output ────────────────────────────────────────────

        if self.advanced_output:
            output_df = build_advanced_output(result, alpha=self.alpha)
        else:
            output_df = build_basic_output(result, alpha=self.alpha)

        return knext.Table.from_pandas(output_df)

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _validate_long_params(self, df: pd.DataFrame) -> None:
        """Check that all three Long Format column selectors have been configured."""
        if self.dv_column is None:
            raise ValueError("No Dependent Variable selected. Please open the node configuration and select a numeric column.")
        if self.within_factor is None:
            raise ValueError(
                "No Within-Subject Factor selected. "
                "Please open the node configuration and select a categorical column "
                "that identifies conditions or time points."
            )
        if self.subject_id is None:
            raise ValueError(
                "No Subject Identifier selected. Please open the node configuration and select the column that uniquely identifies each participant."
            )

        # Existence check for columns in the actual data
        missing = [col for col in [self.dv_column, self.within_factor, self.subject_id] if col not in df.columns]
        if missing:
            raise ValueError(
                f"The following configured column(s) were not found in the input table: "
                f"{', '.join(missing)}. "
                "This can happen when the upstream table has changed. "
                "Please reconfigure the node."
            )
