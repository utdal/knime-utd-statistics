"""
Repeated Measures ANOVA Node for KNIME.

Performs a one-way Repeated Measures ANOVA using statsmodels with Greenhouse-Geisser
sphericity correction. Supports both Long Format (native) and Wide Format (auto-reshaped)
input data, and toggles between a Basic executive-summary output and a full Advanced
technical-validation output.
"""

import knime.extension as knext
import pandas as pd

from .repeated_measures_anova import (
    # Computation
    run_rm_anova,
    build_basic_output,
    build_advanced_output,
    # Parameters
    DataFormat,
    data_format_param,
    dv_column_param,
    within_factor_param,
    subject_id_param,
    wide_columns_param,
    wide_subject_id_param,
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
    description=("Your data table containing participant measurements.\n\nCan be arranged with one row per measurement or one row per participant."),
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
    """

    Repeated Measures ANOVA tests whether the same participants respond differently across multiple conditions or time points.
    Because the same people are measured more than once, the test accounts for individual differences and focuses on whether the changes across conditions are meaningful.

    """

    # ── Parameters ─────────────────────────────────────────────────────────────
    # data_format must be declared first; subsequent .rule() calls reference it
    # at class-body evaluation time, which is how KNIME's rule system works.

    data_format = data_format_param

    # Long Format parameters — shown only when data_format == "LONG"
    dv_column = dv_column_param.rule(
        knext.OneOf(data_format, [DataFormat.LONG.name]),
        knext.Effect.SHOW,
    )
    within_factor = within_factor_param.rule(
        knext.OneOf(data_format, [DataFormat.LONG.name]),
        knext.Effect.SHOW,
    )
    subject_id = subject_id_param.rule(
        knext.OneOf(data_format, [DataFormat.LONG.name]),
        knext.Effect.SHOW,
    )

    # Wide Format parameters — shown only when data_format == "WIDE"
    wide_columns = wide_columns_param.rule(
        knext.OneOf(data_format, [DataFormat.WIDE.name]),
        knext.Effect.SHOW,
    )
    wide_subject_id = wide_subject_id_param.rule(
        knext.OneOf(data_format, [DataFormat.WIDE.name]),
        knext.Effect.SHOW,
    )

    # Shared parameters — always visible
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
        is_wide = self.data_format == DataFormat.WIDE.name

        if is_wide:
            # Wide mode validation
            if self.wide_subject_id is not None:
                try:
                    subj_col = input_spec[self.wide_subject_id]
                    if not is_subject_id(subj_col):
                        raise knext.InvalidParametersError(f"Subject Identifier '{self.wide_subject_id}' must be a string or integer column.")
                except KeyError:
                    pass  # Column not yet present; handled gracefully at execution
        else:
            # Long mode validation
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
          1. Resolve column assignments based on the selected data format.
          2. If Wide Format: reshape with pd.melt() to produce a Long Format DataFrame.
          3. Run pingouin rm_anova() with Greenhouse-Geisser correction.
          4. Return a Basic or Advanced output DataFrame based on the toggle.
        """
        df = input_table.to_pandas()

        is_wide = self.data_format == DataFormat.WIDE.name

        # ── Step 1: Resolve columns ─────────────────────────────────────────

        if is_wide:
            self._validate_wide_params()
            dv, within, subject, df = self._reshape_wide_to_long(df)
        else:
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

    def _validate_wide_params(self) -> None:
        """Check that Wide Format parameters have been configured."""
        if not self.wide_columns or len(list(self.wide_columns)) < 2:
            raise ValueError(
                "Wide Format requires at least 2 Measurement Columns selected. "
                "Each selected column represents one repeated measurement "
                "(e.g., T1, T2, T3). Please configure at least 2 columns."
            )
        if self.wide_subject_id is None:
            raise ValueError(
                "No Subject Identifier selected for Wide Format. "
                "Please open the node configuration and select the column that "
                "uniquely identifies each participant row."
            )

    def _reshape_wide_to_long(self, df: pd.DataFrame):
        """
        Reshape a Wide Format DataFrame to Long Format using pd.melt().

        The column headers of the selected measurement columns become the levels
        of a generated 'Condition' factor column, and their values become 'Value'.

        Returns (dv_name, within_name, subject_name, reshaped_df).
        """
        measurement_cols = list(self.wide_columns)
        subject_col = self.wide_subject_id

        # Check that all selected columns exist in the data
        all_cols = measurement_cols + [subject_col]
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"The following configured column(s) were not found in the input table: {', '.join(missing)}. Please reconfigure the node."
            )

        # Check that measurement columns are numeric
        non_numeric = [c for c in measurement_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(
                f"The following Measurement Column(s) are not numeric: "
                f"{', '.join(non_numeric)}. "
                "Wide Format reshaping requires all selected measurement columns "
                "to contain numeric values."
            )

        dv_name = "Value"
        within_name = "Condition"

        long_df = df.melt(
            id_vars=[subject_col],
            value_vars=measurement_cols,
            var_name=within_name,
            value_name=dv_name,
        )

        return dv_name, within_name, subject_col, long_df
