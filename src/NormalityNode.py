"""
Unified Normality Tests Node for KNIME.

This module provides a single KNIME node that allows users to choose between
different normality tests (Anderson-Darling, Cramer-von Mises) with test-specific
parameters displayed conditionally.
"""

import knime.extension as knext
import numpy as np
import pandas as pd
from .normalityTests import run_ad_test, run_cramer_test


# Create normality tests category
normality_category = knext.category(
    path="/community",
    level_id="normality_tests",
    name="Normality Tests",
    description="Statistical normality testing nodes",
    icon="./icons/icon.png",
)


def _is_numeric(col: knext.Column) -> bool:
    """Helper function to filter for numeric columns."""
    return col.ktype in (knext.double(), knext.int32(), knext.int64())


@knext.parameter_group("Normality Test Settings")
class _NormalityParams:
    """Parameter group for unified normality testing node."""

    # Test type selection
    class _TestType(knext.EnumParameterOptions):
        ANDERSON_DARLING = (
            "Anderson-Darling",
            "Anderson-Darling normality test with critical values or p-values",
        )
        CRAMER = (
            "Cramer-von Mises",
            "Cramer-von Mises normality test with critical values",
        )

    test_type = knext.EnumParameter(
        label="Test Type",
        description="Choose the normality test to perform",
        enum=_TestType,
        default_value=_TestType.ANDERSON_DARLING.name,
    )

    # Common parameters for all tests
    input_column = knext.ColumnParameter(
        label="Data column",
        description="Numeric column to test for normality.",
        column_filter=_is_numeric,
    )

    alpha = knext.DoubleParameter(
        label="Alpha (significance level)",
        description="Decision threshold (e.g., 0.05).",
        default_value=0.05,
        min_value=1e-6,
        max_value=0.5,
    )

    # Mu/Sigma handling (common to both tests)
    class _MuSigma(knext.EnumParameterOptions):
        ESTIMATE = ("Estimate from data", "Estimate μ, σ from the selected column.")
        USER = ("User specified", "Use user-provided μ and σ.")

    mu_sigma_mode = knext.EnumParameter(
        label="μ/σ source",
        description="Choose whether to estimate μ,σ from the data or provide them.",
        enum=_MuSigma,
        default_value=_MuSigma.ESTIMATE.name,
    )

    mu = knext.DoubleParameter(
        label="μ (mean) if user-specified",
        description="Used only if μ/σ source = User specified.",
        default_value=0.0,
    )

    sigma = knext.DoubleParameter(
        label="σ (std) if user-specified",
        description="Must be > 0 when μ/σ source = User specified.",
        default_value=1.0,
        min_value=1e-12,
    )

    # Standardization (common to both tests)
    standardize = knext.BoolParameter(
        label="Standardize (z-score) data before test",
        description="Subtract μ and divide by σ prior to computation.",
        default_value=False,
    )

    # Performance parameters (common to both tests)
    sample_cap = knext.IntParameter(
        label="Sampling cap (rows)",
        description="Optional: uniformly sample to this many rows for performance. Leave empty or 0 for no cap.",
        default_value=0,
        min_value=0,
    )

    seed = knext.IntParameter(
        label="Random seed (sampling)",
        description="Used only if a sampling cap is applied.",
        default_value=42,
    )

    # Small-sample correction (common to both tests)
    class _SSCorr(knext.EnumParameterOptions):
        AUTO = ("Auto", "Apply the common correction for estimated parameters.")
        OFF = ("Off", "Do not apply the small-sample correction.")

    small_sample_correction = knext.EnumParameter(
        label="Small-sample correction",
        description="Adjustment used when parameters are estimated from data.",
        enum=_SSCorr,
        default_value=_SSCorr.AUTO.name,
    )

    # Anderson-Darling specific parameters
    class _SigMode(knext.EnumParameterOptions):
        CRITICAL = (
            "Critical values (SciPy)",
            "Decision via critical values; no p-value.",
        )
        PVALUE = (
            "p-value (statsmodels)",
            "Uses statsmodels to compute an AD p-value (if available).",
        )

    ad_significance_mode = knext.EnumParameter(
        label="Significance mode (Anderson-Darling)",
        description="Choose how to map the AD statistic to a decision.",
        enum=_SigMode,
        default_value=_SigMode.CRITICAL.name,
    )

    class _CritPolicy(knext.EnumParameterOptions):
        INTERP = (
            "Interpolate",
            "Linearly interpolate critical value for non-tabulated alpha.",
        )
        NEAREST = ("Nearest level", "Use nearest tabulated alpha level.")

    ad_critical_policy = knext.EnumParameter(
        label="Critical-value policy (Anderson-Darling)",
        description="How to get critical value if alpha is not a tabulated level.",
        enum=_CritPolicy,
        default_value=_CritPolicy.INTERP.name,
    )


@knext.node(
    name="Normality Tests",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="./icons/icon.png",
    category=normality_category,
)
@knext.input_table(
    name="Input data", description="Table containing the numeric column to test."
)
@knext.output_table(
    name="Results",
    description="Summary of normality test with clear statistical decision and key metrics.",
)
@knext.output_table(
    name="Diagnostics",
    description="Comprehensive diagnostic information organized by category for full transparency.",
)
class NormalityTestsNode:
    """
    Unified normality testing node supporting multiple test types.

    Features:
    • Choose between Anderson-Darling and Cramer-von Mises tests
    • Test-specific parameters shown conditionally
    • Standardized output format for all tests
    • Comprehensive diagnostic information
    """

    params = _NormalityParams()

    def configure(self, cfg_ctx, input_spec):
        """Configure the node's output table schema."""
        # Check for warnings specific to Anderson-Darling
        if (
            self.params.test_type == _NormalityParams._TestType.ANDERSON_DARLING.name
            and self.params.ad_significance_mode
            == _NormalityParams._SigMode.PVALUE.name
        ):
            try:
                from statsmodels.stats.diagnostic import normal_ad
            except ImportError:
                cfg_ctx.set_warning(
                    "Anderson-Darling significance mode is set to p-value, but 'statsmodels' is not available. "
                    "The node will fall back to critical-value mode at execute time."
                )

        # Define flexible schema that works for both test types
        results_cols = [
            knext.Column(knext.string(), "Test"),
            knext.Column(knext.string(), "Column Tested"),
            knext.Column(knext.int32(), "Sample Size (n)"),
            knext.Column(knext.double(), "Test Statistic"),
            knext.Column(knext.string(), "P-Value"),
            knext.Column(knext.double(), "Significance Level"),
            knext.Column(knext.string(), "Statistical Decision"),
            knext.Column(knext.double(), "Critical Value"),
        ]

        diag_cols = [
            knext.Column(knext.string(), "Category"),
            knext.Column(knext.string(), "Detail"),
            knext.Column(knext.string(), "Value"),
        ]

        results_schema = knext.Schema.from_columns(results_cols)
        diag_schema = knext.Schema.from_columns(diag_cols)

        return [results_schema, diag_schema]

    def execute(self, exec_ctx, input_table):
        """Execute the selected normality test."""
        df = input_table.to_pandas()
        col_name = self.params.input_column

        # Guard against no column selection
        if col_name is None:
            raise ValueError(
                "No column selected. Please configure the node and select a numeric data column."
            )

        # Guard against invalid column selection
        if col_name not in df.columns:
            raise ValueError(f"Selected column '{col_name}' not found in input data.")

        # Prepare common parameters
        common_params = {
            "alpha": self.params.alpha,
            "mu_sigma_mode": "estimate"
            if self.params.mu_sigma_mode == _NormalityParams._MuSigma.ESTIMATE.name
            else "user",
            "mu": self.params.mu
            if self.params.mu_sigma_mode == _NormalityParams._MuSigma.USER.name
            else None,
            "sigma": self.params.sigma
            if self.params.mu_sigma_mode == _NormalityParams._MuSigma.USER.name
            else None,
            "standardize": bool(self.params.standardize),
            "sample_cap": int(self.params.sample_cap)
            if int(self.params.sample_cap) > 0
            else None,
            "seed": int(self.params.seed),
            "small_sample_correction": "auto"
            if self.params.small_sample_correction == _NormalityParams._SSCorr.AUTO.name
            else "off",
        }

        # Execute the selected test
        if self.params.test_type == _NormalityParams._TestType.ANDERSON_DARLING.name:
            # Add Anderson-Darling specific parameters
            ad_params = {
                **common_params,
                "significance_mode": "critical"
                if self.params.ad_significance_mode
                == _NormalityParams._SigMode.CRITICAL.name
                else "pvalue",
                "critical_policy": "interpolate"
                if self.params.ad_critical_policy
                == _NormalityParams._CritPolicy.INTERP.name
                else "nearest",
            }
            result = run_ad_test(df[col_name], **ad_params)

        else:  # Cramer-von Mises test
            result = run_cramer_test(df[col_name], **common_params)

        # Format results in standardized way
        return self._format_results(result, col_name)

    def _format_results(self, result: dict, col_name: str) -> tuple:
        """Format test results into standardized KNIME tables."""
        summary = result["summary"]

        # Extract test statistic (different key names for different tests)
        if "statistic_A2" in summary:
            test_statistic = summary["statistic_A2"]
        elif "statistic_W2" in summary:
            test_statistic = summary["statistic_W2"]
        else:
            test_statistic = float("nan")

        # Format p-value display
        if summary["p_value"] is not None:
            p_value_display = f"{summary['p_value']:.6f}"
        else:
            p_value_display = "Not calculated - using critical values"

        # Create results DataFrame
        results_df = pd.DataFrame(
            [
                {
                    "Test": summary["test"],
                    "Column Tested": col_name,
                    "Sample Size (n)": np.int32(summary["n"]),
                    "Test Statistic": test_statistic,
                    "P-Value": p_value_display,
                    "Significance Level": summary["alpha"],
                    "Statistical Decision": summary["decision"],
                    "Critical Value": summary["critical_value"]
                    if summary["critical_value"] is not None
                    else float("nan"),
                }
            ]
        )

        # Create diagnostics table with organized sections
        diag = result["diagnostics"]
        rows = []

        def _put_section(section_name):
            """Add a section header for organization"""
            rows.append(
                {"Category": f"{section_name.upper()}", "Detail": "", "Value": ""}
            )

        def _put(category, detail, value, format_func=None):
            """Add a diagnostic entry with clean formatting"""
            if value is None:
                formatted_value = "N/A"
            elif format_func:
                formatted_value = format_func(value)
            else:
                formatted_value = str(value)
            rows.append(
                {"Category": category, "Detail": detail, "Value": formatted_value}
            )

        # Data Quality Section
        _put_section("Data Quality")
        _put("Sample Size", "Original data points", diag["n_raw"])
        _put("Valid Data", "Points used in analysis", diag["n_used"])
        _put("Excluded", "Invalid/missing points removed", diag["n_dropped"])
        _put(
            "Sampling",
            "Performance sampling applied",
            "Yes" if diag["sampling_applied"] else "No",
        )

        # Parameter Configuration Section
        _put_section("Parameters")
        _put(
            "μ/σ Source",
            "Parameter estimation method",
            "From data" if diag["mu_sigma_mode"] == "estimate" else "User specified",
        )
        _put(
            "Mean (μ)",
            "Distribution center used",
            diag["mu_used"],
            lambda x: f"{float(x):.6f}",
        )
        _put(
            "Std Dev (σ)",
            "Distribution spread used",
            diag["sigma_used"],
            lambda x: f"{float(x):.6f}",
        )
        _put(
            "Standardized",
            "Data z-scored before test",
            "Yes" if diag["standardized"] else "No",
        )

        # Test Configuration Section
        _put_section("Test Setup")
        _put("Test Type", "Selected normality test", summary["test"])
        _put(
            "Decision Method",
            "Significance assessment mode",
            "Critical values"
            if diag["significance_mode"] == "critical"
            else "P-value computation",
        )
        _put("Backend", "Statistical computation engine", diag["backend"])

        # Statistical Details Section (if available)
        if result.get("aux") is not None:
            aux = result["aux"]
            _put_section("Statistical Details")
            _put(
                "Critical Value",
                f"Threshold at α={summary['alpha']}",
                aux.get("critical_value_at_alpha"),
                lambda x: f"{float(x):.4f}",
            )
            _put(
                "Mapping Method",
                "Critical value determination",
                aux.get("alpha_mapping"),
            )
            if aux.get("table_levels"):
                levels_str = ", ".join(
                    [f"{float(x):.3f}" for x in aux.get("table_levels", [])]
                )
                _put("Available Levels", "Reference significance levels", levels_str)

        # Important Notes Section
        notes = diag.get("notes", [])
        if notes:
            _put_section("Important Notes")
            for i, note in enumerate(notes, 1):
                _put("Note", f"Advisory #{i}", note)

        diag_df = pd.DataFrame(rows, columns=["Category", "Detail", "Value"])

        return (knext.Table.from_pandas(results_df), knext.Table.from_pandas(diag_df))
