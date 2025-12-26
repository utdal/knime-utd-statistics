"""
Heteroskedasticity Testing Node for KNIME.

This node performs OLS (Ordinary Least Squares) regression and tests for
heteroskedasticity using one of three statistical tests.

What This Node Does:
--------------------
1. Takes your data with a target variable (y) and predictor variables (X)
2. Automatically handles categorical variables by converting them to dummy variables
3. Fits a regression model: y = β₀ + β₁X₁ + β₂X₂ + ... + error
4. Tests if the error variance is constant (homoskedastic) or changing (heteroskedastic)
5. Returns three outputs:
   - Original data with predictions and residuals
   - Model summary (coefficients, p-values, R²)
   - Heteroskedasticity test results

Why It Matters:
----------------
OLS regression assumes constant error variance (homoskedasticity).
If this assumption is violated:
- Your p-values are unreliable
- Confidence intervals are wrong
- Statistical conclusions may be invalid

This node helps you detect and diagnose such problems.
"""

import knime.extension as knext
import numpy as np
import pandas as pd

from .heteroskedasticity import (
    # Test functions
    run_breusch_pagan_test,
    run_white_test,
    run_goldfeld_quandt_test,
    # Regression core
    prepare_data,
    fit_ols_model,
    generate_predictions_residuals,
    extract_model_summary,
    # Parameters and utilities
    TestType,
    test_type_param,
    target_column_param,
    predictor_columns_param,
    alpha_param,
    gq_sort_variable_param,
    gq_split_fraction_param,
    format_p_value,
)

# Output detail level parameter
output_detail_param = knext.StringParameter(
    label="Output Detail Level",
    description="Choose between basic (essential statistics only) or advanced (full statistical details) model summary output.",
    default_value="Advanced",
    enum=["Basic", "Advanced"],
)


# Create heteroskedasticity category (reuse normality category structure)
heteroskedasticity_category = knext.category(
    path="/community",
    level_id="utd_development",
    name="University of Texas at Dallas Development",
    description="Statistical Analysis Tools",
    icon="./icons/utd.png",
)


@knext.node(
    name="Heteroskedasticity Tests",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="./icons/bell_curve.png",
    category=heteroskedasticity_category,
)
@knext.input_table(name="Input Data", description="Table containing target and predictor variables for regression analysis.")
@knext.output_table(name="Data with Predictions", description="Original data with added prediction and residual columns.")
@knext.output_table(name="Model Summary", description="Regression coefficients, p-values, and model fit statistics.")
@knext.output_table(name="Heteroskedasticity Test", description="Results of the selected heteroskedasticity test.")
class HeteroskedasticityNode:
    """
    KNIME node for OLS regression with heteroskedasticity testing.

    This node combines regression modeling with diagnostic testing to help
    users identify violations of the constant variance assumption.
    """

    # Node parameters
    test_type = test_type_param
    target_column = target_column_param
    predictor_columns = predictor_columns_param
    alpha = alpha_param
    gq_sort_variable = gq_sort_variable_param
    gq_split_fraction = gq_split_fraction_param
    output_detail = output_detail_param

    def configure(self, cfg_ctx, input_spec):
        # Output 1: Data with predictions and residuals
        # Schema will be: original columns + prediction + residual
        # We can't know exact schema until execution, so return None for dynamic schema
        data_schema = None  # Dynamic schema based on input

        # Output 2: Model summary (coefficient table + metrics)
        # Schema depends on output detail level
        if self.output_detail == "Basic":
            # Basic mode: simplified output with essential statistics
            model_summary_cols = [
                knext.Column(knext.string(), "Variable"),
                knext.Column(knext.double(), "Coefficient"),
                knext.Column(knext.string(), "P-Value"),
                knext.Column(knext.string(), "Metric"),
                knext.Column(knext.double(), "Value"),
            ]
        else:
            # Advanced mode: full statistical details
            model_summary_cols = [
                knext.Column(knext.string(), "Variable"),
                knext.Column(knext.double(), "Coefficient"),
                knext.Column(knext.double(), "Std Error"),
                knext.Column(knext.string(), "P>|t|"),
                knext.Column(knext.string(), "Metric"),
                knext.Column(knext.double(), "Value"),
            ]
        model_summary_schema = knext.Schema.from_columns(model_summary_cols)

        # Output 3: Heteroskedasticity test results
        test_results_cols = [
            knext.Column(knext.string(), "Test"),
            knext.Column(knext.double(), "Test Statistic"),
            knext.Column(knext.string(), "P-Value"),
            knext.Column(knext.string(), "Heteroskedasticity"),
            knext.Column(knext.string(), "Interpretation"),
        ]
        test_results_schema = knext.Schema.from_columns(test_results_cols)

        return data_schema, model_summary_schema, test_results_schema

    def execute(self, exec_ctx, input_table):
        # Convert KNIME table to pandas DataFrame
        df = input_table.to_pandas()

        # Step 1: Prepare data (handle missing values, encode categoricals)
        # Always fail on missing values (hard requirement per specification)
        X, y, encoded_column_names, original_data_with_dummies = prepare_data(
            df=df,
            target_column=self.target_column,
            predictor_columns=self.predictor_columns,
            fail_on_missing=True,  # Always fail if missing values present
        )

        # Step 2: Check sample size adequacy
        n_obs = len(X)
        n_predictors = len(encoded_column_names) + 1  # +1 for intercept

        # Soft warning if sample size is small relative to parameters
        if n_obs < n_predictors + 20:
            knext.LOGGER.warning(
                f"Small sample size detected: n={n_obs} observations with k={n_predictors} parameters. "
                f"Ideally, you should have at least n > k + 20 for reliable results. "
                f"Current ratio: n/k = {n_obs / n_predictors:.2f}. "
                "Results may be unreliable."
            )

        # Step 3: Fit OLS regression model
        model = fit_ols_model(X, y)

        # Step 4: Generate predictions and residuals
        predictions, residuals = generate_predictions_residuals(model, X, y)

        if self.test_type == TestType.BREUSCH_PAGAN.name:
            test_result = run_breusch_pagan_test(model, X, self.alpha)

        elif self.test_type == TestType.WHITE.name:
            test_result = run_white_test(model, X, self.alpha)

        elif self.test_type == TestType.GOLDFELD_QUANDT.name:
            # Validate GQ-specific parameters
            if not self.gq_sort_variable:
                raise ValueError("Goldfeld-Quandt test requires a sort variable. Please select a numeric predictor to sort by.")

            test_result = run_goldfeld_quandt_test(
                model, X, y, sort_variable=self.gq_sort_variable, split_fraction=self.gq_split_fraction, alpha=self.alpha
            )
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")

        # Step 6: Format Output 1 - Data with predictions and residuals
        # Only include target, prediction, and residual columns (not excluded predictors)
        output_data = pd.DataFrame({self.target_column: y, "prediction": predictions, "residual": residuals})

        # Step 7: Format Output 2 - Model summary
        coef_table, metrics_table = extract_model_summary(model)

        if self.output_detail == "Basic":
            # Basic mode: simplified output with essential statistics
            coef_table_simple = coef_table[["Variable", "Coefficient", "P>|t|"]].copy()
            coef_table_simple.rename(columns={"P>|t|": "P-Value"}, inplace=True)

            # Format P-Value to avoid scientific notation
            coef_table_simple["P-Value"] = coef_table_simple["P-Value"].apply(format_p_value)

            # Add empty metric columns to coef table
            coef_table_simple["Metric"] = ""
            coef_table_simple["Value"] = np.nan

            # Filter metrics to only essential ones
            key_metrics = ["R-squared", "Adjusted R-squared", "F-statistic", "Prob (F-statistic)", "No. Observations"]
            metrics_table_simple = metrics_table[metrics_table["Metric"].isin(key_metrics)].copy()

            # Add empty variable columns to metrics table
            metrics_table_simple.insert(0, "Variable", "")
            metrics_table_simple.insert(1, "Coefficient", np.nan)
            metrics_table_simple.insert(2, "P-Value", np.nan)

            # Combine both tables
            model_summary = pd.concat([coef_table_simple, metrics_table_simple], ignore_index=True)
        else:
            # Advanced mode: full statistical details (without t-statistic and confidence intervals)
            coef_table_advanced = coef_table[["Variable", "Coefficient", "Std Error", "P>|t|"]].copy()

            # Format P>|t| to avoid scientific notation
            coef_table_advanced["P>|t|"] = coef_table_advanced["P>|t|"].apply(format_p_value)

            # Add empty metric columns to coef table
            coef_table_advanced["Metric"] = ""
            coef_table_advanced["Value"] = np.nan

            # Add empty variable columns to metrics table
            metrics_table.insert(0, "Variable", "")
            metrics_table.insert(1, "Coefficient", np.nan)
            metrics_table.insert(2, "Std Error", np.nan)
            metrics_table.insert(3, "P>|t|", np.nan)

            # Combine both tables
            model_summary = pd.concat([coef_table_advanced, metrics_table], ignore_index=True)

        # Step 8: Format Output 3 - Test results
        test_results_df = pd.DataFrame(
            [
                {
                    "Test": test_result["test"],
                    "Test Statistic": test_result["statistic"],
                    "P-Value": test_result["p_value"],
                    "Heteroskedasticity": test_result["is_heteroskedastic"],
                    "Interpretation": test_result["interpretation"],
                }
            ]
        )

        # Format P-Value to avoid scientific notation (E-22, etc.)
        test_results_df["P-Value"] = test_results_df["P-Value"].apply(format_p_value)

        # Convert to KNIME tables
        output_table_1 = knext.Table.from_pandas(output_data)
        output_table_2 = knext.Table.from_pandas(model_summary)
        output_table_3 = knext.Table.from_pandas(test_results_df)

        return output_table_1, output_table_2, output_table_3
