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
    TestType,  # Enum for test types - needed for .rule() conditions
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
    description=(
        "Choose how much statistical detail to include in the Model Summary output.\n\n"
        "• Basic: Easy-to-read output with essential coefficients and key model statistics.\n"
        "• Advanced: Full statistical detail including coefficient types, standard errors, and degrees of freedom."
    ),
    default_value="Advanced",
    enum=["Basic", "Advanced"],
)


# UTD statistical analysis category
utd_category = knext.category(
    path="/community",
    level_id="utd_development",
    name="University of Texas at Dallas Development",
    description="Statistical analysis tools developed by the University of Texas at Dallas",
    icon="./icons/utd.png",
)


@knext.node(
    name="Heteroskedasticity Tests",
    node_type=knext.NodeType.MANIPULATOR,
    icon_path="./icons/bell_curve.png",
    category=utd_category,
)
@knext.input_table(name="Input Data", description="Data table containing the target variable and predictor variables for regression analysis.")
@knext.output_table(
    name="Data with Predictions",
    description="Output table containing the target variable, model predictions, and residuals (actual minus predicted values).",
)
@knext.output_table(
    name="Model Summary",
    description="Output table containing regression coefficients and key model statistics. Output format depends on your Basic vs Advanced selection.",
)
@knext.output_table(
    name="Heteroskedasticity Test",
    description="Heteroskedasticity test results with test statistic, p-value, and statistical decision.",
)
class HeteroskedasticityNode:
    """
    Tests whether a regression model shows heteroskedasticity (non-constant error variance).

    This node first fits an ordinary least squares (OLS) regression model from your selected target
    and predictor variables. It then runs the selected heteroskedasticity test (Breusch-Pagan, White,
    or Goldfeld-Quandt) to determine whether the model residual variance appears constant.

    Output includes the target variable with predictions/residuals, a regression model summary, and
    a single-row test result with a clear statistical decision based on your chosen significance level.
    """
    # Node parameters
    test_type = test_type_param
    target_column = target_column_param
    predictor_columns = predictor_columns_param
    alpha = alpha_param
    output_detail = output_detail_param

    # Goldfeld-Quandt specific parameters - only visible when GQ test is selected
    gq_sort_variable = gq_sort_variable_param.rule(knext.OneOf(test_type, [TestType.GOLDFELD_QUANDT.name]), knext.Effect.SHOW)
    gq_split_fraction = gq_split_fraction_param.rule(knext.OneOf(test_type, [TestType.GOLDFELD_QUANDT.name]), knext.Effect.SHOW)

    def configure(self, cfg_ctx, input_spec):
        # Output 1: Data with predictions and residuals
        # Schema will be: original columns + prediction + residual
        # We can't know exact schema until execution, so return None for dynamic schema
        data_schema = None  # Dynamic schema based on input

        # Output 2: Model summary (coefficient table + metrics)
        # Schema depends on output detail level
        if self.output_detail == "Basic":
            # Simple mode: end-user friendly with essential statistics
            model_summary_cols = [
                knext.Column(knext.string(), "Information"),
                knext.Column(knext.double(), "Measure"),
                knext.Column(knext.string(), "P-Value"),
            ]
        else:
            # Advanced mode: full statistical details with type classification
            model_summary_cols = [
                knext.Column(knext.string(), "Type"),
                knext.Column(knext.string(), "Information"),
                knext.Column(knext.double(), "Measure"),
                knext.Column(knext.double(), "Std Error"),
                knext.Column(knext.string(), "P-Value"),
            ]
        model_summary_schema = knext.Schema.from_columns(model_summary_cols)

        # Output 3: Heteroskedasticity test results
        test_results_cols = [
            knext.Column(knext.string(), "Test"),
            knext.Column(knext.double(), "Test Statistic"),
            knext.Column(knext.string(), "P-Value"),
            knext.Column(knext.string(), "Heteroskedasticity"),
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

        # Extract F-statistic p-value for use in F-statistic row
        f_pvalue = metrics_table[metrics_table["Metric"] == "Prob (F-statistic)"]["Value"].values[0]

        if self.output_detail == "Basic":
            # SIMPLE MODE: End-user friendly with essential statistics
            # Schema: Information, Measure, P-Value

            # Process coefficients
            simple_rows = []
            for _, row in coef_table.iterrows():
                simple_rows.append({"Information": row["Variable"], "Measure": row["Coefficient"], "P-Value": format_p_value(row["P>|t|"])})

            # Process key metrics (filtered and ordered)
            key_metrics = ["R-squared", "Adjusted R-squared", "F-statistic", "No. Observations"]
            for metric_name in key_metrics:
                metric_row = metrics_table[metrics_table["Metric"] == metric_name]
                if not metric_row.empty:
                    # F-statistic gets the p-value from Prob (F-statistic)
                    p_val = format_p_value(f_pvalue) if metric_name == "F-statistic" else ""
                    simple_rows.append({"Information": metric_name, "Measure": metric_row["Value"].values[0], "P-Value": p_val})

            model_summary = pd.DataFrame(simple_rows)

        else:
            # ADVANCED MODE: Full statistical details with type classification
            # Schema: Type, Information, Measure, Std Error, P-Value

            # Process coefficients (Type = "Coefficient")
            advanced_rows = []
            for _, row in coef_table.iterrows():
                advanced_rows.append(
                    {
                        "Type": "Coefficient",
                        "Information": row["Variable"],
                        "Measure": row["Coefficient"],
                        "Std Error": row["Std Error"],
                        "P-Value": format_p_value(row["P>|t|"]),
                    }
                )

            # Process metrics with appropriate Type classification (ordered)
            # Excludes Prob (F-statistic) - its value goes in F-statistic's P-Value
            advanced_metrics = ["R-squared", "Adjusted R-squared", "F-statistic", "No. Observations", "Df Residuals", "Df Model"]
            for metric_name in advanced_metrics:
                metric_row = metrics_table[metrics_table["Metric"] == metric_name]
                if not metric_row.empty:
                    # Determine Type based on metric name
                    if metric_name == "F-statistic":
                        row_type = "Test statistic"
                        p_val = format_p_value(f_pvalue)
                    else:
                        row_type = "Model statistic"
                        p_val = ""

                    advanced_rows.append(
                        {
                            "Type": row_type,
                            "Information": metric_name,
                            "Measure": metric_row["Value"].values[0],
                            "Std Error": np.nan,
                            "P-Value": p_val,
                        }
                    )

            model_summary = pd.DataFrame(advanced_rows)

        # Step 8: Format Output 3 - Test results
        test_results_df = pd.DataFrame(
            [
                {
                    "Test": test_result["test"],
                    "Test Statistic": test_result["statistic"],
                    "P-Value": test_result["p_value"],
                    "Heteroskedasticity": test_result["is_heteroskedastic"],
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
