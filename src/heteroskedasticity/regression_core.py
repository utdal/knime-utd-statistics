import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, List
from .utils import detect_categorical_columns


def prepare_data(
    df: pd.DataFrame, target_column: str, predictor_columns: List[str], fail_on_missing: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    # Validate columns exist
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in input data.")

    missing_predictors = [col for col in predictor_columns if col not in df.columns]
    if missing_predictors:
        raise ValueError(f"Predictor columns not found: {missing_predictors}")

    # Select relevant columns
    relevant_columns = [target_column] + predictor_columns
    work_df = df[relevant_columns].copy()

    # Check for missing values
    missing_count = work_df.isnull().sum().sum()
    if missing_count > 0:
        if fail_on_missing:
            missing_by_col = work_df.isnull().sum()
            missing_info = missing_by_col[missing_by_col > 0].to_dict()
            raise ValueError(
                f"Missing values detected in {missing_count} cells across columns: {missing_info}. "
                "Please clean your data or configure the node to drop rows with missing values."
            )
        else:
            # Drop rows with any missing values
            original_len = len(work_df)
            work_df = work_df.dropna()
            dropped_count = original_len - len(work_df)
            if dropped_count > 0:
                print(f"Dropped {dropped_count} rows containing missing values.")

    # Check if we have enough data left
    if len(work_df) < 3:
        raise ValueError(f"Insufficient data for regression. Only {len(work_df)} complete rows available. At minimum, 3 observations are required.")

    # Detect categorical columns (automatic detection only)
    categorical_cols = detect_categorical_columns(work_df, predictor_columns)

    # Separate predictors from target
    X = work_df[predictor_columns].copy()
    y = work_df[target_column].copy()

    # Encode categorical variables with drop_first=True
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)
        print(f"Encoded categorical columns: {categorical_cols}")
        print(f"Created dummy variables: {X.columns.tolist()}")

    # Store encoded column names
    encoded_column_names = X.columns.tolist()

    # Create full dataset with dummy variables for output
    original_data_with_dummies = df.copy()
    if categorical_cols:
        # Add dummy variables to original dataframe
        for col in categorical_cols:
            if col in original_data_with_dummies.columns:
                dummies = pd.get_dummies(original_data_with_dummies[col], prefix=col, drop_first=True, dtype=float)
                original_data_with_dummies = pd.concat([original_data_with_dummies, dummies], axis=1)

    return X, y, encoded_column_names, original_data_with_dummies


def fit_ols_model(X: pd.DataFrame, y: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    try:
        # Add constant term (intercept)
        X_with_const = sm.add_constant(X, has_constant="add")

        # Fit OLS model
        model = sm.OLS(y, X_with_const).fit()

        return model

    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Failed to fit regression model. This often happens when:\n"
            "1. Predictors are perfectly correlated (multicollinearity)\n"
            "2. You have more predictors than observations\n"
            "3. A predictor has constant/identical values\n"
            f"Technical error: {str(e)}"
        )
    except Exception as e:
        raise ValueError(f"Unexpected error fitting OLS model: {str(e)}")


def generate_predictions_residuals(
    model: sm.regression.linear_model.RegressionResultsWrapper, X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    # Add constant to match model structure
    X_with_const = sm.add_constant(X, has_constant="add")

    # Generate predictions
    predictions = model.predict(X_with_const)

    # Calculate residuals
    residuals = y - predictions

    return predictions, residuals


def extract_model_summary(model: sm.regression.linear_model.RegressionResultsWrapper) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Extract coefficient table
    coef_table = pd.DataFrame(
        {
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "Std Error": model.bse.values,
            "t-statistic": model.tvalues.values,
            "P>|t|": model.pvalues.values,
            "[0.025": model.conf_int()[0].values,
            "0.975]": model.conf_int()[1].values,
        }
    )

    # Extract model metrics
    metrics_table = pd.DataFrame(
        {
            "Metric": ["R-squared", "Adjusted R-squared", "F-statistic", "Prob (F-statistic)", "No. Observations", "Df Residuals", "Df Model"],
            "Value": [
                model.rsquared,
                model.rsquared_adj,
                model.fvalue,
                model.f_pvalue,
                float(model.nobs),
                float(model.df_resid),
                float(model.df_model),
            ],
        }
    )

    return coef_table, metrics_table
