"""
Core regression functionality for OLS (Ordinary Least Squares) modeling.

This module handles:
- Data preparation (missing values, categorical encoding)
- OLS model fitting using statsmodels
- Prediction and residual calculation
- Model summary extraction

What is OLS Regression?
-----------------------
OLS (Ordinary Least Squares) is a method for finding the "best-fit" line through data.
It finds coefficients (β values) that minimize the sum of squared differences between
actual values and predicted values.

Think of it like this:
- You have a target variable (what you want to predict, like salary)
- You have predictor variables (things that might affect it, like age and experience)
- OLS finds the formula: salary = β₀ + β₁*age + β₂*experience
- The β values tell you how much each predictor matters
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Dict, List
from .utils import detect_categorical_columns


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    predictor_columns: List[str],
    fail_on_missing: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Prepare data for OLS regression by handling missing values and encoding categoricals.
    
    This function:
    1. Checks for missing values (fails or drops based on parameter)
    2. Automatically detects categorical columns
    3. Converts categorical columns to dummy variables (one-hot encoding)
    4. Drops the first dummy category to avoid multicollinearity
    
    What is Multicollinearity?
    ---------------------------
    When dummy variables are perfectly correlated (if you know all but one,
    you can determine the last one). Dropping the first category prevents this.
    
    Args:
        df: Input pandas DataFrame
        target_column: Name of the target (y) variable
        predictor_columns: List of predictor (X) variable names
        fail_on_missing: If True, raise error on missing values. If False, drop rows with missing values.
        
    Returns:
        Tuple of (X, y, encoded_column_names, original_data_with_dummies):
        - X: DataFrame of predictors (with dummy variables)
        - y: Series of target values
        - encoded_column_names: List of final column names after encoding
        - original_data_with_dummies: Full dataset with dummy variables added
        
    Raises:
        ValueError: If missing values found and fail_on_missing=True
        ValueError: If target or predictor columns not found
        ValueError: If insufficient data after preparation
    """
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
        raise ValueError(
            f"Insufficient data for regression. Only {len(work_df)} complete rows available. "
            "At minimum, 3 observations are required."
        )
    
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
                dummies = pd.get_dummies(
                    original_data_with_dummies[col], 
                    prefix=col, 
                    drop_first=True,
                    dtype=float
                )
                original_data_with_dummies = pd.concat([original_data_with_dummies, dummies], axis=1)
    
    return X, y, encoded_column_names, original_data_with_dummies


def fit_ols_model(X: pd.DataFrame, y: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit an Ordinary Least Squares (OLS) regression model.
    
    This function automatically adds a constant (intercept) term to the model.
    The intercept represents the predicted value when all predictors are zero.
    
    The model finds coefficients (β values) that minimize prediction errors:
    y = β₀ + β₁*X₁ + β₂*X₂ + ... + error
    
    Where:
    - β₀ = intercept (constant term)
    - β₁, β₂, ... = coefficients for each predictor
    - error = residual (difference between actual and predicted)
    
    Args:
        X: DataFrame of predictor variables
        y: Series of target variable
        
    Returns:
        Fitted OLS model (statsmodels RegressionResults object)
        
    Raises:
        ValueError: If model fitting fails (e.g., singular matrix, perfect multicollinearity)
    """
    try:
        # Add constant term (intercept)
        X_with_const = sm.add_constant(X, has_constant='add')
        
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
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate predictions and residuals from a fitted OLS model.
    
    What are Predictions?
    ---------------------
    Predictions are the model's "best guess" for each observation based on
    the predictor values and the learned coefficients.
    
    What are Residuals?
    -------------------
    Residuals are the errors: the difference between actual and predicted values.
    residual = actual - predicted
    
    Positive residual = model underestimated
    Negative residual = model overestimated
    
    Why do residuals matter?
    ------------------------
    In a good regression model, residuals should:
    - Be randomly distributed around zero
    - Have constant variance (homoskedastic)
    - Not show patterns
    
    Heteroskedasticity means the variance of residuals changes systematically,
    which violates OLS assumptions.
    
    Args:
        model: Fitted OLS model
        X: DataFrame of predictor variables (without constant)
        y: Series of actual target values
        
    Returns:
        Tuple of (predictions, residuals):
        - predictions: Model's predicted values
        - residuals: Differences between actual and predicted values
    """
    # Add constant to match model structure
    X_with_const = sm.add_constant(X, has_constant='add')
    
    # Generate predictions
    predictions = model.predict(X_with_const)
    
    # Calculate residuals
    residuals = y - predictions
    
    return predictions, residuals


def extract_model_summary(
    model: sm.regression.linear_model.RegressionResultsWrapper
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract model summary statistics in a structured format.
    
    Returns two tables:
    1. Coefficient table: Details about each predictor
    2. Model metrics: Overall model performance
    
    Understanding the Output
    ------------------------
    
    Coefficient Table:
    - Variable: Name of predictor (or 'const' for intercept)
    - Coefficient (β): How much y changes when this predictor increases by 1
    - Std Error: Uncertainty in the coefficient estimate
    - t-statistic: Coefficient divided by standard error (larger = more significant)
    - P>|t|: Probability this coefficient is actually zero (smaller = more important)
    - Confidence Interval: Range where true coefficient likely falls (95% confidence)
    
    Model Metrics:
    - R²: Proportion of variance explained (0 to 1, higher = better fit)
    - Adjusted R²: R² adjusted for number of predictors (penalizes overfitting)
    - F-statistic: Tests if model is better than just predicting the mean
    - Prob(F): P-value for F-statistic (small = model is useful)
    - N: Number of observations used in the model
    
    Example Interpretation:
    -----------------------
    If age has coefficient = 0.5 and p-value = 0.001:
    "For each additional year of age, the target increases by 0.5 units,
    and this effect is statistically significant (p < 0.05)."
    
    Args:
        model: Fitted OLS model
        
    Returns:
        Tuple of (coefficient_table, metrics_table):
        - coefficient_table: DataFrame with one row per variable
        - metrics_table: DataFrame with model-level statistics
    """
    # Extract coefficient table
    coef_table = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std Error': model.bse.values,
        't-statistic': model.tvalues.values,
        'P>|t|': model.pvalues.values,
        '[0.025': model.conf_int()[0].values,
        '0.975]': model.conf_int()[1].values,
    })
    
    # Extract model metrics
    metrics_table = pd.DataFrame({
        'Metric': [
            'R-squared',
            'Adjusted R-squared',
            'F-statistic',
            'Prob (F-statistic)',
            'No. Observations',
            'Df Residuals',
            'Df Model'
        ],
        'Value': [
            model.rsquared,
            model.rsquared_adj,
            model.fvalue,
            model.f_pvalue,
            float(model.nobs),
            float(model.df_resid),
            float(model.df_model)
        ]
    })
    
    return coef_table, metrics_table
