"""
Analysis utilities for oWAR notebooks.

Provides advanced analysis functions:
- Elite/replacement level performance analysis
- Error analysis by group
- SHAP feature importance
- Outlier detection
- Multi-model comparison
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional SHAP import (graceful degradation)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def calculate_elite_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 5.0
) -> Dict[str, float]:
    """
    Calculate MAE for elite players only.

    Args:
        y_true: Actual WAR values
        y_pred: Predicted WAR values
        threshold: Elite threshold (default: 5.0)

    Returns:
        dict: {'elite_MAE': float, 'elite_count': int}

    Example:
        >>> elite_metrics = calculate_elite_performance(y_val, y_pred, threshold=5.0)
        >>> # Returns: {'elite_MAE': 0.68, 'elite_count': 15}
    """
    # Filter to elite players
    elite_mask = y_true > threshold

    if elite_mask.sum() == 0:
        return {'elite_MAE': np.nan, 'elite_count': 0}  # No elite players

    elite_y_true = y_true[elite_mask]
    elite_y_pred = y_pred[elite_mask]

    return {
        'elite_MAE': mean_absolute_error(elite_y_true, elite_y_pred),
        'elite_count': int(elite_mask.sum())
    }


def calculate_replacement_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.0
) -> float:
    """
    Calculate MAE for replacement level players.

    Args:
        y_true: Actual WAR values
        y_pred: Predicted WAR values
        threshold: Replacement threshold (default: 0.0)

    Returns:
        float: MAE for players with actual WAR < threshold

    Example:
        >>> replacement_mae = calculate_replacement_performance(y_val, y_pred)
        >>> # Returns: 0.42
    """
    # Filter to replacement level players
    replacement_mask = y_true < threshold

    if replacement_mask.sum() == 0:
        return np.nan  # No replacement level players

    replacement_y_true = y_true[replacement_mask]
    replacement_y_pred = y_pred[replacement_mask]

    return mean_absolute_error(replacement_y_true, replacement_y_pred)


def analyze_errors_by_group(
    residuals: np.ndarray,
    groups: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Analyze prediction errors by group.

    Args:
        residuals: Prediction residuals (actual - predicted)
        groups: Group labels

    Returns:
        dict: {group: {stat: value}}

    Stats Included:
        - count: Number of samples
        - MAE: Mean absolute error
        - RMSE: Root mean squared error
        - mean_error: Mean error (bias)
        - std_error: Standard deviation of errors
        - median: Median error
        - iqr: Interquartile range
        - large_over: Count of over-predictions >1 WAR
        - large_under: Count of under-predictions <-1 WAR

    Example:
        >>> error_stats = analyze_errors_by_group(residuals, roles)
        >>> # Returns:
        >>> # {
        >>> #   'Starter': {'count': 236, 'MAE': 0.68, 'RMSE': 0.91, ...},
        >>> #   'Reliever': {'count': 326, 'MAE': 0.61, 'RMSE': 0.84, ...}
        >>> # }
    """
    unique_groups = np.unique(groups)

    results = {}

    for group in unique_groups:
        group_mask = groups == group
        group_residuals = residuals[group_mask]

        # Calculate stats
        results[group] = {
            'count': int(len(group_residuals)),
            'MAE': float(np.abs(group_residuals).mean()),
            'RMSE': float(np.sqrt((group_residuals ** 2).mean())),
            'mean_error': float(np.mean(group_residuals)),
            'std_error': float(np.std(group_residuals)),
            'median': float(np.median(group_residuals)),
            'iqr': float(np.percentile(group_residuals, 75) - np.percentile(group_residuals, 25)),
            'large_over': int(np.sum(group_residuals > 1.0)),  # Over-predictions
            'large_under': int(np.sum(group_residuals < -1.0))  # Under-predictions
        }

    return results


def calculate_shap_values(
    model,
    X: pd.DataFrame,
    background_samples: int = 100
):
    """
    Calculate SHAP values for feature importance.

    Args:
        model: Trained model (sklearn estimator)
        X: Feature data
        background_samples: Number of background samples (default: 100)

    Returns:
        shap.Explanation: SHAP values object

    Raises:
        ImportError: If shap package not installed

    Example:
        >>> import shap
        >>> shap_values = calculate_shap_values(model, X_test, background_samples=100)
        >>> shap.plots.waterfall(shap_values[0])  # Plot for first player
    """
    if not SHAP_AVAILABLE:
        raise ImportError(
            "SHAP package not installed. Install with: pip install shap"
        )

    # Sample background data
    if len(X) > background_samples:
        background = X.sample(n=background_samples, random_state=42)
    else:
        background = X

    # Create explainer
    try:
        # Try TreeExplainer first (faster for tree-based models)
        explainer = shap.TreeExplainer(model, background)
    except Exception:
        # Fallback to KernelExplainer (slower but works for any model)
        explainer = shap.KernelExplainer(model.predict, background)

    # Calculate SHAP values
    shap_values = explainer(X)

    return shap_values


def find_outliers(
    residuals: np.ndarray,
    threshold: float = 2.0
) -> np.ndarray:
    """
    Find outlier predictions.

    Args:
        residuals: Prediction residuals
        threshold: Standard deviations for outlier (default: 2.0)

    Returns:
        np.ndarray: Boolean mask of outliers

    Example:
        >>> outliers = find_outliers(residuals, threshold=2.5)
        >>> outlier_names = df[outliers]['Name']
    """
    # Calculate mean and std
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    # Find outliers (beyond threshold standard deviations)
    outlier_mask = np.abs(residuals - mean_residual) > (threshold * std_residual)

    return outlier_mask


def compare_models(
    model_dict: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models on same test set.

    Args:
        model_dict: {model_name: model}
        X_test: Test features
        y_test: Test target

    Returns:
        pd.DataFrame: Comparison table with metrics

    Example:
        >>> models = {
        ...     'RandomForest': rf_model,
        ...     'Keras': keras_model,
        ...     'XGBoost': xgb_model
        ... }
        >>> comparison = compare_models(models, X_test, y_test)
        >>> # Returns:
        >>> #    Model         MAE   RMSE    R²
        >>> # 0  RandomForest  0.52  0.71  0.83
        >>> # 1  Keras         0.48  0.68  0.85
        >>> # 2  XGBoost       0.50  0.69  0.84
    """
    results = []

    for model_name, model in model_dict.items():
        # Make predictions
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"Warning: Failed to predict with {model_name}: {e}")
            continue

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })

    # Convert to DataFrame and sort by MAE (lower is better)
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('MAE')

    return comparison_df.reset_index(drop=True)
