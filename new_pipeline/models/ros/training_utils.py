"""
Training utilities for ROS models.

Handles data preparation, cross-validation, and model evaluation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_ros_training_data(
    multipoint_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'remaining_WAR'
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare training data for ROS models from multipoint split format.

    Expects data from create_multipoint_splits() with current stats + remaining_WAR target.
    Much simpler than old version - data is already in correct format!

    Args:
        multipoint_df: Output from create_multipoint_splits() with columns:
            - playerid, Year, split_point, season_completion_pct
            - current_[stat]: Stats through split point
            - remaining_WAR: Target (actual WAR produced after split point)
            - All ROS feature columns
        feature_columns: List of ROS feature column names (e.g., ROS_HITTER_FEATURES)
        target_column: Target column name (default: 'remaining_WAR')

    Returns:
        Tuple of (training_df, X_features, y_target) where:
        - training_df: Complete DataFrame (for time series conversion if needed)
        - X_features: Feature matrix (n_samples, n_features)
        - y_target: Remaining WAR targets (n_samples,)

    Example:
        >>> # Create multipoint splits (2016-2024, 3 splits per season = 3x data)
        >>> splits = create_multipoint_splits(full_season_2016_2024, [0.25, 0.5, 0.75], 'hitter')
        >>> train_df, X, y = prepare_ros_training_data(splits, ROS_HITTER_FEATURES)
        >>> X.shape
        (4500, 87)  # 1500 player-seasons * 3 splits, 87 features
        >>> y.shape
        (4500,)  # Remaining WAR targets

    Note:
        The multipoint format provides 3x more training data than fixed firsthalf/secondhalf
        splits, and teaches the model about season timing (early-season SSS vs late-season).
    """
    # Validate required columns
    required_cols = ['playerid', 'Year', 'split_point', 'season_completion_pct', target_column]
    missing = [col for col in required_cols if col not in multipoint_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Did you run create_multipoint_splits()?")

    # Filter to rows with valid target
    training_df = multipoint_df[multipoint_df[target_column].notna()].copy()

    if len(training_df) == 0:
        raise ValueError(f"No valid training samples (all {target_column} are NaN)")

    # Extract features (only use features that exist)
    available_features = [col for col in feature_columns if col in training_df.columns]

    if len(available_features) == 0:
        raise ValueError(f"No feature columns found in data. Expected columns from ROS_*_FEATURES")

    missing_features = set(feature_columns) - set(available_features)
    if missing_features:
        print(f"Warning: {len(missing_features)} features not found in data: {list(missing_features)[:5]}...")

    X = training_df[available_features].values
    y = training_df[target_column].values

    return training_df, X, y


def temporal_cv_split(
    X: np.ndarray,
    y: np.ndarray,
    years: Optional[np.ndarray] = None,
    n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create temporal cross-validation splits.

    Uses TimeSeriesSplit to respect temporal ordering.
    Training sets grow progressively, test sets are consecutive periods.

    Args:
        X: Feature matrix
        y: Target values
        years: Year for each sample (optional, for logging)
        n_splits: Number of CV splits

    Returns:
        List of (train_idx, test_idx) tuples

    Example:
        >>> years = np.array([2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023])
        >>> splits = temporal_cv_split(X, y, years, n_splits=3)
        >>> # Split 1: train on indices 0-3, test on indices 4-5
        >>> # Split 2: train on indices 0-5, test on indices 6-7
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    splits = list(tscv.split(X))

    # Optionally log split years for debugging
    if years is not None and len(years) == len(X):
        for i, (train_idx, test_idx) in enumerate(splits):
            train_years = np.unique(years[train_idx])
            test_years = np.unique(years[test_idx])
            # Could print or log here if needed
            # print(f"Split {i+1}: Train {train_years}, Test {test_years}")

    return splits


def calculate_ros_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile_preds: Optional[Dict[float, np.ndarray]] = None,
    elite_threshold: float = 4.0
) -> Dict[str, float]:
    """
    Calculate ROS prediction metrics.

    Args:
        y_true: Actual secondhalf WAR
        y_pred: Predicted secondhalf WAR
        quantile_preds: Optional quantile predictions for calibration
            Format: {0.1: array([...]), 0.5: array([...]), 0.9: array([...])}
        elite_threshold: WAR threshold for elite players (default: 4.0)

    Returns:
        Dictionary of metrics:
        - MAE: Overall mean absolute error
        - RMSE: Root mean squared error
        - R2: Coefficient of determination
        - Bias: Mean error (overprediction/underprediction)
        - Elite_MAE: MAE for elite players
        - Elite_Bias: Bias for elite players
        - Elite_Count: Number of elite players
        - Q10_Coverage, Q90_Coverage: Quantile calibration (if provided)

    Example:
        >>> metrics = calculate_ros_metrics(y_test, y_pred, quantile_preds)
        >>> metrics['MAE']
        0.85
        >>> metrics['Elite_MAE']
        1.12
        >>> metrics['Bias']
        -0.15  # Slight underprediction
    """
    # Overall metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Bias (mean error)
    bias = (y_pred - y_true).mean()

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Bias': bias
    }

    # Elite subset (WAR > threshold)
    elite_mask = y_true > elite_threshold
    if elite_mask.sum() > 0:
        elite_mae = mean_absolute_error(y_true[elite_mask], y_pred[elite_mask])
        elite_bias = (y_pred[elite_mask] - y_true[elite_mask]).mean()
        metrics['Elite_MAE'] = elite_mae
        metrics['Elite_Bias'] = elite_bias
        metrics['Elite_Count'] = elite_mask.sum()
    else:
        metrics['Elite_MAE'] = np.nan
        metrics['Elite_Bias'] = np.nan
        metrics['Elite_Count'] = 0

    # Quantile calibration
    if quantile_preds is not None:
        for q, preds_q in quantile_preds.items():
            # Coverage: proportion of actuals <= predicted quantile
            # For q=0.9, we expect 90% coverage
            coverage = (y_true <= preds_q).mean()
            metrics[f'Q{int(q*100)}_Coverage'] = coverage

    return metrics


def calculate_component_metrics(
    y_true: np.ndarray,
    component_predictions: Dict[str, np.ndarray],
    component_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate metrics for each ensemble component.

    Useful for comparing component performance and optimizing weights.

    Args:
        y_true: Actual secondhalf WAR
        component_predictions: Dict mapping component name to predictions
            Example: {
                'baseline': array([...]),
                'direct': array([...]),
                'temporal': array([...])
            }
        component_names: Optional list to specify order

    Returns:
        DataFrame with metrics per component

    Example:
        >>> comp_metrics = calculate_component_metrics(y_test, {
        ...     'baseline': baseline_pred,
        ...     'direct': direct_pred,
        ...     'temporal': temporal_pred
        ... })
        >>> comp_metrics
                     MAE  RMSE    R2   Bias
        baseline    0.92  1.25  0.31  -0.10
        direct      0.85  1.18  0.35  -0.05
        temporal    0.95  1.30  0.28  -0.15
    """
    if component_names is None:
        component_names = list(component_predictions.keys())

    results = []
    for name in component_names:
        if name not in component_predictions:
            continue

        y_pred = component_predictions[name]

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        bias = (y_pred - y_true).mean()

        results.append({
            'Component': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Bias': bias
        })

    return pd.DataFrame(results).set_index('Component')


def optimize_ensemble_weights(
    y_true: np.ndarray,
    component_predictions: Dict[str, np.ndarray],
    metric: str = 'mae',
    method: str = 'grid'
) -> Dict[str, float]:
    """
    Optimize ensemble weights on validation set.

    Args:
        y_true: Actual values
        component_predictions: Dict of component predictions
        metric: Metric to optimize ('mae', 'rmse', or 'r2')
        method: Optimization method ('grid' or 'scipy')

    Returns:
        Dict of optimal weights (sum to 1.0)

    Example:
        >>> weights = optimize_ensemble_weights(y_val, {
        ...     'baseline': baseline_pred,
        ...     'direct': direct_pred,
        ...     'temporal': temporal_pred
        ... })
        >>> weights
        {'baseline': 0.15, 'direct': 0.50, 'temporal': 0.35}
    """
    component_names = list(component_predictions.keys())
    n_components = len(component_names)

    if method == 'grid':
        # Simple grid search
        best_metric = float('inf') if metric in ['mae', 'rmse'] else float('-inf')
        best_weights = None

        # Try different weight combinations (step size = 0.05)
        from itertools import product

        # Generate weight combinations that sum to 1.0
        step = 0.05
        weight_options = np.arange(0.0, 1.0 + step, step)

        for weights_tuple in product(weight_options, repeat=n_components):
            weights = np.array(weights_tuple)

            # Check if weights sum to ~1.0
            if not np.isclose(weights.sum(), 1.0):
                continue

            # Calculate ensemble prediction
            y_pred = sum(
                weights[i] * component_predictions[name]
                for i, name in enumerate(component_names)
            )

            # Calculate metric
            if metric == 'mae':
                score = mean_absolute_error(y_true, y_pred)
                is_better = score < best_metric
            elif metric == 'rmse':
                score = np.sqrt(mean_squared_error(y_true, y_pred))
                is_better = score < best_metric
            elif metric == 'r2':
                score = r2_score(y_true, y_pred)
                is_better = score > best_metric
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if is_better:
                best_metric = score
                best_weights = weights.copy()

        if best_weights is None:
            # Fallback to equal weights
            best_weights = np.ones(n_components) / n_components

        # Convert to dict
        return {name: float(best_weights[i]) for i, name in enumerate(component_names)}

    elif method == 'scipy':
        from scipy.optimize import minimize

        def objective(weights):
            # Calculate ensemble prediction
            y_pred = sum(
                weights[i] * component_predictions[name]
                for i, name in enumerate(component_names)
            )

            # Return metric to minimize (or negative for maximization)
            if metric == 'mae':
                return mean_absolute_error(y_true, y_pred)
            elif metric == 'rmse':
                return np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == 'r2':
                return -r2_score(y_true, y_pred)  # Negative for minimization
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Constraints: weights sum to 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
        bounds = [(0.0, 1.0) for _ in range(n_components)]

        # Initial guess: equal weights
        x0 = np.ones(n_components) / n_components

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            # Fallback to equal weights
            result.x = np.ones(n_components) / n_components

        # Convert to dict
        return {name: float(result.x[i]) for i, name in enumerate(component_names)}

    else:
        raise ValueError(f"Unknown method: {method}")
