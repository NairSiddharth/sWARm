"""Cross - validation results management for model evaluation.

This module provides functionality for storing and retrieving K-fold
cross-validation results with year-based tracking.
"""

from typing import Dict, List, Any, Optional

import numpy as np

from common_modules.logging import get_logger

logger = get_logger(__name__)


class CrossValidationResults:
    """Class to store K-fold cross-validation results with year information.

    This class manages the storage and retrieval of cross-validation predictions,
    allowing for year-specific analysis and tracking of model performance across
    different time periods.
    """

    def __init__(self):
        """Initialize empty results dictionary."""
        self.results = {}
        logger.debug("Initialized CrossValidationResults instance")

    def store_cv_results(
        self,
        model_name: str,
        player_type: str,
        metric_type: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        player_names: np.ndarray,
        years: np.ndarray,
    ) -> None:
        """Store cross-validation results with year information.

        Args:
            model_name: Name of the model used for prediction
            player_type: Type of player (e.g., 'hitter', 'pitcher')
            metric_type: Type of metric being predicted (e.g., 'war', 'warp')
            y_true: Array of true values
            y_pred: Array of predicted values
            player_names: Array of player names
            years: Array of years corresponding to each prediction
        """
        key = f"{model_name}_{player_type}_{metric_type}"
        self.results[key] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'player_names': player_names,
            'years': years,
        }
        logger.info(f"Stored CV results for {key} with {len(y_true)} predictions")

    def get_year_data(self, year: int) -> Dict[str, Dict[str, np.ndarray]]:
        """Get all predictions for a specific year.

        Args:
            year: The year to retrieve predictions for

        Returns:
            Dictionary of predictions for the specified year
        """
        year_data = {}
        for key, data in self.results.items():
            year_mask = np.array([str(y) == str(year) for y in data['years']])
            if np.any(year_mask):
                year_data[key] = {
                    'y_true': np.array(data['y_true'])[year_mask],
                    'y_pred': np.array(data['y_pred'])[year_mask],
                    'player_names': np.array(data['player_names'])[year_mask],
                    'years': np.array(data['years'])[year_mask],
                }

        if year_data:
            logger.debug(f"Retrieved data for year {year} with {len(year_data)} models")
        else:
            logger.warning(f"No data found for year {year}")

        return year_data

    def get_available_years(self) -> List[str]:
        """Get all years with predictions.

        Returns:
            Sorted list of years with available predictions
        """
        all_years = set()
        for data in self.results.values():
            all_years.update([str(y) for y in data['years']])

        sorted_years = sorted(list(all_years))
        logger.debug(f"Available years: {sorted_years}")
        return sorted_years

    def get_model_metrics(
        self,
        model_name: Optional[str] = None,
        player_type: Optional[str] = None,
        metric_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get metrics for specific model / player/metric combinations.

        Args:
            model_name: Filter by model name (optional)
            player_type: Filter by player type (optional)
            metric_type: Filter by metric type (optional)

        Returns:
            Dictionary of filtered results
        """
        filtered_results = {}

        for key, data in self.results.items():
            # parts = key.split('_')  # noqa: F841

            # Check if key matches filters
            if model_name and not key.startswith(model_name):
                continue
            if player_type and player_type not in key:
                continue
            if metric_type and not key.endswith(metric_type):
                continue

            filtered_results[key] = data

        logger.debug(f"Filtered to {len(filtered_results)} results")
        return filtered_results

    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results = {}
        logger.info("Cleared all cross-validation results")


def print_cv_summary(results: CrossValidationResults) -> None:
    """Print a summary of cross-validation results.

    Args:
        results: CrossValidationResults instance containing predictions
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    logger.info("Cross - Validation Summary:")
    print("\n" + "=" * 60)
    print("CROSS - VALIDATION RESULTS SUMMARY")
    print("=" * 60 + "\n")

    for key, data in results.results.items():
        y_true = data['y_true']
        y_pred = data['y_pred']

        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Print results
        model_name, player_type, metric_type = key.rsplit('_', 2)
        print(f"{model_name} - {player_type.capitalize()} {metric_type.upper()}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  Samples: {len(y_true)}")
        print()

        logger.debug(f"{key}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # Print year-wise summary
    available_years = results.get_available_years()
    if available_years:
        print(f"Years with predictions: {', '.join(available_years)}")
        print(f"Total years: {len(available_years)}")
        print("=" * 60 + "\n")
