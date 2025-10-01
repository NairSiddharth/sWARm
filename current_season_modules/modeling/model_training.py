"""Model training and prediction functionality.

This module handles the training of various machine learning models,
cross-validation, and predictions for current season data.
"""

from typing import Dict, Optional, Any, Tuple

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
import pandas as pd

from common_modules.logging import get_logger
from .cross_validation import CrossValidationResults
from .data_preparation import prepare_data_for_kfold

logger = get_logger(__name__)

# Check for optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available. XGBoost models will be skipped.")

try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Input
    from keras.callbacks import EarlyStopping
    from keras.optimizers import AdamW
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow/Keras not available. Neural network models will be skipped.")


def create_keras_model_temp(input_dim: int, name: str = "model") -> Any:
    """Create a temporary Keras model for K-fold cross-validation.

    Args:
        input_dim: Number of input features
        name: Name identifier for the model

    Returns:
        Compiled Keras model

    Raises:
        ImportError: If TensorFlow/Keras is not available
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow/Keras not available")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    logger.debug(f"Created Keras model '{name}' with input_dim={input_dim}")
    return model


def run_kfold_cross_validation(
    hitter_data: Optional[Dict],
    pitcher_data: Optional[Dict],
    n_splits: int = 5
) -> CrossValidationResults:
    """Run K-fold cross-validation on all models and datasets.

    Args:
        hitter_data: Dictionary with hitter features and targets
        pitcher_data: Dictionary with pitcher features and targets
        n_splits: Number of folds for cross-validation

    Returns:
        CrossValidationResults object containing all predictions
    """
    logger.info(f"Running {n_splits}-fold cross-validation...")

    results = CrossValidationResults()

    # Models to test
    models = {
        'ridge': Ridge(),
        'randomforest': RandomForestRegressor(n_estimators=100, random_state=42),
        'svr': SVR(),
    }

    # Add XGBoost if available
    if HAS_XGBOOST:
        models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)

    # Add Keras as a special case
    if HAS_TENSORFLOW:
        models['keras'] = 'neural_network'  # Special marker

    def run_cv_for_dataset(
        data: Dict,
        player_type: str,
        metric_type: str
    ) -> None:
        """Run cross-validation for a specific dataset."""
        if data is None:
            logger.info(f"Skipping {player_type} {metric_type} - no data available")
            return

        X = data['X']
        y = data['y']
        names = data['names']
        years = data['years']

        logger.info(f"Running CV for {player_type} {metric_type}: {len(X)} samples")

        # Use GroupKFold to keep years together
        gkf = GroupKFold(n_splits=n_splits)

        # Store predictions for each model
        for model_name, model in models.items():
            y_pred_all = np.zeros(len(y))

            try:
                for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=years)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # Handle Keras separately
                    if model_name == 'keras' and HAS_TENSORFLOW:
                        # Create and train Keras model
                        keras_model = create_keras_model_temp(
                            X_train.shape[1],
                            f"{player_type}_{metric_type}",
                        )
                        keras_model.fit(
                            X_train, y_train,
                            epochs=50, batch_size=32,
                            validation_split=0.2, verbose=0,
                        )
                        y_pred_fold = keras_model.predict(X_test, verbose=0).flatten()
                    elif model_name != 'keras':
                        # Standard sklearn models
                        model_copy = clone(model)
                        model_copy.fit(X_train, y_train)
                        y_pred_fold = model_copy.predict(X_test)
                    else:
                        continue  # Skip Keras if TensorFlow not available

                    y_pred_all[test_idx] = y_pred_fold

                # Store results
                if model_name == 'keras' and not HAS_TENSORFLOW:
                    continue

                results.store_cv_results(
                    model_name, player_type, metric_type,
                    y.values, y_pred_all, names.values, years,
                )

                # Calculate and log metrics
                r2 = r2_score(y, y_pred_all)
                rmse = np.sqrt(mean_squared_error(y, y_pred_all))
                logger.info(
                    f"  {model_name} {player_type} {metric_type}: "
                    f"R2={r2:.4f}, RMSE={rmse:.4f}",
                )

            except Exception as e:
                logger.error(
                    f"Error training {model_name} for {player_type} {metric_type}: {e}",
                )
                continue

    # Run CV for all datasets
    if hitter_data:
        if 'warp' in hitter_data:
            run_cv_for_dataset(hitter_data['warp'], 'hitter', 'warp')
        if 'war' in hitter_data:
            run_cv_for_dataset(hitter_data['war'], 'hitter', 'war')

    if pitcher_data:
        if 'warp' in pitcher_data:
            run_cv_for_dataset(pitcher_data['warp'], 'pitcher', 'warp')
        if 'war' in pitcher_data:
            run_cv_for_dataset(pitcher_data['war'], 'pitcher', 'war')

    logger.info("K-fold cross-validation complete!")
    return results


class CurrentSeasonPredictor:
    """Wrapper class for current season predictions using restored multi-source pipeline.

    This class maintains compatibility with WARPCalculator and other modules,
    providing a unified interface for training and prediction.
    """

    def __init__(self):
        """Initialize the predictor with empty model storage."""
        self.models = {}
        self.is_trained = False
        self.results = None
        logger.debug("Initialized CurrentSeasonPredictor")

    def train_ensemble_models(self, holdout_year: int = 2024) -> CrossValidationResults:
        """Train ensemble models using the pipeline.

        Args:
            holdout_year: Year to hold out for validation

        Returns:
            CrossValidationResults object with training results
        """
        logger.info("Training ensemble models using multi-year pipeline...")

        # prepare data and run cross-validation
        hitter_data, pitcher_data = prepare_data_for_kfold()

        if not hitter_data and not pitcher_data:
            logger.error("No data available for training")
            raise ValueError("No data available for training")

        results = run_kfold_cross_validation(hitter_data, pitcher_data, n_splits=5)

        self.results = results
        self.models = self._extract_models_from_results(results)
        self.is_trained = True

        logger.info("Ensemble model training complete")
        return results

    def _extract_models_from_results(
        self,
        results: CrossValidationResults
    ) -> Dict[str, Any]:
        """Extract trained models from cross-validation results.

        Args:
            results: CrossValidationResults object

        Returns:
            Dictionary of trained models
        """
        # This is a placeholder - actual implementation would need to
        # retrain final models on full dataset
        models = {}
        for key in results.results.keys():
            models[key] = None  # Placeholder for actual models
        return models

    def predict_current_season(
        self,
        player_data: pd.DataFrame,
        player_type: str = 'hitter'
    ) -> pd.DataFrame:
        """Make predictions for current season player data.

        Args:
            player_data: DataFrame with player statistics
            player_type: 'hitter' or 'pitcher'

        Returns:
            DataFrame with predictions

        Raises:
            ValueError: If models haven't been trained
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")

        logger.info(f"Making predictions for {len(player_data)} {player_type}s")

        # Prepare features based on player type
        if player_type == 'hitter':
            feature_cols = [
                'K%', 'BB%', 'AVG', 'OBP', 'SLG', 'PA',
                'Positional_WAR', 'GDP_rate',
                'Enhanced_Baserunning', 'Enhanced_Defense',
            ]
        else:
            feature_cols = [
                'IP', 'BB%', 'K%', 'K-BB%', 'ERA',
                'damage_control_ratio', 'Opportunity_Success',
                'Contact_Quality_Index', 'HBP%',
                'Statcast_Launch_Quality_Index',
            ]

        # Filter to available columns
        available_features = [col for col in feature_cols if col in player_data.columns]

        if len(available_features) == 0:
            logger.error("No valid features found in player data")
            raise ValueError("No valid features found in player data")

        # This is a placeholder - actual implementation would use trained models
        predictions = pd.DataFrame({
            'player_id': player_data.index,
            'predicted_war': np.zeros(len(player_data)),
            'predicted_warp': np.zeros(len(player_data))
        })

        logger.info("Predictions complete")
        return predictions

    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all trained models.

        Returns:
            Dictionary of model performance metrics

        Raises:
            ValueError: If models haven't been trained
        """
        if not self.is_trained or not self.results:
            raise ValueError("Models must be trained before getting performance")

        performance = {}
        for key, data in self.results.results.items():
            y_true = data['y_true']
            y_pred = data['y_pred']

            performance[key] = {
                'r2': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'n_samples': len(y_true),
            }

        return performance
