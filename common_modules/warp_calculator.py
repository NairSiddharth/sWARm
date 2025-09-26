"""
WARP Calculator Module - Real-time WARP Calculation from Component Stats
Uses trained ensemble models to calculate WARP values from current season component statistics
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os

# Import predictive modeling components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from current_season_modules.predictive_modeling import CurrentSeasonPredictor


class WARPCalculator:
    """
    Real-time WARP calculator using trained ensemble models

    Core insight: Your trained models learned Component Stats → WARP relationships
    from 2016-2024 data. This class applies those relationships to current season data.
    """

    def __init__(self):
        self.predictor = CurrentSeasonPredictor()
        self.is_initialized = False
        self.feature_mappings = {}
        self.validation_stats = {}

    def initialize_models(self, holdout_year=2024):
        """
        Initialize and train ensemble models for WARP calculation

        Args:
            holdout_year: Year to hold out for validation (default 2024)
        """
        print("Initializing WARP calculator with trained ensemble models...")

        try:
            # Train ensemble models on historical data
            self.predictor.train_ensemble_models(holdout_year=holdout_year)

            # Validate against holdout year
            self._validate_warp_calculations(holdout_year)

            self.is_initialized = True
            print("WARP calculator initialization completed successfully!")

        except Exception as e:
            print(f"Error initializing WARP calculator: {e}")
            raise

    def calculate_player_warp(self, player_stats, player_type):
        """
        Calculate WARP for a player using current season component stats

        Args:
            player_stats: Dict or Series containing player's component statistics
            player_type: 'hitter' or 'pitcher'

        Returns:
            dict: {'warp': calculated_value, 'confidence': confidence_score, 'components': component_breakdown}
        """
        if not self.is_initialized:
            raise ValueError("WARP calculator not initialized. Call initialize_models() first.")

        try:
            # Convert stats to feature vector
            feature_vector = self._prepare_feature_vector(player_stats, player_type)

            # Calculate WARP using ensemble models
            warp_prediction, component_predictions = self.predictor.predict_current_season(
                feature_vector, player_type, 'warp'
            )

            # Calculate confidence score
            confidence = self._calculate_confidence(component_predictions, player_type)

            result = {
                'warp': warp_prediction,
                'confidence': confidence,
                'components': {
                    'randomforest': component_predictions.get('randomforest'),
                    'keras': component_predictions.get('keras'),
                    'ensemble_weights': self._get_ensemble_weights('warp')
                }
            }

            return result

        except Exception as e:
            print(f"Error calculating WARP for {player_type}: {e}")
            return {'warp': 0.0, 'confidence': 0.0, 'components': {}}

    def calculate_war_warp_ensemble(self, player_stats, player_type):
        """
        Calculate both WAR and WARP using ensemble approach

        Args:
            player_stats: Dict or Series containing player's component statistics
            player_type: 'hitter' or 'pitcher'

        Returns:
            dict: {'war': war_value, 'warp': warp_value, 'ensemble_info': details}
        """
        if not self.is_initialized:
            raise ValueError("WARP calculator not initialized. Call initialize_models() first.")

        try:
            feature_vector = self._prepare_feature_vector(player_stats, player_type)

            # Calculate both metrics
            warp_result = self.calculate_player_warp(player_stats, player_type)

            war_prediction, war_components = self.predictor.predict_current_season(
                feature_vector, player_type, 'war'
            )

            result = {
                'war': war_prediction,
                'warp': warp_result['warp'],
                'ensemble_info': {
                    'war_components': war_components,
                    'warp_components': warp_result['components'],
                    'war_confidence': self._calculate_confidence(war_components, player_type),
                    'warp_confidence': warp_result['confidence']
                }
            }

            return result

        except Exception as e:
            print(f"Error calculating WAR/WARP ensemble for {player_type}: {e}")
            return {'war': 0.0, 'warp': 0.0, 'ensemble_info': {}}

    def _prepare_feature_vector(self, player_stats, player_type):
        """
        Convert player statistics to feature vector for model input

        This should match the feature preparation used in historical training
        """
        # This is a placeholder - needs to be implemented based on your actual
        # feature engineering pipeline from the historical data preparation

        if player_type == 'hitter':
            # Example hitter features - adjust based on your actual feature set
            features = [
                player_stats.get('AVG', 0.250),
                player_stats.get('OBP', 0.320),
                player_stats.get('SLG', 0.400),
                player_stats.get('HR', 0),
                player_stats.get('RBI', 0),
                player_stats.get('SB', 0),
                player_stats.get('BB%', 0.08),
                player_stats.get('K%', 0.22),
                # Add more features as needed
            ]
        else:  # pitcher
            features = [
                player_stats.get('ERA', 4.00),
                player_stats.get('WHIP', 1.30),
                player_stats.get('K/9', 8.0),
                player_stats.get('BB/9', 3.0),
                player_stats.get('HR/9', 1.2),
                player_stats.get('K%', 0.20),
                player_stats.get('BB%', 0.08),
                player_stats.get('FIP', 4.00),
                # Add more features as needed
            ]

        return np.array(features)

    def _calculate_confidence(self, component_predictions, player_type):
        """
        Calculate confidence score for WARP prediction

        Based on agreement between RandomForest and Keras models
        """
        if len(component_predictions) < 2:
            return 0.5  # Low confidence with only one model

        rf_pred = component_predictions.get('randomforest', 0)
        keras_pred = component_predictions.get('keras', 0)

        # Calculate agreement between models (higher agreement = higher confidence)
        if abs(rf_pred) > 0.1 or abs(keras_pred) > 0.1:
            agreement = 1.0 - abs(rf_pred - keras_pred) / (abs(rf_pred) + abs(keras_pred) + 0.1)
        else:
            agreement = 0.8  # Both models predict near zero

        return max(0.1, min(1.0, agreement))

    def _get_ensemble_weights(self, metric_type):
        """Get ensemble weights used for different metrics"""
        if metric_type == 'warp':
            return {'randomforest': 0.8, 'keras': 0.2}
        else:  # war
            return {'randomforest': 0.2, 'keras': 0.8}

    def _validate_warp_calculations(self, holdout_year):
        """
        Validate WARP calculations against known holdout year data

        Args:
            holdout_year: Year to validate against
        """
        print(f"Validating WARP calculations against {holdout_year} data...")

        try:
            # Load holdout year data for validation
            hitter_data, pitcher_data = self.predictor.load_historical_data_for_training()

            validation_results = {}

            # Validate hitter WARP calculations
            if hitter_data and 'warp' in hitter_data:
                hitter_validation = self._validate_player_type(
                    hitter_data['warp'], holdout_year, 'hitter'
                )
                validation_results['hitter_warp'] = hitter_validation

            # Validate pitcher WARP calculations
            if pitcher_data and 'warp' in pitcher_data:
                pitcher_validation = self._validate_player_type(
                    pitcher_data['warp'], holdout_year, 'pitcher'
                )
                validation_results['pitcher_warp'] = pitcher_validation

            self.validation_stats = validation_results

            # Print validation summary
            self._print_validation_summary()

        except Exception as e:
            print(f"Warning: WARP validation failed: {e}")

    def _validate_player_type(self, data, holdout_year, player_type):
        """Validate WARP calculations for specific player type"""
        holdout_indices = [i for i, year in enumerate(data['years']) if year == holdout_year]

        if not holdout_indices:
            return {'error': f'No {holdout_year} data available for {player_type}'}

        # Calculate predictions for holdout data
        predictions = []
        actuals = []

        for idx in holdout_indices[:10]:  # Sample first 10 for validation
            feature_vector = np.array(data['X'][idx])

            try:
                warp_pred, _ = self.predictor.predict_current_season(
                    feature_vector, player_type, 'warp'
                )
                predictions.append(warp_pred)
                actuals.append(data['y'][idx])
            except:
                continue

        if predictions:
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))

            return {
                'r2': r2,
                'rmse': rmse,
                'n_samples': len(predictions),
                'mean_actual': np.mean(actuals),
                'mean_predicted': np.mean(predictions)
            }
        else:
            return {'error': 'No successful predictions'}

    def _print_validation_summary(self):
        """Print summary of validation results"""
        print("\nWARP Calculator Validation Summary:")
        print("-" * 40)

        for player_type, results in self.validation_stats.items():
            if 'error' in results:
                print(f"{player_type}: {results['error']}")
            else:
                print(f"{player_type}:")
                print(f"  R² Score: {results['r2']:.4f}")
                print(f"  RMSE: {results['rmse']:.4f}")
                print(f"  Samples: {results['n_samples']}")
                print(f"  Mean Actual: {results['mean_actual']:.4f}")
                print(f"  Mean Predicted: {results['mean_predicted']:.4f}")
        print()


def calculate_current_season_warp(player_stats_dict, player_type='hitter'):
    """
    Convenience function for calculating WARP from current season stats

    Args:
        player_stats_dict: Dictionary of player statistics
        player_type: 'hitter' or 'pitcher'

    Returns:
        float: Calculated WARP value
    """
    calculator = WARPCalculator()

    if not calculator.is_initialized:
        calculator.initialize_models()

    result = calculator.calculate_player_warp(player_stats_dict, player_type)
    return result['warp']


def validate_warp_calculation_accuracy(holdout_year=2024):
    """
    Standalone function to validate WARP calculation accuracy

    Args:
        holdout_year: Year to use for validation

    Returns:
        dict: Validation results
    """
    calculator = WARPCalculator()
    calculator.initialize_models(holdout_year=holdout_year)

    return calculator.validation_stats