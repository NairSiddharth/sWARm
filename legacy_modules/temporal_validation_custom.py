"""
Custom Temporal Cross-Validation Implementation
=============================================

This module contains the custom temporal validation logic that was originally
developed before switching to sklearn's TimeSeriesSplit. Preserved for
learning purposes and comparison with standard implementation.

Key Features:
- Manual temporal splitting with expanding windows
- Custom fold generation logic
- Tier-specific performance analysis
- Elite player protection factor optimization

Educational Value:
- Shows how to implement temporal CV from scratch
- Demonstrates the logic behind preventing data leakage
- Provides insight into custom ML validation approaches
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

class CustomTemporalCrossValidator:
    """
    Custom implementation of temporal cross-validation for time series data.

    This was the original approach before switching to sklearn TimeSeriesSplit.
    Demonstrates manual implementation of temporal validation concepts.
    """

    def __init__(self, years_range=(2016, 2024)):
        self.training_start = years_range[0]
        self.validation_end = years_range[1]
        self.data_years = list(range(years_range[0], years_range[1] + 1))

    def rolling_validation(self, min_train_years=3, forecast_horizon=1):
        """
        Custom rolling temporal cross-validation implementation.

        This manually creates expanding window splits:
        2016-2018 → predict 2019
        2016-2019 → predict 2020
        2016-2020 → predict 2021
        2016-2021 → predict 2022
        2016-2022 → predict 2023
        2016-2023 → predict 2024

        Args:
            min_train_years: Minimum training years before validation starts
            forecast_horizon: Years ahead to predict (typically 1)

        Returns:
            List of validation results for each fold
        """
        validation_results = []

        for target_year in range(self.training_start + min_train_years, self.validation_end + 1):
            train_years = list(range(self.training_start, target_year))

            if len(train_years) >= min_train_years:
                # Train both allocation methods
                uniform_model = self.train_uniform_allocation(train_years)
                dynamic_model = self.train_dynamic_allocation(train_years)

                # Test on target year
                uniform_results = uniform_model.predict(target_year)
                dynamic_results = dynamic_model.predict(target_year)

                # Measure tier-specific accuracy
                accuracy_comparison = self.measure_tier_accuracy(
                    uniform_results, dynamic_results, target_year
                )

                validation_results.append({
                    'target_year': target_year,
                    'train_years': train_years,
                    'uniform_accuracy': accuracy_comparison['uniform'],
                    'dynamic_accuracy': accuracy_comparison['dynamic'],
                    'tier_breakdown': accuracy_comparison['by_tier']
                })

        return validation_results

    def expanding_window_validation(self):
        """
        Custom expanding window validation for protection factor optimization.

        Tests multiple protection factor combinations across temporal folds:
        - Conservative: Small elite protection
        - Moderate: Balanced protection
        - Aggressive: High elite protection

        Returns:
            Optimal protection factors based on validation performance
        """
        protection_variants = {
            'conservative': {'superstar': 1.05, 'elite': 1.03, 'average': 1.0, 'below': 0.97},
            'moderate': {'superstar': 1.10, 'elite': 1.05, 'average': 1.0, 'below': 0.95},
            'aggressive': {'superstar': 1.20, 'elite': 1.10, 'average': 1.0, 'below': 0.90}
        }

        results = {}
        for variant_name, factors in protection_variants.items():
            variant_results = self.rolling_validation_with_factors(factors)
            results[variant_name] = variant_results

        return self.select_optimal_factors(results)

    def measure_tier_accuracy(self, uniform_pred, dynamic_pred, target_year):
        """
        Custom tier-specific accuracy measurement.

        Compares uniform vs dynamic allocation performance across
        elite, average, and below-average player tiers.

        Args:
            uniform_pred: Uniform allocation predictions
            dynamic_pred: Dynamic allocation predictions
            target_year: Year being validated

        Returns:
            Dictionary with tier-specific accuracy metrics
        """
        actual_data = self.load_actual_performance(target_year)

        # Define performance tiers based on actual performance
        elite_mask = actual_data['WAR'] >= 4.0
        average_mask = (actual_data['WAR'] >= 1.0) & (actual_data['WAR'] < 4.0)
        below_mask = actual_data['WAR'] < 1.0

        results = {
            'uniform': {
                'elite_rmse': self.calculate_rmse(uniform_pred[elite_mask], actual_data[elite_mask]),
                'average_rmse': self.calculate_rmse(uniform_pred[average_mask], actual_data[average_mask]),
                'below_rmse': self.calculate_rmse(uniform_pred[below_mask], actual_data[below_mask]),
                'overall_rmse': self.calculate_rmse(uniform_pred, actual_data)
            },
            'dynamic': {
                'elite_rmse': self.calculate_rmse(dynamic_pred[elite_mask], actual_data[elite_mask]),
                'average_rmse': self.calculate_rmse(dynamic_pred[average_mask], actual_data[average_mask]),
                'below_rmse': self.calculate_rmse(dynamic_pred[below_mask], actual_data[below_mask]),
                'overall_rmse': self.calculate_rmse(dynamic_pred, actual_data)
            }
        }

        # Calculate improvement metrics
        results['improvement'] = {
            'elite_improvement': results['uniform']['elite_rmse'] - results['dynamic']['elite_rmse'],
            'average_degradation': results['dynamic']['average_rmse'] - results['uniform']['average_rmse'],
            'below_degradation': results['dynamic']['below_rmse'] - results['uniform']['below_rmse'],
            'net_improvement': results['uniform']['overall_rmse'] - results['dynamic']['overall_rmse']
        }

        return results

    def create_temporal_splits(self, data_years, n_splits=5, min_train_years=3):
        """
        Create temporal splits manually (before sklearn approach).

        This demonstrates the manual logic for creating temporal folds
        that respect time ordering and prevent data leakage.

        Args:
            data_years: List of available years
            n_splits: Number of validation folds
            min_train_years: Minimum training years

        Returns:
            List of (train_years, test_year) tuples
        """
        splits = []
        total_years = len(data_years)

        # Calculate split points
        for i in range(n_splits):
            # Expanding window: training set grows with each fold
            test_year_idx = min_train_years + i

            if test_year_idx < total_years:
                train_years = data_years[:test_year_idx]
                test_year = data_years[test_year_idx]

                splits.append((train_years, test_year))

        return splits

    def validate_temporal_constraints(self, splits):
        """
        Validate that temporal constraints are satisfied.

        Ensures no data leakage by checking that all training
        years come before test years.

        Args:
            splits: List of (train_years, test_year) tuples

        Returns:
            Boolean indicating if constraints are satisfied
        """
        for train_years, test_year in splits:
            # Check that all training years come before test year
            if any(train_year >= test_year for train_year in train_years):
                return False

            # Check minimum training size
            if len(train_years) < 3:
                return False

        return True

    # Placeholder methods for completeness
    def train_uniform_allocation(self, train_years):
        """Placeholder for uniform allocation training."""
        return MockModel("uniform")

    def train_dynamic_allocation(self, train_years):
        """Placeholder for dynamic allocation training."""
        return MockModel("dynamic")

    def load_actual_performance(self, year):
        """Placeholder for loading actual performance data."""
        return pd.DataFrame({'WAR': np.random.normal(2, 1, 100)})

    def calculate_rmse(self, predictions, actuals):
        """Calculate root mean squared error."""
        return np.sqrt(np.mean((predictions - actuals) ** 2))

    def rolling_validation_with_factors(self, factors):
        """Placeholder for rolling validation with specific factors."""
        return {'mean_rmse': np.random.uniform(0.8, 1.2)}

    def select_optimal_factors(self, results):
        """Placeholder for optimal factor selection."""
        return min(results.keys(), key=lambda k: results[k]['mean_rmse'])


class MockModel:
    """Mock model for demonstration purposes."""

    def __init__(self, model_type):
        self.model_type = model_type

    def predict(self, year):
        """Generate mock predictions."""
        return np.random.normal(2, 1, 100)


# Example usage demonstrating the custom approach
if __name__ == "__main__":
    print("Custom Temporal Cross-Validation Example")
    print("=" * 45)

    # Initialize custom validator
    validator = CustomTemporalCrossValidator(years_range=(2016, 2024))

    # Create temporal splits manually
    data_years = list(range(2016, 2025))
    splits = validator.create_temporal_splits(data_years, n_splits=5)

    print("Manual temporal splits:")
    for i, (train_years, test_year) in enumerate(splits, 1):
        print(f"  Fold {i}: Train {train_years[0]}-{train_years[-1]} -> Test {test_year}")

    # Validate constraints
    constraints_satisfied = validator.validate_temporal_constraints(splits)
    print(f"\nTemporal constraints satisfied: {constraints_satisfied}")

    print("\nThis custom implementation demonstrates the same concepts")
    print("that sklearn.model_selection.TimeSeriesSplit provides in a")
    print("standard, well-tested package.")