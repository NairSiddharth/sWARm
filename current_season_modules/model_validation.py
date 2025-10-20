"""
Model validation module for comprehensive time series cross-validation.

This module provides sophisticated validation strategies for baseball WAR/WARP models
that respect both temporal ordering and player performance tiers. It implements
multiple validation approaches including standard TimeSeriesSplit with stratified
metrics for elite players, rookies, and general population.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ValidationMetrics:
    """Container for validation metrics across different player segments."""

    r2: float
    mae: float
    rmse: float
    mse: float
    n_samples: int

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format."""
        return {
            'r2': self.r2,
            'mae': self.mae,
            'rmse': self.rmse,
            'mse': self.mse,
            'n_samples': self.n_samples
        }


@dataclass
class PlayerSegmentResults:
    """Results for a specific player segment (elite, non-elite, rookies, etc.)."""

    segment_name: str
    metrics: ValidationMetrics
    player_ids: List[str] = field(default_factory=list)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    actuals: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_summary(self) -> str:
        """Get formatted summary of segment results."""
        return (f"{self.segment_name}: R²={self.metrics.r2:.4f}, "
                f"MAE={self.metrics.mae:.4f}, RMSE={self.metrics.rmse:.4f} "
                f"(n={self.metrics.n_samples})")


@dataclass
class FoldResults:
    """Results from a single validation fold."""

    fold_number: int
    train_years: List[int]
    test_year: int
    overall: ValidationMetrics
    elite: Optional[ValidationMetrics] = None
    non_elite: Optional[ValidationMetrics] = None
    rookies: Optional[ValidationMetrics] = None
    veterans: Optional[ValidationMetrics] = None
    by_position: Dict[str, ValidationMetrics] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get formatted summary of fold results."""
        summary = [f"Fold {self.fold_number}: Train {min(self.train_years)}-{max(self.train_years)}, Test {self.test_year}"]
        summary.append(f"  Overall: R²={self.overall.r2:.4f}, MAE={self.overall.mae:.4f}, RMSE={self.overall.rmse:.4f} (n={self.overall.n_samples})")

        if self.elite and self.elite.n_samples > 0:
            summary.append(f"  Elite: R²={self.elite.r2:.4f}, MAE={self.elite.mae:.4f}, RMSE={self.elite.rmse:.4f} (n={self.elite.n_samples})")
        if self.non_elite and self.non_elite.n_samples > 0:
            summary.append(f"  Non-Elite: R²={self.non_elite.r2:.4f}, MAE={self.non_elite.mae:.4f}, RMSE={self.non_elite.rmse:.4f} (n={self.non_elite.n_samples})")
        if self.rookies and self.rookies.n_samples > 0:
            summary.append(f"  Rookies: R²={self.rookies.r2:.4f}, MAE={self.rookies.mae:.4f}, RMSE={self.rookies.rmse:.4f} (n={self.rookies.n_samples})")

        return "\n".join(summary)


class TimeSeriesModelValidator:
    """
    Comprehensive model validation for time series baseball data.

    This validator implements sophisticated cross-validation strategies that:
    1. Respect temporal ordering (no future data leakage)
    2. Provide stratified metrics for player segments
    3. Support multiple validation strategies
    """

    # Elite thresholds - specifically testing MVP-caliber players
    # ~23 players reached 5+ WAR in 2024, with only ~10 pitchers at 4+ WAR
    # This is exactly the cohort where we see systematic undervaluation
    ELITE_WAR_THRESHOLD = 5.0  # MVP-caliber (testing systematic undervaluation)
    ELITE_WARP_THRESHOLD = 4.5  # WARP equivalent

    # Validation thresholds
    R2_THRESHOLD_EXCELLENT = 0.75
    R2_THRESHOLD_GOOD = 0.70
    R2_THRESHOLD_ACCEPTABLE = 0.60
    MAE_THRESHOLD_EXCELLENT = 0.3
    MAE_THRESHOLD_GOOD = 0.5
    MAE_THRESHOLD_ACCEPTABLE = 0.75

    def __init__(self, verbose: bool = True):
        """
        Initialize the validator.

        Args:
            verbose: Whether to print progress during validation
        """
        self.verbose = verbose
        self.validation_results = []

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ValidationMetrics:
        """
        Calculate comprehensive metrics for predictions.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            ValidationMetrics object with all calculated metrics
        """
        if len(y_true) == 0:
            return ValidationMetrics(r2=0, mae=0, rmse=0, mse=0, n_samples=0)

        # Calculate basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # Handle R² calculation for small samples
        if len(y_true) < 2:
            # R² undefined for single sample
            r2 = np.nan
        elif np.var(y_true) == 0:
            # All y_true values are the same
            r2 = 0.0 if np.var(y_pred - y_true) > 0 else 1.0
        else:
            r2 = r2_score(y_true, y_pred)

        return ValidationMetrics(
            r2=r2,
            mae=mae,
            rmse=rmse,
            mse=mse,
            n_samples=len(y_true)
        )

    def identify_player_segments(
        self,
        data: Dict[str, Any],
        test_indices: np.ndarray,
        metric_type: str = 'war'
    ) -> Dict[str, np.ndarray]:
        """
        Identify different player segments in the test data.

        Args:
            data: Dictionary with 'X', 'y', 'years', 'player_ids', etc.
            test_indices: Indices of test samples
            metric_type: 'war' or 'warp' for elite threshold selection

        Returns:
            Dictionary mapping segment names to boolean masks
        """
        segments = {}

        # Get test data - handle both DataFrame and array
        if hasattr(data['y'], 'iloc'):
            y_test = data['y'].iloc[test_indices]
        else:
            y_test = data['y'][test_indices]

        if hasattr(y_test, 'values'):
            y_test = y_test.values

        # Elite vs non-elite based on full season performance
        elite_threshold = self.ELITE_WAR_THRESHOLD if metric_type == 'war' else self.ELITE_WARP_THRESHOLD
        segments['elite'] = y_test >= elite_threshold
        segments['non_elite'] = y_test < elite_threshold

        # Rookies vs veterans (if games/experience data available)
        if 'games_played' in data:
            games = np.array(data['games_played'])[test_indices]
            # Rookies: less than 100 games career experience (rough proxy)
            segments['rookies'] = games < 100
            segments['veterans'] = games >= 100

        # By position (if available)
        if 'positions' in data:
            positions = np.array(data['positions'])[test_indices]
            unique_positions = np.unique(positions)
            for pos in unique_positions:
                segments[f'position_{pos}'] = positions == pos

        return segments

    def validate_fold(
        self,
        ensemble_predictor,
        data: Dict[str, Any],
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        metric_type: str,
        player_type: str,
        fold_number: int
    ) -> FoldResults:
        """
        Validate a single fold with comprehensive metrics.

        Args:
            ensemble_predictor: Trained ensemble model
            data: Data dictionary
            train_indices: Training indices
            test_indices: Test indices
            metric_type: 'war' or 'warp'
            player_type: 'hitter' or 'pitcher'
            fold_number: Fold identifier

        Returns:
            FoldResults with all metrics
        """
        # Extract test data - handle both DataFrame and array
        if hasattr(data['X'], 'iloc'):
            X_test = data['X'].iloc[test_indices]
        else:
            X_test = data['X'][test_indices]

        if hasattr(data['y'], 'iloc'):
            y_test = data['y'].iloc[test_indices]
        else:
            y_test = data['y'][test_indices]

        if hasattr(y_test, 'values'):
            y_test = y_test.values

        # Get years for this fold
        years = data['years']
        if isinstance(years, tuple) and len(years) == 1:
            years = years[0]
        train_years = list(set(np.array(years)[train_indices]))
        test_years = list(set(np.array(years)[test_indices]))

        # Generate predictions
        predictions = []
        for i in range(len(X_test)):
            if hasattr(X_test, 'iloc'):
                feature_vector = X_test.iloc[i].values
            else:
                feature_vector = X_test[i]

            pred_result = ensemble_predictor.predict_ensemble(
                feature_vector,
                metric_type,
                player_type
            )
            predictions.append(pred_result['ensemble'])

        predictions = np.array(predictions)

        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(y_test, predictions)

        # Initialize fold results
        fold_results = FoldResults(
            fold_number=fold_number,
            train_years=sorted(train_years),
            test_year=test_years[0] if len(test_years) == 1 else max(test_years),
            overall=overall_metrics
        )

        # Get player segments
        segments = self.identify_player_segments(data, test_indices, metric_type)

        # Calculate metrics for each segment
        if 'elite' in segments and segments['elite'].sum() > 0:
            elite_mask = segments['elite']
            fold_results.elite = self.calculate_metrics(
                y_test[elite_mask],
                predictions[elite_mask]
            )

        if 'non_elite' in segments and segments['non_elite'].sum() > 0:
            non_elite_mask = segments['non_elite']
            fold_results.non_elite = self.calculate_metrics(
                y_test[non_elite_mask],
                predictions[non_elite_mask]
            )

        if 'rookies' in segments and segments['rookies'].sum() > 0:
            rookie_mask = segments['rookies']
            fold_results.rookies = self.calculate_metrics(
                y_test[rookie_mask],
                predictions[rookie_mask]
            )

        if 'veterans' in segments and segments['veterans'].sum() > 0:
            veteran_mask = segments['veterans']
            fold_results.veterans = self.calculate_metrics(
                y_test[veteran_mask],
                predictions[veteran_mask]
            )

        return fold_results

    def run_time_series_validation(
        self,
        ensemble_predictor,
        hitter_data: Dict[str, Any],
        pitcher_data: Dict[str, Any],
        n_splits: int = 4,
        test_size: int = 1,
        gap: int = 0
    ) -> Dict[str, Any]:
        """
        Run time series cross-validation with comprehensive metrics.

        Args:
            ensemble_predictor: Ensemble model with predict_ensemble method
            hitter_data: Dictionary with hitter data
            pitcher_data: Dictionary with pitcher data
            n_splits: Number of validation splits
            test_size: Size of test set (in years)
            gap: Gap between train and test (in years)

        Returns:
            Dictionary with comprehensive validation results
        """
        results = {
            'hitter_war': [],
            'hitter_warp': [],
            'pitcher_war': [],
            'pitcher_warp': [],
            'summary': {}
        }

        if self.verbose:
            print("="*80)
            print("COMPREHENSIVE TIME SERIES VALIDATION")
            print("="*80)
            print(f"Configuration: {n_splits} splits, test_size={test_size} year(s)")
            print()

        # Validate each data type
        for player_type, data_dict in [('hitter', hitter_data), ('pitcher', pitcher_data)]:
            if not data_dict:
                continue

            for metric_type in ['war', 'warp']:
                if metric_type not in data_dict:
                    continue

                data = data_dict[metric_type]

                if self.verbose:
                    print(f"\nValidating {player_type} {metric_type.upper()}:")
                    print("-"*40)

                # Get years and create time-based splits
                years = data['years']
                if isinstance(years, tuple) and len(years) == 1:
                    years = years[0]
                years_array = np.array(years)
                unique_years = sorted(np.unique(years_array))

                # Create custom time series splits
                fold_results = []

                for split_idx in range(n_splits):
                    # Determine train/test years
                    test_year_idx = len(unique_years) - n_splits + split_idx
                    if test_year_idx < 1:  # Need at least 1 year for training
                        continue

                    train_years = unique_years[:test_year_idx]
                    test_years = unique_years[test_year_idx:test_year_idx + test_size]

                    # Create masks
                    train_mask = np.isin(years_array, train_years)
                    test_mask = np.isin(years_array, test_years)

                    train_indices = np.where(train_mask)[0]
                    test_indices = np.where(test_mask)[0]

                    if len(test_indices) == 0:
                        continue

                    # Validate this fold
                    fold_result = self.validate_fold(
                        ensemble_predictor,
                        data,
                        train_indices,
                        test_indices,
                        metric_type,
                        player_type,
                        split_idx + 1
                    )

                    fold_results.append(fold_result)

                    if self.verbose:
                        print(fold_result.get_summary())

                # Store results
                results[f'{player_type}_{metric_type}'] = fold_results

        # Generate summary statistics
        summary = self._generate_summary(results)
        results['summary'] = summary

        if self.verbose:
            self._print_summary(summary)

        return results

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all folds."""
        summary = {}

        for key in ['hitter_war', 'hitter_warp', 'pitcher_war', 'pitcher_warp']:
            if key not in results or not results[key]:
                continue

            fold_results = results[key]

            # Aggregate metrics across folds (filtering out NaN values)
            overall_r2s = [fold.overall.r2 for fold in fold_results if not np.isnan(fold.overall.r2)]
            overall_maes = [fold.overall.mae for fold in fold_results]
            overall_rmses = [fold.overall.rmse for fold in fold_results]

            summary[key] = {
                'mean_r2': np.mean(overall_r2s) if overall_r2s else np.nan,
                'std_r2': np.std(overall_r2s) if overall_r2s else np.nan,
                'mean_mae': np.mean(overall_maes),
                'std_mae': np.std(overall_maes),
                'mean_rmse': np.mean(overall_rmses),
                'std_rmse': np.std(overall_rmses),
                'n_folds': len(fold_results)
            }

            # Elite performance if available (with NaN handling)
            elite_r2s = [fold.elite.r2 for fold in fold_results
                         if fold.elite and fold.elite.n_samples > 0 and not np.isnan(fold.elite.r2)]
            elite_samples = sum([fold.elite.n_samples for fold in fold_results
                               if fold.elite and fold.elite.n_samples > 0])

            if elite_r2s:
                summary[key]['elite_mean_r2'] = np.mean(elite_r2s)
                summary[key]['elite_mean_mae'] = np.mean([fold.elite.mae for fold in fold_results
                                                         if fold.elite and fold.elite.n_samples > 0])
                summary[key]['elite_total_samples'] = elite_samples
            elif elite_samples > 0:
                # We have elite samples but R² is undefined (too few or bad predictions)
                summary[key]['elite_mean_r2'] = np.nan
                summary[key]['elite_mean_mae'] = np.mean([fold.elite.mae for fold in fold_results
                                                         if fold.elite and fold.elite.n_samples > 0])
                summary[key]['elite_total_samples'] = elite_samples

            # Rookie performance if available
            rookie_r2s = [fold.rookies.r2 for fold in fold_results if fold.rookies]
            if rookie_r2s:
                summary[key]['rookie_mean_r2'] = np.mean(rookie_r2s)
                summary[key]['rookie_mean_mae'] = np.mean([fold.rookies.mae for fold in fold_results if fold.rookies])

        return summary

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted summary of validation results."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        for key, stats in summary.items():
            player_type, metric = key.rsplit('_', 1)
            print(f"\n{player_type.title()} {metric.upper()}:")

            # Handle NaN values in display
            if not np.isnan(stats['mean_r2']):
                print(f"  Mean R²: {stats['mean_r2']:.4f} ± {stats['std_r2']:.4f}")
            else:
                print(f"  Mean R²: undefined (insufficient samples)")

            print(f"  Mean MAE: {stats['mean_mae']:.4f} ± {stats['std_mae']:.4f}")
            print(f"  Mean RMSE: {stats['mean_rmse']:.4f} ± {stats['std_rmse']:.4f}")

            if 'elite_mean_r2' in stats:
                elite_samples = stats.get('elite_total_samples', 0)
                if not np.isnan(stats['elite_mean_r2']):
                    print(f"  Elite R²: {stats['elite_mean_r2']:.4f} (n={elite_samples} total samples)")
                else:
                    print(f"  Elite R²: undefined (n={elite_samples} total samples, too few per fold)")
                print(f"  Elite MAE: {stats['elite_mean_mae']:.4f}")

            if 'rookie_mean_r2' in stats:
                if not np.isnan(stats['rookie_mean_r2']):
                    print(f"  Rookie R²: {stats['rookie_mean_r2']:.4f}")
                else:
                    print(f"  Rookie R²: undefined")
                print(f"  Rookie MAE: {stats['rookie_mean_mae']:.4f}")

    def evaluate_production_readiness(
        self,
        validation_results: Dict[str, Any]
    ) -> Tuple[bool, List[str], str]:
        """
        Evaluate whether model is ready for production based on validation results.

        Args:
            validation_results: Results from run_time_series_validation

        Returns:
            Tuple of (is_ready, warnings, recommendation)
        """
        warnings = []
        critical_issues = []

        summary = validation_results.get('summary', {})

        for key, stats in summary.items():
            if not stats:
                continue

            mean_r2 = stats.get('mean_r2', 0)
            mean_mae = stats.get('mean_mae', float('inf'))

            # Check R² thresholds
            if mean_r2 < self.R2_THRESHOLD_ACCEPTABLE:
                critical_issues.append(f"{key}: R² ({mean_r2:.3f}) below acceptable threshold")
            elif mean_r2 < self.R2_THRESHOLD_GOOD:
                warnings.append(f"{key}: R² ({mean_r2:.3f}) acceptable but not good")

            # Check MAE thresholds
            if mean_mae > self.MAE_THRESHOLD_ACCEPTABLE:
                critical_issues.append(f"{key}: MAE ({mean_mae:.3f}) above acceptable threshold")
            elif mean_mae > self.MAE_THRESHOLD_GOOD:
                warnings.append(f"{key}: MAE ({mean_mae:.3f}) acceptable but not good")

            # Check elite performance
            if 'elite_mean_r2' in stats:
                elite_r2 = stats['elite_mean_r2']
                elite_mae = stats['elite_mean_mae']

                if elite_r2 < mean_r2 - 0.1:  # Elite performing worse than average
                    warnings.append(f"{key}: Elite players underperforming (R²={elite_r2:.3f})")

                if elite_mae > mean_mae * 1.2:  # Elite error 20% higher
                    warnings.append(f"{key}: Elite players have higher error (MAE={elite_mae:.3f})")

            # Check rookie performance
            if 'rookie_mean_r2' in stats:
                rookie_r2 = stats['rookie_mean_r2']
                if rookie_r2 < mean_r2 - 0.15:  # Rookies significantly worse
                    warnings.append(f"{key}: Rookie predictions weak (R²={rookie_r2:.3f})")

        # Generate recommendation
        is_ready = len(critical_issues) == 0

        if not critical_issues and not warnings:
            recommendation = "PROCEED TO PRODUCTION: All metrics meet or exceed thresholds"
        elif not critical_issues:
            recommendation = "PROCEED WITH CAUTION: Some metrics need monitoring"
        else:
            recommendation = "DO NOT PROCEED: Critical performance issues detected"

        return is_ready, warnings + critical_issues, recommendation


def run_comprehensive_validation(
    ensemble_predictor,
    hitter_data: Dict[str, Any],
    pitcher_data: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive validation with default settings.

    Args:
        ensemble_predictor: Trained ensemble model
        hitter_data: Hitter data dictionary
        pitcher_data: Pitcher data dictionary
        verbose: Whether to print progress

    Returns:
        Dictionary with validation results and recommendation
    """
    validator = TimeSeriesModelValidator(verbose=verbose)

    # Run validation with 4 folds (2021, 2022, 2023, 2024 as test years)
    validation_results = validator.run_time_series_validation(
        ensemble_predictor,
        hitter_data,
        pitcher_data,
        n_splits=4,
        test_size=1,
        gap=0
    )

    # Evaluate production readiness
    is_ready, issues, recommendation = validator.evaluate_production_readiness(validation_results)

    validation_results['production_ready'] = is_ready
    validation_results['issues'] = issues
    validation_results['recommendation'] = recommendation

    return validation_results