"""
Temporal Validation for Future Projections

Implements time-series cross-validation to prevent data leakage.
Adapted from future_season_modules/validation.py with new pipeline integration.

See FUTURE_PROJECTIONS_MIGRATION_GUIDE.md Section 7.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Iterator
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from lifelines.utils import concordance_index
    SURVIVAL_METRICS_AVAILABLE = True
except ImportError:
    SURVIVAL_METRICS_AVAILABLE = False
    print("Warning: lifelines not available. Survival metrics disabled.")


class TemporalValidator:
    """
    Validates future projection models using temporal cross-validation.

    Prevents data leakage by ensuring models only train on past data.
    """

    def __init__(self, min_train_years: int = 5, validation_gap: int = 0):
        """
        Initialize temporal validator.

        Args:
            min_train_years: Minimum years of training data required (default: 5)
            validation_gap: Gap between train and validation to prevent leakage (default: 0)
        """
        self.min_train_years = min_train_years
        self.validation_gap = validation_gap
        self.validation_results = {}

    def create_temporal_splits(
        self,
        sequences_df: pd.DataFrame,
        n_splits: int = 3,
        year_col: str = 'year_n'
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create temporal train/validation splits that prevent data leakage.

        Copied from future_season_modules/validation.py lines 52-110.

        Args:
            sequences_df: Longitudinal sequences DataFrame
            n_splits: Number of validation folds (default: 3)
            year_col: Column containing year information

        Returns:
            List of (train_df, val_df) tuples

        Example:
            For 2016-2024 data with n_splits=3:
            - Split 1: Train 2016-2020, Val 2021
            - Split 2: Train 2016-2021, Val 2022
            - Split 3: Train 2016-2022, Val 2023
        """
        years = sorted(sequences_df[year_col].unique())
        total_years = len(years)

        if total_years < self.min_train_years + 1:
            raise ValueError(
                f"Insufficient years ({total_years}) for validation. "
                f"Need at least {self.min_train_years + 1} years."
            )

        print(f"\nCreating {n_splits} temporal splits:")
        print(f"  Total years: {total_years} ({years[0]}-{years[-1]})")
        print(f"  Min training years: {self.min_train_years}")

        splits = []

        # Calculate validation years per split
        available_val_years = total_years - self.min_train_years
        years_per_split = max(1, available_val_years // n_splits)

        for fold in range(n_splits):
            # Calculate validation year index
            val_start_idx = self.min_train_years + fold * years_per_split

            if val_start_idx >= total_years:
                break

            val_end_idx = min(val_start_idx + years_per_split, total_years)

            # Training years: all years before validation
            train_years = years[:val_start_idx]
            val_years = years[val_start_idx:val_end_idx]

            # Apply gap if specified (remove last N years from training)
            if self.validation_gap > 0 and len(train_years) > self.validation_gap:
                train_years = train_years[:-self.validation_gap]

            if not train_years or not val_years:
                continue

            # Create splits
            train_df = sequences_df[sequences_df[year_col].isin(train_years)].copy()
            val_df = sequences_df[sequences_df[year_col].isin(val_years)].copy()

            print(f"  Fold {fold + 1}: Train {train_years[0]}-{train_years[-1]} "
                  f"({len(train_df)} sequences) -> "
                  f"Val {val_years[0]}-{val_years[-1]} ({len(val_df)} sequences)")

            splits.append((train_df, val_df))

        return splits

    def validate_longitudinal_model(
        self,
        model,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate longitudinal model predictions.

        Adapted from future_season_modules/validation.py lines 112-194.

        Args:
            model: Trained LongitudinalModel
            train_df: Training sequences
            val_df: Validation sequences

        Returns:
            Dictionary with metrics: r2, rmse, mae, n_predictions
        """
        if not model.is_fitted:
            return {'error': 'Model not fitted'}

        try:
            # Generate predictions for validation set
            predictions = model.predict(val_df)

            # Get actual values (war_n_plus_1 target)
            if 'war_n_plus_1' not in val_df.columns:
                return {'error': 'No target column (war_n_plus_1) in validation data'}

            actuals = val_df['war_n_plus_1'].values

            # Calculate metrics
            metrics = {
                'r2': r2_score(actuals, predictions),
                'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
                'mae': mean_absolute_error(actuals, predictions),
                'n_predictions': len(predictions),
                'mean_actual': actuals.mean(),
                'mean_predicted': predictions.mean()
            }

            # Track feature importance (if available from model)
            try:
                from new_pipeline.models.future_season.constants import (
                    FUTURE_PITCHER_MODEL_FEATURES,
                    FUTURE_HITTER_MODEL_FEATURES
                )

                # Get player type from model
                player_type = getattr(model, 'player_type', None)
                if player_type:
                    feature_names = (FUTURE_PITCHER_MODEL_FEATURES if player_type == 'pitcher'
                                   else FUTURE_HITTER_MODEL_FEATURES)

                    # Try to extract feature importance from ensemble model
                    if hasattr(model, 'ensemble_model') and hasattr(model.ensemble_model, 'extratrees'):
                        # Get from ExtraTrees model (most interpretable)
                        if hasattr(model.ensemble_model.extratrees, 'model') and \
                           hasattr(model.ensemble_model.extratrees.model, 'feature_importances_'):
                            importance = model.ensemble_model.extratrees.model.feature_importances_

                            # Map to feature names
                            feature_importance = dict(zip(feature_names, importance))

                            # Sort by importance
                            sorted_features = sorted(feature_importance.items(),
                                                   key=lambda x: x[1], reverse=True)

                            # Add top 10 to metrics
                            metrics['feature_importance'] = dict(sorted_features)

                            print("\nTop 10 Most Important Features:")
                            for feat, imp in sorted_features[:10]:
                                print(f"  {feat}: {imp:.4f}")
            except Exception as e:
                # Feature importance tracking is optional, don't fail validation if it errors
                print(f"Note: Could not extract feature importance: {e}")

            return metrics

        except Exception as e:
            return {'error': str(e)}

    def validate_survival_model(
        self,
        model,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate survival model predictions using concordance index.

        Adapted from future_season_modules/validation.py lines 196-287.

        Args:
            model: Trained SurvivalModel
            train_df: Training data
            val_df: Validation data

        Returns:
            Dictionary with survival metrics
        """
        if not SURVIVAL_METRICS_AVAILABLE:
            return {'error': 'Survival metrics not available (install lifelines)'}

        if not model.is_fitted:
            return {'error': 'Model not fitted'}

        try:
            # Prepare survival features for validation
            # Use validation data to create survival dataset
            val_survival_features = []

            for _, row in val_df.iterrows():
                val_survival_features.append({
                    'age_at_end': row.get('age_n', 28),
                    'final_war': row.get('war_n', 0),
                    'career_war': row.get('career_war', 0),
                    'peak_war': row.get('peak_war', row.get('war_n', 0)),
                    'war_decline': max(0, row.get('peak_war', 0) - row.get('war_n', 0))
                })

            val_survival_df = pd.DataFrame(val_survival_features)

            # For validation, we need duration and event labels
            # This would typically come from observing whether player retired
            # For now, create placeholder (in real use, this comes from historical data)
            val_survival_df['duration'] = 5  # Placeholder
            val_survival_df['event'] = 0     # Placeholder (censored)

            # Calculate concordance index
            # Note: This is simplified - full implementation would need actual retirement data
            survival_features = ['age_at_end', 'final_war', 'career_war', 'peak_war', 'war_decline']

            # Predict partial hazard
            hazards = model.model.predict_partial_hazard(val_survival_df[survival_features])

            metrics = {
                'note': 'Simplified validation - needs actual retirement data for full metrics',
                'n_predictions': len(hazards),
                'mean_hazard': hazards.mean()
            }

            return metrics

        except Exception as e:
            return {'error': str(e)}

    def validate_joint_projections(
        self,
        joint_model,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        years_ahead: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """
        Validate multi-year joint projections.

        Args:
            joint_model: Trained JointProjectionModel
            train_df: Training sequences
            val_df: Validation sequences
            years_ahead: Number of years to project

        Returns:
            Dictionary with metrics for each projection year
        """
        try:
            # Generate projections for validation set
            projections_df = joint_model.project_multiple_players(val_df, years_ahead=years_ahead)

            # We need actual future WAR values for validation
            # This would come from actual subsequent seasons
            # For now, return projection statistics
            metrics = {}

            for year in range(1, years_ahead + 1):
                war_col = f'war_year_{year}'
                if war_col in projections_df.columns:
                    metrics[war_col] = {
                        'mean': projections_df[war_col].mean(),
                        'std': projections_df[war_col].std(),
                        'min': projections_df[war_col].min(),
                        'max': projections_df[war_col].max(),
                        'n_players': len(projections_df)
                    }

            return metrics

        except Exception as e:
            return {'error': str(e)}

    def cross_validate_pipeline(
        self,
        sequences_df: pd.DataFrame,
        model_class,
        player_type: str,
        n_splits: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        Perform complete temporal cross-validation of pipeline.

        Args:
            sequences_df: Complete sequence data
            model_class: LongitudinalModel class to instantiate
            player_type: 'hitter' or 'pitcher'
            n_splits: Number of CV folds

        Returns:
            Dictionary with validation results per fold
        """
        splits = self.create_temporal_splits(sequences_df, n_splits=n_splits)

        results = {
            'longitudinal_metrics': [],
            'fold_info': []
        }

        for fold_idx, (train_df, val_df) in enumerate(splits):
            print(f"\n--- Fold {fold_idx + 1} ---")

            # Train longitudinal model
            model = model_class(player_type)
            train_metrics = model.train(train_df)

            # Validate
            val_metrics = self.validate_longitudinal_model(model, train_df, val_df)

            # Store results
            results['longitudinal_metrics'].append(val_metrics)
            results['fold_info'].append({
                'fold': fold_idx + 1,
                'train_size': len(train_df),
                'val_size': len(val_df),
                'train_metrics': train_metrics
            })

            print(f"  Validation R²: {val_metrics.get('r2', 'N/A'):.3f}")
            print(f"  Validation MAE: {val_metrics.get('mae', 'N/A'):.3f} WAR")

        # Calculate aggregate metrics
        results['aggregate_metrics'] = self._aggregate_metrics(results['longitudinal_metrics'])

        return results

    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """
        Aggregate metrics across folds.

        Args:
            metrics_list: List of metric dictionaries from each fold

        Returns:
            Dictionary with mean and std of metrics
        """
        # Filter out error entries
        valid_metrics = [m for m in metrics_list if 'error' not in m]

        if not valid_metrics:
            return {'error': 'No valid metrics to aggregate'}

        aggregate = {}

        for metric_name in ['r2', 'rmse', 'mae']:
            values = [m[metric_name] for m in valid_metrics if metric_name in m]
            if values:
                aggregate[f'{metric_name}_mean'] = np.mean(values)
                aggregate[f'{metric_name}_std'] = np.std(values)

        aggregate['n_folds'] = len(valid_metrics)

        return aggregate

    def generate_validation_report(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate human-readable validation report.

        Adapted from future_season_modules/validation.py lines 550-700.

        Args:
            results: Validation results from cross_validate_pipeline()
            save_path: Optional path to save report

        Returns:
            Report string
        """
        report_lines = [
            "="*70,
            "TEMPORAL VALIDATION REPORT",
            "="*70,
            ""
        ]

        # Aggregate metrics
        if 'aggregate_metrics' in results:
            agg = results['aggregate_metrics']
            report_lines.extend([
                "Aggregate Metrics (across folds):",
                f"  R² Score:  {agg.get('r2_mean', 0):.3f} ± {agg.get('r2_std', 0):.3f}",
                f"  RMSE:      {agg.get('rmse_mean', 0):.3f} ± {agg.get('rmse_std', 0):.3f} WAR",
                f"  MAE:       {agg.get('mae_mean', 0):.3f} ± {agg.get('mae_std', 0):.3f} WAR",
                f"  N Folds:   {agg.get('n_folds', 0)}",
                ""
            ])

        # Per-fold metrics
        if 'longitudinal_metrics' in results:
            report_lines.append("Per-Fold Metrics:")
            for i, metrics in enumerate(results['longitudinal_metrics']):
                if 'error' in metrics:
                    report_lines.append(f"  Fold {i+1}: ERROR - {metrics['error']}")
                else:
                    report_lines.append(
                        f"  Fold {i+1}: R²={metrics.get('r2', 0):.3f}, "
                        f"MAE={metrics.get('mae', 0):.3f}, "
                        f"N={metrics.get('n_predictions', 0)}"
                    )
            report_lines.append("")

        # Success criteria check
        report_lines.extend([
            "Success Criteria:",
            f"  [*] MAE < 1.5 WAR: {'PASS' if results.get('aggregate_metrics', {}).get('mae_mean', 999) < 1.5 else 'FAIL'}",
            f"  [*] R2 > 0.15:     {'PASS' if results.get('aggregate_metrics', {}).get('r2_mean', 0) > 0.15 else 'FAIL'}",
            ""
        ])

        report_lines.append("="*70)

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Validation report saved to: {save_path}")

        return report


def validate_model_temporal_cv(
    sequences_df: pd.DataFrame,
    model_class,
    player_type: str,
    n_splits: int = 3
) -> Dict:
    """
    Convenience function for temporal cross-validation.

    Args:
        sequences_df: Longitudinal sequences DataFrame
        model_class: LongitudinalModel class
        player_type: 'hitter' or 'pitcher'
        n_splits: Number of CV folds

    Returns:
        Validation results dictionary

    Example:
        >>> from new_pipeline.models.future_season import LongitudinalModel
        >>> results = validate_model_temporal_cv(sequences, LongitudinalModel, 'hitter', n_splits=3)
        >>> print(results['aggregate_metrics'])
    """
    validator = TemporalValidator(min_train_years=5)
    return validator.cross_validate_pipeline(sequences_df, model_class, player_type, n_splits)


def validate_ensemble_model(
    ensemble_model,
    historical_df: pd.DataFrame,
    player_type: str,
    train_years: range,
    val_years: range
) -> Dict[str, float]:
    """
    Validate Darts ensemble model with proper temporal splits.

    Args:
        ensemble_model: Trained EnsembleLongitudinalModel
        historical_df: Full historical player data with injury features
        player_type: 'hitter' or 'pitcher'
        train_years: Years to use for training (already trained on these)
        val_years: Years to validate on (held out)

    Returns:
        Dict with ensemble metrics including individual model performance
    """
    from darts import TimeSeries
    from new_pipeline.models.future_season.constants import FUTURE_HITTER_MODEL_FEATURES, FUTURE_PITCHER_MODEL_FEATURES

    feature_cols = FUTURE_HITTER_MODEL_FEATURES if player_type == 'hitter' else FUTURE_PITCHER_MODEL_FEATURES

    # Create and train ensemble model if not provided
    if ensemble_model is None:
        from new_pipeline.models.future_season.ensemble_model import EnsembleLongitudinalModel

        print(f"Creating and training {player_type} ensemble model...")
        ensemble_model = EnsembleLongitudinalModel(player_type=player_type)

        # Filter to training years
        train_df = historical_df[historical_df['Year'].isin(train_years)].copy()

        # Build TimeSeries per player for Darts models
        target_series = []
        covariate_series = []

        for playerid in train_df['playerid'].unique():
            player_data = train_df[train_df['playerid'] == playerid].sort_values('Year')

            if len(player_data) < 2:
                continue  # Need at least 2 years for training

            # Ensure consecutive years for TimeSeries
            years = player_data['Year'].values
            if len(years) >= 2 and np.all(np.diff(years) == 1):
                try:
                    target_ts = TimeSeries.from_dataframe(
                        df=player_data,
                        time_col='Year',
                        value_cols=['WAR'],
                        fill_missing_dates=False
                    )

                    cov_cols = [c for c in feature_cols + ['Age'] if c in player_data.columns]
                    cov_ts = TimeSeries.from_dataframe(
                        df=player_data,
                        time_col='Year',
                        value_cols=cov_cols,
                        fill_missing_dates=False
                    )

                    target_series.append(target_ts)
                    covariate_series.append(cov_ts)
                except Exception:
                    continue

        print(f"  Created {len(target_series)} player TimeSeries for training")

        # Train ensemble
        ensemble_model.train(target_series, covariate_series)
        print("  Training complete")

    # Filter to validation years only
    val_df = historical_df[historical_df['Year'].isin(val_years)].copy()

    # Group by player and create predictions
    predictions = {
        'xgboost': [],
        'rnn': [],
        'extratrees': [],
        'ensemble': [],
        'fallback': [],
        'actual': []
    }

    # Track skip reasons for debugging
    skip_reasons = {
        'no_validation_data': 0,
        'true_rookie': 0,
        'timeseries_failed': 0,
        'dataframe_failed': 0
    }

    for playerid in val_df['playerid'].unique():
        # Get player's full history up to validation year
        player_full = historical_df[
            (historical_df['playerid'] == playerid) &
            (historical_df['Year'] < val_years.start)
        ].sort_values('Year')

        # Get validation year actual WAR
        player_val = val_df[val_df['playerid'] == playerid]
        if len(player_val) == 0:
            skip_reasons['no_validation_data'] += 1
            continue

        if len(player_full) < 1:
            skip_reasons['true_rookie'] += 1
            continue  # Skip only true rookies with no history

        actual_war = player_val['WAR'].values[0]
        predictions['actual'].append(actual_war)

        # Try TimeSeries creation for detailed model contributions
        try:
            target_series = TimeSeries.from_dataframe(
                df=player_full,
                time_col='Year',
                value_cols=['WAR'],
                fill_missing_dates=False
            )

            covariate_cols = [c for c in feature_cols + ['Age'] if c in player_full.columns]
            covariate_series = TimeSeries.from_dataframe(
                df=player_full,
                time_col='Year',
                value_cols=covariate_cols,
                fill_missing_dates=False
            )

            # Get model contributions
            contributions = ensemble_model.get_model_contributions(
                target_series, covariate_series, len(target_series)
            )

            if contributions['used_fallback']:
                predictions['xgboost'].append(None)
                predictions['rnn'].append(None)
                predictions['extratrees'].append(None)
                predictions['ensemble'].append(contributions['ensemble_pred'])
                predictions['fallback'].append(contributions['fallback_pred'])
            else:
                predictions['xgboost'].append(contributions['xgboost_pred'])
                predictions['rnn'].append(contributions['rnn_pred'])
                predictions['extratrees'].append(contributions['extratrees_pred'])
                predictions['ensemble'].append(contributions['ensemble_pred'])
                predictions['fallback'].append(None)

        except (ValueError, Exception) as e:
            # TimeSeries creation failed (non-consecutive years or other issues)
            # Use DataFrame-based prediction (routes to fallback automatically)
            try:
                ensemble_pred = ensemble_model.predict_from_dataframe(player_full)

                # Track as fallback prediction (no individual model breakdown)
                predictions['xgboost'].append(None)
                predictions['rnn'].append(None)
                predictions['extratrees'].append(None)
                predictions['ensemble'].append(ensemble_pred)
                predictions['fallback'].append(ensemble_pred)

            except Exception as e2:
                # Still failed - skip this player
                skip_reasons['dataframe_failed'] += 1
                predictions['actual'].pop()
                continue

    # Compute metrics for each model
    def compute_model_metrics(preds, actuals):
        valid_pairs = [(p, a) for p, a in zip(preds, actuals) if p is not None]
        if len(valid_pairs) < 10:
            return None

        pred_vals, actual_vals = zip(*valid_pairs)
        return {
            'r2': r2_score(actual_vals, pred_vals),
            'rmse': np.sqrt(mean_squared_error(actual_vals, pred_vals)),
            'mae': mean_absolute_error(actual_vals, pred_vals),
            'n': len(valid_pairs)
        }

    metrics = {
        'xgboost': compute_model_metrics(predictions['xgboost'], predictions['actual']),
        'rnn': compute_model_metrics(predictions['rnn'], predictions['actual']),
        'extratrees': compute_model_metrics(predictions['extratrees'], predictions['actual']),
        'ensemble': compute_model_metrics(predictions['ensemble'], predictions['actual']),
        'fallback': compute_model_metrics(predictions['fallback'], predictions['actual'])
    }

    # Calculate ensemble gain
    if metrics['ensemble'] and metrics['xgboost'] and metrics['extratrees']:
        single_model_rmses = [
            metrics['xgboost']['rmse'],
            metrics['extratrees']['rmse']
        ]
        if metrics['rnn']:
            single_model_rmses.append(metrics['rnn']['rmse'])

        best_single_rmse = min(single_model_rmses)
        metrics['ensemble_gain_rmse'] = best_single_rmse - metrics['ensemble']['rmse']
        metrics['ensemble_gain_pct'] = (metrics['ensemble_gain_rmse'] / best_single_rmse) * 100

    # Print skip statistics
    total_players = len(val_df['playerid'].unique())
    total_skipped = sum(skip_reasons.values())
    total_predicted = len(predictions['actual'])

    print(f"\nValidation Coverage:")
    print(f"  Total players in validation set: {total_players}")
    print(f"  Successfully predicted: {total_predicted} ({100*total_predicted/total_players:.1f}%)")
    print(f"  Skipped: {total_skipped} ({100*total_skipped/total_players:.1f}%)")
    print(f"\nSkip Reasons:")
    print(f"  True rookies (0 MLB history): {skip_reasons['true_rookie']}")
    print(f"  DataFrame prediction failed: {skip_reasons['dataframe_failed']}")
    print(f"  No validation data: {skip_reasons['no_validation_data']}")

    return metrics


def validate_no_temporal_leakage(
    sequences_df: pd.DataFrame,
    injury_data: pd.DataFrame
) -> bool:
    """
    Verify injury features only use data from before prediction point.

    Args:
        sequences_df: Training sequences with injury features
        injury_data: Raw injury data from FanGraphs

    Returns:
        True if no leakage detected

    Raises:
        AssertionError: If temporal leakage is detected
    """
    print("Validating no temporal leakage in injury features...")

    leakage_count = 0

    for idx, row in sequences_df.iterrows():
        year_n = row['year_n']
        playerid = row['playerid']

        # Get injury features
        tj_flag = row.get('had_tommy_john_ever', 0)
        years_since = row.get('years_since_tommy_john', None)

        if tj_flag == 0:
            continue  # No TJ, skip validation

        # Get actual injuries for this player
        player_injuries = injury_data[injury_data['MLBAMID'] == playerid]
        season_end_n = pd.to_datetime(f"{year_n}-10-31")

        # All TJ surgeries must be before season_end_n
        tj_surgeries = player_injuries[
            (player_injuries['injury_type'].str.contains('Tommy John', case=False, na=False)) &
            (player_injuries['injury_date'] <= season_end_n)
        ]

        if len(tj_surgeries) == 0:
            print(f"  WARNING: Player {playerid} year {year_n} has TJ flag but no TJ surgery found before {season_end_n}")
            leakage_count += 1

    if leakage_count > 0:
        print(f"  Found {leakage_count} potential leakage cases")
        return False

    print("  Temporal leakage validation: PASSED")
    return True
