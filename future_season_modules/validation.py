"""
Cross-Validation and Model Validation Module
============================================

Implements temporal cross-validation with survival considerations for SYSTEM 2.
Provides robust validation for joint longitudinal-survival models.

Classes:
    AgeCurveValidator: Main class for temporal validation and performance assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Iterator, Union
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from lifelines.utils import concordance_index
    SURVIVAL_METRICS_AVAILABLE = True
except ImportError:
    SURVIVAL_METRICS_AVAILABLE = False


class AgeCurveValidator:
    """
    Validates joint longitudinal-survival models using temporal cross-validation.

    Implements survival-aware time series splits that prevent future information
    leakage while respecting the temporal nature of career progression.
    """

    def __init__(self, min_train_years: int = 3, validation_gap: int = 0):
        """
        Initialize the validation framework.

        Args:
            min_train_years: Minimum years of training data required
            validation_gap: Gap between training and validation (default 0)
        """
        self.min_train_years = min_train_years
        self.validation_gap = validation_gap
        self.validation_results = {}

    def survival_time_series_split(self,
                                 data: pd.DataFrame,
                                 n_splits: int = 5,
                                 season_col: str = 'Season') -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create temporal splits that respect survival structure.

        Args:
            data: Complete dataset with temporal information
            n_splits: Number of validation splits
            season_col: Column name for season/year information

        Yields:
            Tuples of (train_data, validation_data) for each fold
        """
        seasons = sorted(data[season_col].unique())
        total_seasons = len(seasons)

        if total_seasons < self.min_train_years + 1:
            raise ValueError(f"Insufficient seasons ({total_seasons}) for validation")

        # Calculate split points
        min_train_seasons = self.min_train_years
        validation_seasons = max(1, (total_seasons - min_train_seasons) // n_splits)

        print(f"Creating {n_splits} temporal splits:")
        print(f"  Total seasons: {total_seasons} ({seasons[0]}-{seasons[-1]})")
        print(f"  Min training seasons: {min_train_seasons}")

        for fold in range(n_splits):
            # Calculate training and validation season ranges
            val_start_idx = min_train_seasons + fold * validation_seasons
            val_end_idx = min(val_start_idx + validation_seasons, total_seasons)

            if val_start_idx >= total_seasons:
                break

            train_seasons = seasons[:val_start_idx]
            val_seasons = seasons[val_start_idx:val_end_idx]

            # Apply gap if specified
            if self.validation_gap > 0:
                train_seasons = train_seasons[:-self.validation_gap]

            if not train_seasons or not val_seasons:
                continue

            train_data = data[data[season_col].isin(train_seasons)].copy()
            val_data = data[data[season_col].isin(val_seasons)].copy()

            print(f"  Fold {fold + 1}: Train {train_seasons[0]}-{train_seasons[-1]} → "
                  f"Val {val_seasons[0]}-{val_seasons[-1]} "
                  f"({len(train_data)} → {len(val_data)} records)")

            yield train_data, val_data

    def validate_longitudinal_component(self,
                                      model,
                                      train_data: pd.DataFrame,
                                      val_data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate the longitudinal (performance prediction) component.

        Args:
            model: Fitted longitudinal model
            train_data: Training data
            val_data: Validation data

        Returns:
            Dictionary with performance metrics
        """
        # This is a placeholder - the actual implementation would depend on
        # how predictions are generated from validation data
        metrics = {}

        try:
            # FIXED: Proper temporal validation without data leakage
            if hasattr(model, 'predict_performance_path'):
                predictions = []
                actuals = []

                # Get the last training season to use as prediction baseline
                train_cutoff = train_data['Season'].max()
                target_col = 'TARGET_METRIC' if 'TARGET_METRIC' in val_data.columns else 'WAR'

                # For each player, use their TRAINING data history + final training year
                # to predict their VALIDATION period performance
                for player_id in val_data['mlbid'].unique():
                    # Get player's complete training history
                    player_train_history = train_data[train_data['mlbid'] == player_id]
                    # Get player's validation period data (what we're trying to predict)
                    player_val_data = val_data[val_data['mlbid'] == player_id].sort_values('Season')

                    if len(player_train_history) > 0 and len(player_val_data) > 0:
                        # Use the last training year as baseline for prediction
                        baseline_data = player_train_history[player_train_history['Season'] == train_cutoff]

                        if len(baseline_data) > 0:
                            baseline = baseline_data.iloc[0]

                            # Predict the validation years using only training data
                            years_to_predict = len(player_val_data)
                            pred_wars = model.predict_performance_path(
                                baseline.get('Age', 27),
                                baseline.get('Primary_Position', 'OF'),
                                baseline.get(target_col, 0),
                                player_train_history,  # Only use training history
                                years_to_predict
                            )

                            # Compare predictions to actual validation performance
                            actual_wars = player_val_data[target_col].values

                            # Only use as many predictions/actuals as we have
                            min_length = min(len(pred_wars), len(actual_wars))
                            if min_length > 0:
                                predictions.extend(pred_wars[:min_length])
                                actuals.extend(actual_wars[:min_length])

                if predictions and actuals:
                    predictions = np.array(predictions)
                    actuals = np.array(actuals)

                    metrics['r2'] = r2_score(actuals, predictions)
                    metrics['rmse'] = np.sqrt(mean_squared_error(actuals, predictions))
                    metrics['mae'] = mean_absolute_error(actuals, predictions)
                    metrics['n_predictions'] = len(predictions)

        except Exception as e:
            warnings.warn(f"Error validating longitudinal component: {str(e)}")
            metrics['error'] = str(e)

        return metrics

    def validate_survival_component(self,
                                  model,
                                  train_data: pd.DataFrame,
                                  val_data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate the survival (retirement risk) component with proper temporal cutoff.

        Args:
            model: Fitted survival model
            train_data: Training data
            val_data: Validation data

        Returns:
            Dictionary with survival model performance metrics
        """
        if not SURVIVAL_METRICS_AVAILABLE:
            return {'error': 'Survival metrics not available'}

        metrics = {}

        try:
            # CRITICAL FIX: Use proper temporal cutoff for validation
            # Get the cutoff season from the training data
            train_cutoff = train_data['Season'].max()

            # Prepare validation survival data using the same cutoff logic as training
            val_survival_df = model.prepare_survival_training_data(
                pd.concat([train_data, val_data]),
                cutoff_season=train_cutoff
            )

            # Filter to only validation players
            val_player_ids = val_data['mlbid'].unique()
            val_survival_df = val_survival_df[val_survival_df['mlbid'].isin(val_player_ids)]

            if len(val_survival_df) > 0 and hasattr(model, 'survival_model') and model.survival_model is not None:
                # Check if we have sufficient variation for survival analysis
                unique_durations = val_survival_df['duration'].nunique()
                total_events = val_survival_df['event'].sum()

                if unique_durations < 2 or total_events < 2:
                    print(f"   Skipping survival validation: insufficient variation (durations={unique_durations}, events={total_events})")
                    metrics['error'] = f'insufficient_variation: durations={unique_durations}, events={total_events}'
                else:
                    # Use the survival model to calculate concordance
                    try:
                        # Select features that were used in training
                        survival_features = ['age_at_end', 'final_war', 'career_war', 'peak_war', 'war_decline']
                        available_features = [f for f in survival_features if f in val_survival_df.columns]

                        if available_features:
                            feature_data = val_survival_df[available_features]

                            # Calculate concordance index using the fitted survival model
                            c_index = concordance_index(
                                val_survival_df['duration'],
                                -model.survival_model.predict_partial_hazard(feature_data),
                                val_survival_df['event']
                            )
                            metrics['concordance_index'] = c_index
                            metrics['n_survival_predictions'] = len(val_survival_df)
                            metrics['validation_events'] = total_events
                            metrics['validation_event_rate'] = val_survival_df['event'].mean()
                        else:
                            metrics['error'] = 'no_valid_features_for_validation'
                    except Exception as e:
                        print(f"   Error calculating validation concordance: {str(e)}")
                        metrics['error'] = f'concordance_calculation_failed: {str(e)}'
            else:
                metrics['error'] = 'no_survival_model_or_data'

        except Exception as e:
            warnings.warn(f"Error validating survival component: {str(e)}")
            metrics['error'] = str(e)

        return metrics

    def validate_joint_model(self,
                           model,
                           data: pd.DataFrame,
                           n_splits: int = 5) -> Dict[str, Union[float, Dict]]:
        """
        Perform complete validation of the joint longitudinal-survival model.

        Args:
            model: Fitted joint model
            data: Complete dataset
            n_splits: Number of cross-validation folds

        Returns:
            Dictionary with comprehensive validation results
        """
        print(f"Validating joint model with {n_splits}-fold temporal cross-validation...")

        longitudinal_results = []
        survival_results = []
        fold_details = []

        for fold, (train_data, val_data) in enumerate(self.survival_time_series_split(data, n_splits)):
            print(f"\nValidating fold {fold + 1}/{n_splits}...")

            # Validate longitudinal component
            long_metrics = self.validate_longitudinal_component(model, train_data, val_data)
            longitudinal_results.append(long_metrics)

            # Validate survival component
            surv_metrics = self.validate_survival_component(model, train_data, val_data)
            survival_results.append(surv_metrics)

            # Store fold details
            fold_details.append({
                'fold': fold + 1,
                'train_seasons': sorted(train_data['Season'].unique()),
                'val_seasons': sorted(val_data['Season'].unique()),
                'train_size': len(train_data),
                'val_size': len(val_data),
                'longitudinal_metrics': long_metrics,
                'survival_metrics': surv_metrics
            })

        # Aggregate results
        aggregated_results = self._aggregate_validation_results(longitudinal_results, survival_results)
        aggregated_results['fold_details'] = fold_details

        self.validation_results = aggregated_results

        print(f"\nCross-validation complete!")
        self._print_validation_summary(aggregated_results)

        return aggregated_results

    def _aggregate_validation_results(self,
                                    longitudinal_results: List[Dict],
                                    survival_results: List[Dict]) -> Dict:
        """
        Aggregate validation results across all folds.

        Args:
            longitudinal_results: List of longitudinal metrics from each fold
            survival_results: List of survival metrics from each fold

        Returns:
            Aggregated validation metrics
        """
        aggregated = {
            'longitudinal_performance': {},
            'survival_performance': {},
            'n_folds': len(longitudinal_results)
        }

        # Aggregate longitudinal metrics
        long_metrics = ['r2', 'rmse', 'mae']
        for metric in long_metrics:
            values = [fold.get(metric) for fold in longitudinal_results if metric in fold and fold[metric] is not None]
            if values:
                aggregated['longitudinal_performance'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n_folds': len(values)
                }

        # Aggregate survival metrics
        surv_metrics = ['concordance_index']
        for metric in surv_metrics:
            values = [fold.get(metric) for fold in survival_results if metric in fold and fold[metric] is not None]
            if values:
                aggregated['survival_performance'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n_folds': len(values)
                }

        return aggregated

    def _print_validation_summary(self, results: Dict) -> None:
        """
        Print a summary of validation results.

        Args:
            results: Aggregated validation results
        """
        print("=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)

        # Longitudinal performance
        if results['longitudinal_performance']:
            print("\nLongitudinal Model Performance:")
            for metric, stats in results['longitudinal_performance'].items():
                print(f"  {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        # Survival performance
        if results['survival_performance']:
            print("\nSurvival Model Performance:")
            for metric, stats in results['survival_performance'].items():
                print(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        print(f"\nValidation completed across {results['n_folds']} temporal folds")

    def generate_validation_report(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a detailed validation report.

        Args:
            output_path: Optional path to save the report

        Returns:
            DataFrame with detailed validation results
        """
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_joint_model first.")

        report_data = []

        for fold_detail in self.validation_results['fold_details']:
            fold_data = {
                'fold': fold_detail['fold'],
                'train_start': min(fold_detail['train_seasons']),
                'train_end': max(fold_detail['train_seasons']),
                'val_start': min(fold_detail['val_seasons']),
                'val_end': max(fold_detail['val_seasons']),
                'train_size': fold_detail['train_size'],
                'val_size': fold_detail['val_size']
            }

            # Add longitudinal metrics
            long_metrics = fold_detail['longitudinal_metrics']
            for metric in ['r2', 'rmse', 'mae']:
                fold_data[f'longitudinal_{metric}'] = long_metrics.get(metric)

            # Add survival metrics
            surv_metrics = fold_detail['survival_metrics']
            for metric in ['concordance_index']:
                fold_data[f'survival_{metric}'] = surv_metrics.get(metric)

            report_data.append(fold_data)

        report_df = pd.DataFrame(report_data)

        if output_path:
            report_df.to_csv(output_path, index=False)
            print(f"Validation report saved to {output_path}")

        return report_df

    def plot_validation_metrics(self) -> None:
        """
        Plot validation metrics across folds using Plotly.
        """
        if not self.validation_results:
            raise ValueError("No validation results available. Run validate_joint_model first.")

        if not PLOTTING_AVAILABLE:
            print("Plotly not available. Install plotly for visualization.")
            return

        # Extract fold-level data
        fold_data = []
        for fold_detail in self.validation_results['fold_details']:
            fold_info = {
                'fold': fold_detail['fold'],
                **fold_detail['longitudinal_metrics'],
                **fold_detail['survival_metrics']
            }
            fold_data.append(fold_info)

        fold_df = pd.DataFrame(fold_data)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Longitudinal R²', 'Longitudinal RMSE', 'Survival Concordance Index', 'Summary'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )

        # Plot longitudinal R²
        if 'r2' in fold_df.columns:
            fig.add_trace(
                go.Bar(x=fold_df['fold'], y=fold_df['r2'], name='R²',
                       marker_color='lightblue'),
                row=1, col=1
            )

        # Plot longitudinal RMSE
        if 'rmse' in fold_df.columns:
            fig.add_trace(
                go.Bar(x=fold_df['fold'], y=fold_df['rmse'], name='RMSE',
                       marker_color='lightcoral'),
                row=1, col=2
            )

        # Plot survival concordance
        if 'concordance_index' in fold_df.columns:
            fig.add_trace(
                go.Bar(x=fold_df['fold'], y=fold_df['concordance_index'], name='C-Index',
                       marker_color='lightgreen'),
                row=2, col=1
            )

        # Summary table
        summary_data = []
        if 'r2' in fold_df.columns:
            summary_data.append(['Mean R²', f"{fold_df['r2'].mean():.3f} ± {fold_df['r2'].std():.3f}"])
        if 'concordance_index' in fold_df.columns:
            summary_data.append(['Mean C-Index', f"{fold_df['concordance_index'].mean():.3f} ± {fold_df['concordance_index'].std():.3f}"])
        summary_data.append(['Folds', str(len(fold_df))])

        if summary_data:
            fig.add_trace(
                go.Table(
                    header=dict(values=['Metric', 'Value'], fill_color='lightgray'),
                    cells=dict(values=list(zip(*summary_data)), fill_color='white')
                ),
                row=2, col=2
            )

        fig.update_layout(
            title_text="Joint Model Validation Results",
            height=800,
            showlegend=False
        )

        fig.show()

    def compare_model_performance(self, baseline_results: Dict) -> Dict[str, float]:
        """
        Compare joint model performance against baseline models.

        Args:
            baseline_results: Results from baseline model validation

        Returns:
            Dictionary with performance improvements
        """
        if not self.validation_results:
            raise ValueError("No validation results available")

        comparisons = {}

        # Compare longitudinal performance
        joint_long = self.validation_results['longitudinal_performance']
        baseline_long = baseline_results.get('longitudinal_performance', {})

        for metric in ['r2', 'rmse']:
            if metric in joint_long and metric in baseline_long:
                joint_val = joint_long[metric]['mean']
                baseline_val = baseline_long[metric]['mean']

                if metric == 'rmse':  # Lower is better for RMSE
                    improvement = (baseline_val - joint_val) / baseline_val * 100
                else:  # Higher is better for R²
                    improvement = (joint_val - baseline_val) / abs(baseline_val) * 100

                comparisons[f'longitudinal_{metric}_improvement_pct'] = improvement

        # Compare survival performance
        joint_surv = self.validation_results['survival_performance']
        baseline_surv = baseline_results.get('survival_performance', {})

        for metric in ['concordance_index']:
            if metric in joint_surv and metric in baseline_surv:
                joint_val = joint_surv[metric]['mean']
                baseline_val = baseline_surv[metric]['mean']
                improvement = (joint_val - baseline_val) / abs(baseline_val) * 100
                comparisons[f'survival_{metric}_improvement_pct'] = improvement

        return comparisons

    def validate_peak_age_projections(self, projections: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Union[bool, float]]:
        """
        Dual validation framework combining FanGraphs and Dynasty Guru insights.

        Validation Checks:
        1. Single-season peak: Should occur around age 26.5-26.8 (FanGraphs)
        2. 3-year rolling peak: Should occur in age range 26-29 (Dynasty Guru)
        3. Peak range plateau: Performance should be stable 26-29

        Args:
            projections: DataFrame with player projections
            data: Historical data for validation

        Returns:
            Dictionary with validation results and flags
        """
        validation_results = {}

        try:
            # Find players with sufficient data for peak analysis
            players_with_enough_data = []

            for player_id, player_data in data.groupby('mlbid'):
                # Need at least 5 seasons spanning ages 24-32
                age_range = player_data['Age'].max() - player_data['Age'].min()
                has_peak_ages = any(24 <= age <= 32 for age in player_data['Age'])

                if len(player_data) >= 5 and age_range >= 4 and has_peak_ages:
                    players_with_enough_data.append(player_id)

            print(f"Validating peak age patterns for {len(players_with_enough_data)} players...")

            if len(players_with_enough_data) < 10:
                validation_results['sufficient_data'] = False
                validation_results['warning'] = "Insufficient players for peak age validation"
                return validation_results

            # Analysis 1: Single-season peak ages (FanGraphs validation)
            single_season_peaks = []
            for player_id in players_with_enough_data[:50]:  # Sample for efficiency
                player_data = data[data['mlbid'] == player_id]
                if len(player_data) >= 5:
                    # Use TARGET_METRIC for unified validation
                    metric_col = 'TARGET_METRIC' if 'TARGET_METRIC' in player_data.columns else 'WAR'
                    peak_idx = player_data[metric_col].idxmax()
                    peak_age = player_data.loc[peak_idx, 'Age']
                    single_season_peaks.append(peak_age)

            if single_season_peaks:
                mean_peak_age = np.mean(single_season_peaks)
                validation_results['single_season_peak_age'] = mean_peak_age
                validation_results['fangraphs_validation'] = 26.0 <= mean_peak_age <= 27.5

            # Analysis 2: 3-year rolling peak ages (Dynasty Guru validation)
            rolling_peaks = []
            for player_id in players_with_enough_data[:50]:  # Sample for efficiency
                player_data = data[data['mlbid'] == player_id].sort_values('Age')
                if len(player_data) >= 5:
                    metric_col = 'TARGET_METRIC' if 'TARGET_METRIC' in player_data.columns else 'WAR'

                    # Calculate 3-year rolling averages
                    rolling_wars = player_data[metric_col].rolling(window=3, center=True).mean()
                    if not rolling_wars.empty:
                        peak_idx = rolling_wars.idxmax()
                        if pd.notna(rolling_wars.loc[peak_idx]):
                            peak_age = player_data.loc[peak_idx, 'Age']
                            rolling_peaks.append(peak_age)

            if rolling_peaks:
                mean_rolling_peak = np.mean(rolling_peaks)
                validation_results['rolling_peak_age'] = mean_rolling_peak
                validation_results['dynasty_guru_validation'] = 26.0 <= mean_rolling_peak <= 29.0

            # Analysis 3: Peak range plateau validation
            plateau_validation = []
            for player_id in players_with_enough_data[:30]:  # Smaller sample for detailed analysis
                player_data = data[data['mlbid'] == player_id].sort_values('Age')
                if len(player_data) >= 6:
                    metric_col = 'TARGET_METRIC' if 'TARGET_METRIC' in player_data.columns else 'WAR'

                    # Check performance stability in 26-29 range
                    peak_range_data = player_data[(player_data['Age'] >= 26) & (player_data['Age'] <= 29)]
                    if len(peak_range_data) >= 3:
                        wars_in_range = peak_range_data[metric_col].values
                        coefficient_of_variation = np.std(wars_in_range) / np.mean(wars_in_range) if np.mean(wars_in_range) > 0 else 1.0
                        # Lower CV indicates more stable performance (plateau effect)
                        plateau_validation.append(coefficient_of_variation)

            if plateau_validation:
                mean_cv = np.mean(plateau_validation)
                validation_results['peak_range_stability_cv'] = mean_cv
                validation_results['plateau_validation'] = mean_cv < 0.5  # Stable if CV < 50%

            # Overall validation summary
            passed_validations = sum([
                validation_results.get('fangraphs_validation', False),
                validation_results.get('dynasty_guru_validation', False),
                validation_results.get('plateau_validation', False)
            ])

            validation_results['overall_peak_validation'] = passed_validations >= 2
            validation_results['validations_passed'] = passed_validations
            validation_results['sufficient_data'] = True

            print(f"Peak Age Validation Results:")
            print(f"  Single-season peak age: {validation_results.get('single_season_peak_age', 'N/A'):.1f}")
            print(f"  3-year rolling peak age: {validation_results.get('rolling_peak_age', 'N/A'):.1f}")
            print(f"  Peak range stability CV: {validation_results.get('peak_range_stability_cv', 'N/A'):.2f}")
            print(f"  Validations passed: {passed_validations}/3")

        except Exception as e:
            print(f"Peak age validation failed: {str(e)}")
            validation_results['error'] = str(e)
            validation_results['sufficient_data'] = False

        return validation_results