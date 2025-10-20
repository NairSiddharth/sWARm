"""
Ensemble Comparison Script

Compares the new three-tree ensemble (RF + XGBoost + LightGBM) against the
original RF + Keras ensemble to validate the hypothesis that tree-based models
better handle the IP discontinuity problem for pitchers.

Key metrics:
- Overall R² on holdout year
- Pitcher-specific R² (relievers vs starters)
- Individual reliever predictions (elite relievers should improve)
- Feature importance comparison

Usage:
    python compare_ensembles.py
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from current_season_modules.modeling import ensemble_modeling
from current_season_modules.modeling import ensemble_modeling_trees
from common_modules.logging import get_logger

logger = get_logger(__name__)


def analyze_pitcher_predictions(predictions, actuals, pitcher_data, ip_threshold=100):
    """
    Analyze predictions specifically for relievers vs starters

    Args:
        predictions: Array of predicted values
        actuals: Array of actual values
        pitcher_data: DataFrame with pitcher info including IP
        ip_threshold: IP cutoff for starter/reliever classification

    Returns:
        dict: Analysis results
    """
    # Classify pitchers
    is_reliever = pitcher_data['IP'] < ip_threshold
    is_starter = pitcher_data['IP'] >= ip_threshold

    results = {
        'all_pitchers': {
            'r2': r2_score(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mae': mean_absolute_error(actuals, predictions),
            'n_samples': len(actuals)
        }
    }

    if is_reliever.any():
        reliever_idx = is_reliever.values
        results['relievers'] = {
            'r2': r2_score(actuals[reliever_idx], predictions[reliever_idx]),
            'rmse': np.sqrt(mean_squared_error(actuals[reliever_idx], predictions[reliever_idx])),
            'mae': mean_absolute_error(actuals[reliever_idx], predictions[reliever_idx]),
            'n_samples': reliever_idx.sum()
        }

    if is_starter.any():
        starter_idx = is_starter.values
        results['starters'] = {
            'r2': r2_score(actuals[starter_idx], predictions[starter_idx]),
            'rmse': np.sqrt(mean_squared_error(actuals[starter_idx], predictions[starter_idx])),
            'mae': mean_absolute_error(actuals[starter_idx], predictions[starter_idx]),
            'n_samples': starter_idx.sum()
        }

    return results


def find_elite_relievers(pitcher_data, predictions_old, predictions_new, actuals, metric_type='war'):
    """
    Find elite relievers and compare old vs new predictions

    Elite relievers: IP < 100, actual WAR > 2.0 (or WARP > 1.5)
    """
    threshold_war = 2.0 if metric_type == 'war' else 1.5

    is_reliever = pitcher_data['IP'] < 100
    is_elite = actuals > threshold_war
    elite_relievers = is_reliever & is_elite

    if not elite_relievers.any():
        return None

    elite_idx = elite_relievers.values
    comparison = pd.DataFrame({
        'Name': pitcher_data.loc[elite_idx, 'Name'].values if 'Name' in pitcher_data.columns else [f"Player_{i}" for i in range(elite_idx.sum())],
        'IP': pitcher_data.loc[elite_idx, 'IP'].values,
        'Actual': actuals[elite_idx],
        'Old_Pred': predictions_old[elite_idx],
        'New_Pred': predictions_new[elite_idx],
        'Old_Error': predictions_old[elite_idx] - actuals[elite_idx],
        'New_Error': predictions_new[elite_idx] - actuals[elite_idx],
        'Improvement': np.abs(predictions_old[elite_idx] - actuals[elite_idx]) - np.abs(predictions_new[elite_idx] - actuals[elite_idx])
    })

    comparison['Old_Error_Pct'] = (comparison['Old_Error'] / comparison['Actual']) * 100
    comparison['New_Error_Pct'] = (comparison['New_Error'] / comparison['Actual']) * 100

    return comparison.sort_values('Improvement', ascending=False)


def compare_ensembles(hitter_data, pitcher_data, holdout_year=2024):
    """
    Main comparison function

    Args:
        hitter_data: Dictionary with 'warp' and 'war' data for hitters
        pitcher_data: Dictionary with 'warp' and 'war' data for pitchers
        holdout_year: Year to use for testing

    Returns:
        dict: Comprehensive comparison results
    """
    print("="*80)
    print("ENSEMBLE COMPARISON: RF+Keras vs RF+XGBoost+LightGBM")
    print("="*80)
    print()

    results = {}

    # Train both ensembles
    print("Training ORIGINAL ensemble (RF + Keras)...")
    original_ensemble = ensemble_modeling.create_ensemble_for_data(
        hitter_data, pitcher_data, holdout_year
    )
    print()

    print("Training NEW ensemble (RF + XGBoost + LightGBM)...")
    trees_ensemble = ensemble_modeling_trees.create_ensemble_for_data(
        hitter_data, pitcher_data, holdout_year
    )
    print()

    # Compare on holdout year for each combination
    for player_type, data_dict in [('pitcher', pitcher_data), ('hitter', hitter_data)]:
        if not data_dict:
            continue

        results[player_type] = {}

        for metric_type, data in data_dict.items():
            if not data:
                continue

            print(f"\n{'='*80}")
            print(f"Comparing {player_type.upper()} {metric_type.upper()} predictions")
            print(f"{'='*80}")

            # Get holdout data
            years_data = data.get('years', [])
            if isinstance(years_data, tuple):
                years_data = years_data[0]

            holdout_mask = np.array(years_data) == holdout_year
            if not holdout_mask.any():
                print(f"No holdout data for {player_type} {metric_type}")
                continue

            X_holdout = data['X'][holdout_mask].values if hasattr(data['X'], 'values') else np.array(data['X'])[holdout_mask]
            y_holdout = data['y'][holdout_mask].values if hasattr(data['y'], 'values') else np.array(data['y'])[holdout_mask]

            # Generate predictions from both ensembles
            original_preds = []
            trees_preds = []

            for features in X_holdout:
                orig_result = original_ensemble.predict_ensemble(features, metric_type, player_type)
                tree_result = trees_ensemble.predict_ensemble(features, metric_type, player_type)
                original_preds.append(orig_result['ensemble'])
                trees_preds.append(tree_result['ensemble'])

            original_preds = np.array(original_preds)
            trees_preds = np.array(trees_preds)

            # Overall metrics
            orig_r2 = r2_score(y_holdout, original_preds)
            tree_r2 = r2_score(y_holdout, trees_preds)
            orig_rmse = np.sqrt(mean_squared_error(y_holdout, original_preds))
            tree_rmse = np.sqrt(mean_squared_error(y_holdout, trees_preds))

            print(f"\nOVERALL PERFORMANCE:")
            print(f"  Original (RF+Keras):      R² = {orig_r2:.4f}, RMSE = {orig_rmse:.4f}")
            print(f"  New (RF+XGB+LGBM):        R² = {tree_r2:.4f}, RMSE = {tree_rmse:.4f}")
            print(f"  Improvement:              ΔR² = {tree_r2 - orig_r2:+.4f}, ΔRMSE = {tree_rmse - orig_rmse:+.4f}")

            results[player_type][metric_type] = {
                'overall': {
                    'original_r2': orig_r2,
                    'trees_r2': tree_r2,
                    'original_rmse': orig_rmse,
                    'trees_rmse': tree_rmse,
                    'r2_improvement': tree_r2 - orig_r2,
                    'rmse_improvement': orig_rmse - tree_rmse
                }
            }

            # Pitcher-specific analysis
            if player_type == 'pitcher':
                # Reconstruct pitcher info from holdout data
                # Need IP for classification - get from X_holdout
                pitcher_info = pd.DataFrame({
                    'IP': X_holdout[:, 0],  # IP is first feature
                })

                print(f"\nPITCHER ROLE ANALYSIS:")

                # Analyze relievers vs starters
                original_analysis = analyze_pitcher_predictions(
                    original_preds, y_holdout, pitcher_info
                )
                trees_analysis = analyze_pitcher_predictions(
                    trees_preds, y_holdout, pitcher_info
                )

                for role in ['relievers', 'starters']:
                    if role in original_analysis and role in trees_analysis:
                        orig = original_analysis[role]
                        tree = trees_analysis[role]
                        print(f"\n  {role.upper()} (n={orig['n_samples']}):")
                        print(f"    Original:  R² = {orig['r2']:.4f}, RMSE = {orig['rmse']:.4f}")
                        print(f"    New:       R² = {tree['r2']:.4f}, RMSE = {tree['rmse']:.4f}")
                        print(f"    Improvement: ΔR² = {tree['r2'] - orig['r2']:+.4f}")

                        results[player_type][metric_type][role] = {
                            'original_r2': orig['r2'],
                            'trees_r2': tree['r2'],
                            'r2_improvement': tree['r2'] - orig['r2'],
                            'n_samples': orig['n_samples']
                        }

                # Find elite relievers
                elite_comparison = find_elite_relievers(
                    pitcher_info, original_preds, trees_preds, y_holdout, metric_type
                )

                if elite_comparison is not None:
                    print(f"\nELITE RELIEVERS (IP < 100, {metric_type.upper()} > threshold):")
                    print(f"  Found {len(elite_comparison)} elite relievers")
                    print(f"\n  Top 5 improvements:")
                    print(elite_comparison.head().to_string(index=False))

                    avg_improvement = elite_comparison['Improvement'].mean()
                    print(f"\n  Average improvement in absolute error: {avg_improvement:+.4f}")

                    results[player_type][metric_type]['elite_relievers'] = {
                        'n_elite': len(elite_comparison),
                        'avg_improvement': avg_improvement,
                        'details': elite_comparison
                    }

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nKey findings:")

    for player_type in results:
        for metric_type in results[player_type]:
            data = results[player_type][metric_type]
            overall = data['overall']

            if overall['r2_improvement'] > 0.01:
                verdict = "✓ SIGNIFICANT IMPROVEMENT"
            elif overall['r2_improvement'] > 0:
                verdict = "✓ Slight improvement"
            else:
                verdict = "✗ No improvement"

            print(f"\n{player_type.upper()} {metric_type.upper()}: {verdict}")
            print(f"  ΔR² = {overall['r2_improvement']:+.4f}")

            # Pitcher role-specific
            if player_type == 'pitcher':
                if 'relievers' in data:
                    rel_improvement = data['relievers']['r2_improvement']
                    print(f"  Relievers: ΔR² = {rel_improvement:+.4f}")
                if 'starters' in data:
                    sta_improvement = data['starters']['r2_improvement']
                    print(f"  Starters: ΔR² = {sta_improvement:+.4f}")

    return results


def main():
    """Main execution function"""
    print("Loading and preparing data...")

    try:
        from current_season_modules.modeling.data_preparation import prepare_data_for_kfold

        hitter_data, pitcher_data = prepare_data_for_kfold()

        if hitter_data is None or pitcher_data is None:
            print("Error: Could not load data")
            return None

        # Run comparison
        results = compare_ensembles(hitter_data, pitcher_data, holdout_year=2024)

        # Save results
        import pickle
        output_path = Path(__file__).parent / "ensemble_comparison_results.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {output_path}")

        return results

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
