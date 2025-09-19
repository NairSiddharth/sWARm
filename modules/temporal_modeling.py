"""
Temporal Modeling Module for oWAR Analysis

This module provides year-over-year prediction capabilities for proper animation:
- Sequential year training (train on past, predict future)
- Year-organized prediction results
- Temporal analysis utilities for animated visualizations
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any

__all__ = [
    'create_temporal_splits',
    'run_temporal_predictions',
    'organize_results_by_year',
    'validate_temporal_data'
]

def create_temporal_splits(X, y, player_names, seasons,
                          min_train_years: int = 3,
                          prediction_years: List[int] = None) -> Dict[str, Any]:
    """
    Create temporal train/test splits for year-over-year predictions

    Args:
        X: Feature matrix
        y: Target values
        player_names: Player names
        seasons: Season information
        min_train_years: Minimum years needed for training
        prediction_years: Specific years to predict (None = auto-detect)

    Returns:
        Dict with temporal splits organized by prediction year
    """

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({
        'features': [list(x) for x in X],
        'target': y,
        'player': player_names,
        'season': seasons
    })

    # Get available years
    available_years = sorted(df['season'].unique())

    if prediction_years is None:
        # Predict on years that have enough training data
        prediction_years = [year for year in available_years
                          if len([y for y in available_years if y < year]) >= min_train_years]

    temporal_splits = {}

    for pred_year in prediction_years:
        # Training data: all years before prediction year
        train_years = [y for y in available_years if y < pred_year]

        if len(train_years) < min_train_years:
            continue

        # Get training and test data
        train_df = df[df['season'].isin(train_years)]
        test_df = df[df['season'] == pred_year]

        if len(test_df) == 0:
            continue

        # Extract features and targets
        X_train = np.array(train_df['features'].tolist())
        y_train = np.array(train_df['target'].tolist())
        names_train = train_df['player'].tolist()
        seasons_train = train_df['season'].tolist()

        X_test = np.array(test_df['features'].tolist())
        y_test = np.array(test_df['target'].tolist())
        names_test = test_df['player'].tolist()
        seasons_test = test_df['season'].tolist()

        temporal_splits[pred_year] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'names_train': names_train,
            'names_test': names_test,
            'seasons_train': seasons_train,
            'seasons_test': seasons_test,
            'train_years': train_years,
            'test_year': pred_year
        }

    return temporal_splits

def run_temporal_predictions(model, temporal_splits: Dict[str, Any],
                           model_name: str, player_type: str, metric: str) -> Dict[str, Any]:
    """
    Run predictions for each year using temporal splits

    Args:
        model: Sklearn/Keras model to use
        temporal_splits: Output from create_temporal_splits
        model_name: Name of the model
        player_type: 'hitter' or 'pitcher'
        metric: 'war' or 'warp'

    Returns:
        Dict with year-organized predictions
    """

    temporal_results = {}

    for pred_year, splits in temporal_splits.items():
        print(f"Training {model_name} {player_type} {metric} for {pred_year} prediction...")
        print(f"  Training years: {splits['train_years']}")
        print(f"  Training samples: {len(splits['X_train'])}")
        print(f"  Test samples: {len(splits['X_test'])}")

        # Train model on historical data
        model.fit(splits['X_train'], splits['y_train'])

        # Predict for the target year
        y_pred = model.predict(splits['X_test'])

        temporal_results[pred_year] = {
            'y_true': splits['y_test'],
            'y_pred': y_pred,
            'player_names': splits['names_test'],
            'seasons': splits['seasons_test'],  # Should all be pred_year
            'model': model_name,
            'player_type': player_type,
            'metric': metric,
            'train_years': splits['train_years'],
            'test_year': pred_year
        }

    return temporal_results

def organize_results_by_year(model_results) -> Dict[str, Any]:
    """
    Reorganize existing ModelResults by year for animation compatibility

    Args:
        model_results: ModelResults instance

    Returns:
        Dict organized by year for each model/player_type/metric combination
    """

    year_organized = {}

    for key, results in model_results.results.items():
        # Parse key (e.g., "linear_hitter_war")
        parts = key.split('_')
        model_name = parts[0]
        player_type = parts[1]
        metric = parts[2]

        # Get seasons
        seasons = results.get('Season', ['2021'] * len(results['y_true']))

        # Group by year
        year_groups = {}
        for i, season in enumerate(seasons):
            if season not in year_groups:
                year_groups[season] = {
                    'y_true': [],
                    'y_pred': [],
                    'player_names': [],
                    'seasons': []
                }

            year_groups[season]['y_true'].append(results['y_true'][i])
            year_groups[season]['y_pred'].append(results['y_pred'][i])
            year_groups[season]['player_names'].append(results['player_names'][i])
            year_groups[season]['seasons'].append(season)

        # Store organized results
        year_organized[key] = {
            'model': model_name,
            'player_type': player_type,
            'metric': metric,
            'by_year': year_groups
        }

    return year_organized

def validate_temporal_data(temporal_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate temporal prediction results for animation compatibility

    Args:
        temporal_results: Output from run_temporal_predictions or organize_results_by_year

    Returns:
        Validation report
    """

    validation = {
        'total_years': len(temporal_results),
        'years_covered': sorted(temporal_results.keys()) if temporal_results else [],
        'predictions_per_year': {},
        'player_coverage': {},
        'warnings': []
    }

    all_players = set()

    for year, results in temporal_results.items():
        if isinstance(results, dict) and 'player_names' in results:
            players = set(results['player_names'])
            validation['predictions_per_year'][year] = len(players)
            validation['player_coverage'][year] = players
            all_players.update(players)
        elif isinstance(results, dict) and 'by_year' in results:
            # Handle organize_results_by_year format
            for sub_year, sub_results in results['by_year'].items():
                players = set(sub_results['player_names'])
                validation['predictions_per_year'][sub_year] = len(players)
                validation['player_coverage'][sub_year] = players
                all_players.update(players)

    # Check for consistent player coverage across years
    year_counts = len(validation['years_covered'])
    if year_counts > 1:
        consistent_players = set(validation['player_coverage'][validation['years_covered'][0]])
        for year in validation['years_covered'][1:]:
            year_players = validation['player_coverage'][year]
            consistent_players = consistent_players.intersection(year_players)

        validation['consistent_players'] = len(consistent_players)
        validation['total_unique_players'] = len(all_players)

        if len(consistent_players) < 10:
            validation['warnings'].append(f"Only {len(consistent_players)} players appear in all years")

    return validation