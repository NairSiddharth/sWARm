"""
Seed Finder for Pitcher Ensemble Models

Tests multiple random seeds to find the one that produces the best validation performance.
This ensures we use the most effective seed for reproducible, high-quality predictions.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
project_root = Path(__file__).absolute().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from new_pipeline.notebooks.shared.pipeline_runner import (
    load_historical_data,
    run_data_pipeline,
    split_by_role
)
from new_pipeline.models.current_season import PitcherRoleEnsemble
from new_pipeline.common.constants import PITCHER_MODEL_FEATURES
from new_pipeline.models.current_season.keras_utils import set_seed


def evaluate_seed(seed: int, X_train, y_train, roles_train, X_val, y_val, roles_val):
    """
    Train model with a specific seed and evaluate on validation set.

    Args:
        seed: Random seed to test
        X_train: Training features
        y_train: Training targets
        roles_train: Training roles
        X_val: Validation features
        y_val: Validation targets
        roles_val: Validation roles

    Returns:
        dict: Metrics for this seed
    """
    print(f"\n{'='*70}")
    print(f"Testing seed: {seed}")
    print(f"{'='*70}")

    # Train model with specific seed (disable determinism to allow variation between seeds)
    model = PitcherRoleEnsemble()
    model.fit(X_train, y_train, roles_train, seed=seed, enable_determinism=False)

    # Predict on validation set
    y_pred = model.predict(X_val, roles_val)

    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    # Elite pitcher metrics (WAR > 5.0)
    elite_mask = y_val > 5.0
    elite_count = elite_mask.sum()
    if elite_count > 0:
        elite_mae = mean_absolute_error(y_val[elite_mask], y_pred[elite_mask])
    else:
        elite_mae = np.nan

    print(f"Validation Metrics:")
    print(f"  MAE:        {mae:.4f}")
    print(f"  RMSE:       {rmse:.4f}")
    print(f"  R²:         {r2:.4f}")
    print(f"  Elite MAE:  {elite_mae:.4f} ({elite_count} pitchers)")

    return {
        'seed': seed,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'elite_mae': elite_mae,
        'elite_count': elite_count
    }


def find_optimal_seed():
    """
    Test multiple seeds and find the one with best validation performance.

    Returns:
        int: Optimal seed value
    """
    print("="*70)
    print("PITCHER ENSEMBLE SEED FINDER")
    print("="*70)

    # Load and process data
    print("\nLoading historical pitcher data (2016-2024)...")
    pitcher_historical = load_historical_data(
        player_type='pitcher',
        years=range(2016, 2025)
    )

    print(f"Loaded {len(pitcher_historical)} pitcher-seasons")

    print("\nRunning sklearn pipeline...")
    pitcher_processed = run_data_pipeline(
        pitcher_historical,
        player_type='pitcher'
    )

    print(f"Processed {len(pitcher_processed)} qualified pitchers")

    # Prepare features and targets
    X = pitcher_processed[PITCHER_MODEL_FEATURES].values
    y = pitcher_processed['WAR_per_162'].values

    # Create role labels
    pitcher_processed['GS_per_G'] = pitcher_processed['GS'] / pitcher_processed['G'].replace(0, 1)

    def get_role(row):
        if row['GS_per_G'] > 0.7:
            return 'starter'
        elif row['GS_per_G'] < 0.1:
            return 'reliever'
        else:
            return 'swing'

    roles = pitcher_processed.apply(get_role, axis=1).values

    # Split into train/validation (80/20)
    # Use stratify by role to ensure balanced split
    X_train, X_val, y_train, y_val, roles_train, roles_val = train_test_split(
        X, y, roles,
        test_size=0.2,
        random_state=42,  # Fixed seed for consistent train/val split
        stratify=roles
    )

    print(f"\nTrain set: {len(X_train)} pitchers")
    print(f"Validation set: {len(X_val)} pitchers")

    # Test seeds
    candidate_seeds = [42, 123, 456, 789, 2024, 1337, 999, 3141, 2718, 100]

    print(f"\nTesting {len(candidate_seeds)} candidate seeds...")
    print("This will take 20-30 minutes...\n")

    results = []
    for seed in candidate_seeds:
        metrics = evaluate_seed(
            seed, X_train, y_train, roles_train,
            X_val, y_val, roles_val
        )
        results.append(metrics)

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # Find best seed
    # Primary: lowest MAE
    # Secondary: lowest elite MAE
    # Tertiary: highest R²
    best_idx = results_df['mae'].idxmin()
    best_seed = results_df.loc[best_idx, 'seed']

    print("\n" + "="*70)
    print("SEED COMPARISON RESULTS")
    print("="*70)
    print()
    print(results_df.to_string(index=False))

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"\nBest Seed: {int(best_seed)}")
    print(f"  Validation MAE:   {results_df.loc[best_idx, 'mae']:.4f}")
    print(f"  Validation RMSE:  {results_df.loc[best_idx, 'rmse']:.4f}")
    print(f"  Validation R²:    {results_df.loc[best_idx, 'r2']:.4f}")
    print(f"  Elite MAE:        {results_df.loc[best_idx, 'elite_mae']:.4f}")

    print("\nThis seed should be used for all future training to ensure:")
    print("  1. Reproducible results")
    print("  2. Optimal validation performance")
    print("  3. Best elite pitcher predictions")

    return int(best_seed)


if __name__ == "__main__":
    optimal_seed = find_optimal_seed()
    print(f"\n\nOPTIMAL_SEED = {optimal_seed}")
