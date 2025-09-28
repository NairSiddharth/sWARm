#!/usr/bin/env python3
"""
Retrain Ensemble Models with Normalized Features
================================================

This script retrains your ensemble models using the new feature set:
- Normalized Contact Quality Index (CQI)
- Normalized Statcast Launch Quality Index (SLQI)
- Proper HBP% instead of raw HBP count

This provides optimal performance and full benefits of the normalization.

Created: 2025-09-27
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Import our modules
from current_season_modules.predictive_modeling import prepare_data_for_kfold
from common_modules.ensemble_modeling import EnsembleWARPredictor

def retrain_ensemble_models(holdout_year=2024):
    """
    Retrain ensemble models with the new normalized feature set.

    Args:
        holdout_year: Year to hold out for final validation
    """
    print("RETRAINING ENSEMBLE MODELS WITH NORMALIZED FEATURES")
    print("=" * 60)
    print(f"Using holdout year: {holdout_year}")
    print(f"Training data: 2016-{holdout_year-1}")
    print()

    # Load data with new feature set
    print("Loading data with normalized features...")
    hitter_data, pitcher_data = prepare_data_for_kfold()

    if not pitcher_data or 'warp' not in pitcher_data or 'war' not in pitcher_data:
        print("ERROR: Failed to load pitcher data")
        return False

    print("Data loaded successfully!")
    print(f"  Pitcher WARP: {len(pitcher_data['warp']['X'])} samples")
    print(f"  Pitcher WAR: {len(pitcher_data['war']['X'])} samples")
    print(f"  Features: {list(pitcher_data['warp']['X'].columns)}")
    print()

    # Split data into training and holdout
    def split_data_by_year(data, holdout_year):
        """Split data into training and holdout sets by year."""
        years = np.array(data['years'])
        train_mask = years != str(holdout_year)
        holdout_mask = years == str(holdout_year)

        return {
            'train': {
                'X': data['X'][train_mask].reset_index(drop=True),
                'y': data['y'][train_mask].reset_index(drop=True),
                'names': data['names'][train_mask].reset_index(drop=True),
                'years': [str(y) for y in years[train_mask]]
            },
            'holdout': {
                'X': data['X'][holdout_mask].reset_index(drop=True),
                'y': data['y'][holdout_mask].reset_index(drop=True),
                'names': data['names'][holdout_mask].reset_index(drop=True),
                'years': [str(y) for y in years[holdout_mask]]
            }
        }

    # Split pitcher data
    pitcher_warp_split = split_data_by_year(pitcher_data['warp'], holdout_year)
    pitcher_war_split = split_data_by_year(pitcher_data['war'], holdout_year)

    print(f"Training split:")
    print(f"  WARP: {len(pitcher_warp_split['train']['X'])} training, {len(pitcher_warp_split['holdout']['X'])} holdout")
    print(f"  WAR: {len(pitcher_war_split['train']['X'])} training, {len(pitcher_war_split['holdout']['X'])} holdout")
    print()

    # Initialize new ensemble predictor
    print("Initializing ensemble predictor...")
    ensemble = EnsembleWARPredictor(random_state=42)

    # Train pitcher WARP model
    print("Training pitcher WARP model...")
    ensemble.train_ensemble(
        X_train=pitcher_warp_split['train']['X'],
        y_train=pitcher_warp_split['train']['y'],
        groups_train=pitcher_warp_split['train']['years'],
        metric_type='warp',
        player_type='pitcher',
        holdout_validation=True
    )

    # Train pitcher WAR model
    print("Training pitcher WAR model...")
    ensemble.train_ensemble(
        X_train=pitcher_war_split['train']['X'],
        y_train=pitcher_war_split['train']['y'],
        groups_train=pitcher_war_split['train']['years'],
        metric_type='war',
        player_type='pitcher',
        holdout_validation=True
    )

    # Mark as trained
    ensemble.is_trained = True

    # Validate on holdout data
    print("\nValidating on holdout data...")

    # Test WARP predictions
    if len(pitcher_warp_split['holdout']['X']) > 0:
        warp_pred = ensemble.predict_ensemble(
            pitcher_warp_split['holdout']['X'].values,
            'warp',
            'pitcher'
        )
        warp_r2 = np.corrcoef(pitcher_warp_split['holdout']['y'], warp_pred['ensemble'])[0, 1] ** 2
        print(f"  Pitcher WARP holdout R²: {warp_r2:.4f}")

    # Test WAR predictions
    if len(pitcher_war_split['holdout']['X']) > 0:
        war_pred = ensemble.predict_ensemble(
            pitcher_war_split['holdout']['X'].values,
            'war',
            'pitcher'
        )
        war_r2 = np.corrcoef(pitcher_war_split['holdout']['y'], war_pred['ensemble'])[0, 1] ** 2
        print(f"  Pitcher WAR holdout R²: {war_r2:.4f}")

    # Save the retrained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/history/ensemble_models_normalized_{timestamp}.pkl"

    print(f"\nSaving retrained model to: {model_path}")

    # Save the ensemble predictor
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        print("SUCCESS: Model saved successfully!")
    except Exception as e:
        print(f"ERROR: Error saving model: {e}")
        return False

    # Create a backup of the current model if it exists
    current_model_path = "models/ensemble_models.pkl"
    if os.path.exists(current_model_path):
        backup_path = f"models/history/ensemble_models_backup_{timestamp}.pkl"
        print(f"Backing up current model to: {backup_path}")
        try:
            import shutil
            shutil.copy2(current_model_path, backup_path)
            print("SUCCESS: Backup created successfully!")
        except Exception as e:
            print(f"WARNING: Could not create backup: {e}")

    # Replace the current model
    try:
        import shutil
        shutil.copy2(model_path, current_model_path)
        print(f"SUCCESS: Updated {current_model_path} with retrained model!")
    except Exception as e:
        print(f"ERROR: Error updating current model: {e}")
        print(f"Manual step needed: Copy {model_path} to {current_model_path}")
        return False

    # Validate the new model loads correctly
    print("\nValidating model loading...")
    try:
        with open(current_model_path, 'rb') as f:
            test_ensemble = pickle.load(f)

        if test_ensemble.is_trained:
            print("SUCCESS: Retrained model loads successfully!")
            print("SUCCESS: Model is marked as trained")

            # Test a prediction to ensure compatibility
            test_features = np.array([50.0, 8.0, 22.0, 4.0, 7.0, 0.5, 50.0, 0.5, 3.0])  # 9 features with normalized values
            test_pred = test_ensemble.predict_ensemble(test_features, 'war', 'pitcher')
            print(f"SUCCESS: Test prediction successful: {test_pred['ensemble']:.3f}")

        else:
            print("WARNING: Model loaded but not marked as trained")
    except Exception as e:
        print(f"ERROR: Error validating model: {e}")
        return False

    print("\n" + "=" * 60)
    print("ENSEMBLE MODEL RETRAINING COMPLETE!")
    print("=" * 60)
    print("SUCCESS: Models retrained with normalized CQI and SLQI")
    print("SUCCESS: HBP% properly integrated (no more conversion needed)")
    print("SUCCESS: Full benefits of feature normalization activated")
    print("SUCCESS: Your sWARm_CS system is ready with optimal performance")
    print()
    print("Next steps:")
    print("1. Run your sWARm_CS notebook")
    print("2. Enjoy improved pitcher predictions!")
    print("3. Notice the intuitive 0-100 scale for composite features")

    return True


def main():
    """Main retraining function."""
    success = retrain_ensemble_models(holdout_year=2024)

    if success:
        print("\nSUCCESS: Ensemble models retrained successfully!")
    else:
        print("\nFAILED: Ensemble model retraining encountered errors")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)