#!/usr/bin/env python3
"""
Quick Retrain with 10 Features
===============================

Retrain the ensemble models with the correct 10-feature set that matches
the current system expectations.

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

def quick_retrain():
    """Quick retrain with correct feature count."""
    print("QUICK RETRAIN WITH 10 FEATURES")
    print("=" * 40)

    # Load data
    print("Loading data...")
    hitter_data, pitcher_data = prepare_data_for_kfold()

    if not pitcher_data:
        print("ERROR: Failed to load pitcher data")
        return False

    print(f"Pitcher WARP features: {list(pitcher_data['warp']['X'].columns)}")
    print(f"Pitcher WAR features: {list(pitcher_data['war']['X'].columns)}")
    print(f"Feature count: {len(pitcher_data['warp']['X'].columns)}")

    # Initialize and train ensemble
    print("Training ensemble models...")
    ensemble = EnsembleWARPredictor(random_state=42)

    # Train pitcher WARP model
    ensemble.train_ensemble(
        X_train=pitcher_data['warp']['X'],
        y_train=pitcher_data['warp']['y'],
        groups_train=pitcher_data['warp']['years'],
        metric_type='warp',
        player_type='pitcher',
        holdout_validation=False
    )

    # Train pitcher WAR model
    ensemble.train_ensemble(
        X_train=pitcher_data['war']['X'],
        y_train=pitcher_data['war']['y'],
        groups_train=pitcher_data['war']['years'],
        metric_type='war',
        player_type='pitcher',
        holdout_validation=False
    )

    ensemble.is_trained = True

    # Test with real feature vectors
    print("\nTesting with real player features...")

    # Skubal features (from previous output)
    skubal_features = np.array([4.648074, 30.278884, 8.7719298, 72.0, 2.4, 0.0, 50.0, 0.0, 0.0, 52.24428006177364])
    ohtani_features = np.array([10.357814999999999, 31.450094, 16.6666666, 72.0, 2.4, 0.0, 50.0, 0.0, 0.0, 52.90603041509424])

    print("Testing predictions...")
    try:
        skubal_pred = ensemble.predict_ensemble(skubal_features, 'war', 'pitcher')
        ohtani_pred = ensemble.predict_ensemble(ohtani_features, 'war', 'pitcher')

        print(f"Skubal WAR prediction: {skubal_pred['ensemble']:.3f}")
        print(f"Ohtani WAR prediction: {ohtani_pred['ensemble']:.3f}")

        # Check if predictions are reasonable
        if abs(skubal_pred['ensemble']) < 50 and abs(ohtani_pred['ensemble']) < 50:
            print("SUCCESS: Predictions are reasonable!")
        else:
            print("WARNING: Still getting extreme predictions")

    except Exception as e:
        print(f"ERROR during prediction test: {e}")

    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"ensemble_models_10feat_{timestamp}.pkl"

    print(f"\nSaving model to: {model_path}")
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        print("Model saved successfully!")
    except Exception as e:
        print(f"ERROR saving model: {e}")
        return False

    # Update current model
    try:
        import shutil
        shutil.copy2(model_path, 'ensemble_models.pkl')
        print("Current model updated!")
    except Exception as e:
        print(f"ERROR updating current model: {e}")
        return False

    return True

if __name__ == "__main__":
    success = quick_retrain()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)