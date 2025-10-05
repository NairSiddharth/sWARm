"""
Validate notebook predictions match debug script predictions.

This script simulates the exact notebook prediction flow and compares
it to the debug script's approach to identify where they diverge.

Usage:
    python -m current_season_modules.modeling.validate_notebook_predictions
"""

# Standard library imports
import sys
from pathlib import Path
import glob

# Third-party imports
import numpy as np
import pandas as pd
from prettytable import PrettyTable

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Local imports
from current_season_modules.modeling.pitcher_roles_ensemble_standalone import (
    train_three_path_ensemble,
    PITCHER_RATE_FEATURES
)
from current_season_modules.modeling import prepare_data_for_kfold
from common_modules.pitcher_workload_calculator import calculate_pitcher_projections
from common_modules.logging import get_logger

logger = get_logger(__name__)


def load_pitcher_data_and_ensemble():
    """
    Load training data and create Phase 1 ensemble (like notebook Cell 6).

    Returns:
        tuple: (pitcher_data_dict, pitcher_ensemble, hitter_ensemble, ensemble_predictor)
    """
    print("Loading training data and creating ensembles...")
    print("=" * 80)

    # Load training data
    hitter_data, pitcher_data = prepare_data_for_kfold()

    # Load G/GS from CSV files (like Cell 6)
    gs_g_data = {}
    csv_pattern = "MLB Player Data/FanGraphs_Data/pitchers/fangraphs_pitchers_*_standard.csv"
    csv_files = glob.glob(csv_pattern)

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            filename_parts = csv_file.split('_')
            year_str = filename_parts[-2]
            try:
                year = int(year_str)
            except ValueError:
                continue

            if 'G' in df.columns and 'GS' in df.columns and 'Name' in df.columns:
                for _, row in df.iterrows():
                    name = row['Name']
                    gs_g_data[(year, name)] = (row['G'], row['GS'])
        except Exception:
            continue

    print(f"Loaded G/GS data for {len(gs_g_data)} pitcher-seasons")

    # Prepare pitcher data for Phase 1 (like Cell 6)
    pitcher_dfs = {}
    for metric in ['war', 'warp']:
        if metric in pitcher_data:
            data = pitcher_data[metric]
            X_df = data['X']
            y_series = data['y']
            years_data = data['years']
            names_data = data['names']

            n_samples = len(X_df)

            # Handle nested structures
            if isinstance(years_data, list) and len(years_data) > 0:
                years_list = [item[0] if isinstance(item, (list, tuple)) else item
                             for item in years_data]
            elif hasattr(years_data, 'values'):
                years_list = list(years_data.values)
            else:
                years_list = [2024] * n_samples

            if isinstance(names_data, list) and len(names_data) > 0:
                names_list = [item[0] if isinstance(item, (list, tuple)) else item
                             for item in names_data]
            elif hasattr(names_data, 'values'):
                names_list = list(names_data.values)
            else:
                names_list = ["Unknown"] * n_samples

            # Get G and GS
            G_array = np.zeros(n_samples)
            GS_array = np.zeros(n_samples)

            for i in range(n_samples):
                year = int(years_list[i])
                name = str(names_list[i])
                key = (year, name)

                if key in gs_g_data:
                    G_array[i], GS_array[i] = gs_g_data[key]
                else:
                    G_array[i] = 20
                    GS_array[i] = 0

            # Create DataFrame
            pitcher_df = X_df.copy()
            pitcher_df['Season'] = years_list
            pitcher_df['G'] = G_array
            pitcher_df['GS'] = GS_array
            pitcher_df[metric.upper()] = y_series.values if hasattr(y_series, 'values') else y_series

            pitcher_dfs[metric] = pitcher_df

    # Train Phase 1 ensemble
    pitcher_ensemble = train_three_path_ensemble(pitcher_dfs, holdout_year=None)
    print("[OK] Phase 1 pitcher ensemble trained\n")

    # Create EnsembleWrapper (like Cell 6)
    class EnsembleWrapper:
        def __init__(self, hitter_ens, pitcher_ens):
            self.hitter_ensemble = hitter_ens
            self.pitcher_ensemble = pitcher_ens

        def predict_ensemble(self, features, metric_type, player_type):
            if player_type == 'pitcher' and self.pitcher_ensemble:
                return {'ensemble': 0.0}  # Dummy - should be overridden
            return {'ensemble': 0.0}

    ensemble_predictor = EnsembleWrapper(None, pitcher_ensemble)

    return pitcher_data, pitcher_ensemble, ensemble_predictor, gs_g_data


def simulate_notebook_prediction(player_data, player_feature_vector,
                                 ensemble_predictor, gs_g_data):
    """
    Simulate exact notebook prediction flow (Cell 12).

    Args:
        player_data: DataFrame row with player stats
        player_feature_vector: Feature array (10 features including IP)
        ensemble_predictor: EnsembleWrapper object
        gs_g_data: Dict of (year, name) -> (G, GS)

    Returns:
        dict: Prediction results + diagnostic info
    """
    print("\n[NOTEBOOK PATH]")
    print("-" * 80)

    # Get G and GS (Cell 12 might not have this)
    player_name = player_data.get('Name', 'Unknown')
    key = (2025, player_name)

    if key in gs_g_data:
        G, GS = gs_g_data[key]
        player_data_copy = player_data.copy()
        player_data_copy['G'] = G
        player_data_copy['GS'] = GS
    else:
        player_data_copy = player_data

    print(f"Calling: calculate_pitcher_projections()")
    print(f"  - Feature count: {len(player_feature_vector)}")
    print(f"  - Features include IP: {player_feature_vector[0]:.1f}")
    print(f"  - ensemble_predictor type: {type(ensemble_predictor).__name__}")

    # Call like Cell 12 does
    try:
        result = calculate_pitcher_projections(
            player_data_copy,
            ensemble_predictor,
            player_feature_vector,
            total_remaining_games=None
        )

        current_war = result.get('current_war', 0)
        print(f"  - Prediction: {current_war:.3f} WAR")

        return {
            'method': 'calculate_pitcher_projections',
            'feature_count': len(player_feature_vector),
            'includes_ip': True,
            'prediction': current_war,
            'full_result': result
        }

    except Exception as e:
        print(f"  - ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'method': 'calculate_pitcher_projections',
            'error': str(e)
        }


def simulate_debug_prediction(player_data, pitcher_ensemble, gs_g_data):
    """
    Simulate debug script prediction flow (what works correctly).

    Args:
        player_data: DataFrame row with player stats
        pitcher_ensemble: ThreePathPitcherEnsemble object
        gs_g_data: Dict of (year, name) -> (G, GS)

    Returns:
        dict: Prediction results + diagnostic info
    """
    print("\n[DEBUG SCRIPT PATH]")
    print("-" * 80)

    player_name = player_data.get('Name', 'Unknown')
    key = (2025, player_name)

    if key in gs_g_data:
        G, GS = gs_g_data[key]
    else:
        G = player_data.get('G', 0)
        GS = player_data.get('GS', 0)

    IP = player_data.get('IP', 0)

    # Extract features WITHOUT IP (like debug script)
    features = []
    for feat in PITCHER_RATE_FEATURES:
        features.append(player_data.get(feat, 0))

    print(f"Calling: pitcher_ensemble.predict()")
    print(f"  - Feature count: {len(features)} (no IP)")
    print(f"  - G={G}, GS={GS}, IP={IP:.1f}")

    try:
        result = pitcher_ensemble.predict(
            features=np.array(features),
            GS=int(GS),
            G=int(G),
            IP=float(IP),
            metric_type='war'
        )

        current_war = result['current_war']
        print(f"  - Prediction: {current_war:.3f} WAR")

        return {
            'method': 'pitcher_ensemble.predict',
            'feature_count': len(features),
            'includes_ip': False,
            'prediction': current_war,
            'full_result': result
        }

    except Exception as e:
        print(f"  - ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'method': 'pitcher_ensemble.predict',
            'error': str(e)
        }


def compare_predictions(player_name, notebook_result, debug_result):
    """
    Compare both approaches and identify divergence.

    Args:
        player_name: Player's name
        notebook_result: Result from notebook path
        debug_result: Result from debug path

    Returns:
        dict: Comparison summary
    """
    print(f"\n{'=' * 80}")
    print(f"PREDICTION COMPARISON - {player_name}")
    print(f"{'=' * 80}\n")

    table = PrettyTable()
    table.field_names = ['Metric', 'Notebook Path', 'Debug Path', 'Match']
    table.align['Metric'] = 'l'
    table.align['Notebook Path'] = 'l'
    table.align['Debug Path'] = 'l'
    table.align['Match'] = 'c'

    # Method
    table.add_row([
        'Method Called',
        notebook_result.get('method', 'N/A'),
        debug_result.get('method', 'N/A'),
        'OK' if notebook_result.get('method') == debug_result.get('method') else 'DIFF'
    ])

    # Feature count
    nb_feat = notebook_result.get('feature_count', 0)
    db_feat = debug_result.get('feature_count', 0)
    table.add_row([
        'Feature Count',
        nb_feat,
        db_feat,
        'OK' if nb_feat == db_feat else 'DIFF'
    ])

    # IP included
    nb_ip = notebook_result.get('includes_ip', False)
    db_ip = debug_result.get('includes_ip', False)
    table.add_row([
        'Includes IP',
        'Yes' if nb_ip else 'No',
        'Yes' if db_ip else 'No',
        'OK' if nb_ip == db_ip else 'DIFF'
    ])

    # Prediction
    nb_pred = notebook_result.get('prediction', 0)
    db_pred = debug_result.get('prediction', 0)
    pred_diff = abs(nb_pred - db_pred) if nb_pred and db_pred else 0
    table.add_row([
        'WAR Prediction',
        f"{nb_pred:.3f}" if nb_pred else 'ERROR',
        f"{db_pred:.3f}" if db_pred else 'ERROR',
        'OK' if pred_diff < 0.01 else 'DIFF'
    ])

    print(table)
    print()

    # Diagnosis
    print("DIAGNOSIS:")
    if pred_diff < 0.01:
        print("[OK] Predictions MATCH - No divergence detected")
    else:
        print(f"[FAIL] Predictions DIVERGE by {pred_diff:.3f} WAR")

        if nb_feat != db_feat:
            print(f"  -> CAUSE: Feature count mismatch ({nb_feat} vs {db_feat})")

        if nb_ip != db_ip:
            print(f"  -> CAUSE: IP handling differs (notebook includes it, debug excludes it)")

        if notebook_result.get('method') != debug_result.get('method'):
            print(f"  -> CAUSE: Different methods called")

        if 'error' in notebook_result:
            print(f"  -> ERROR in notebook path: {notebook_result['error']}")

        if 'error' in debug_result:
            print(f"  -> ERROR in debug path: {debug_result['error']}")

    print()

    return {
        'player_name': player_name,
        'diverges': pred_diff >= 0.01,
        'prediction_diff': pred_diff,
        'notebook_prediction': nb_pred,
        'debug_prediction': db_pred
    }


def main():
    """Main validation function."""
    print("=" * 80)
    print("NOTEBOOK vs DEBUG SCRIPT VALIDATION")
    print("=" * 80)
    print()

    # Load data and ensembles
    pitcher_data, pitcher_ensemble, ensemble_predictor, gs_g_data = load_pitcher_data_and_ensemble()

    # Load 2025 production data
    csv_path = "MLB Player Data/FanGraphs_Data/pitchers/fangraphs_pitchers_2025_firsthalf_standard.csv"

    if not Path(csv_path).exists():
        print(f"ERROR: {csv_path} not found")
        return

    production_df = pd.read_csv(csv_path)

    # Test on Tarik Skubal
    test_player = 'Tarik Skubal'
    player_row = production_df[production_df['Name'] == test_player]

    if player_row.empty:
        print(f"ERROR: {test_player} not found in data")
        return

    player_data = player_row.iloc[0]

    # Create feature vector (10 features including IP - like notebook)
    player_feature_vector = np.array([
        player_data.get('IP', 0),
        player_data.get('BB%', 0),
        player_data.get('K%', 0),
        player_data.get('K-BB%', 0),
        player_data.get('ERA', 0),
        player_data.get('damage_control_ratio', 0),
        player_data.get('Opportunity_Success', 0),
        player_data.get('Contact_Quality_Index', 0),
        player_data.get('HBP%', 0),
        player_data.get('Statcast_Launch_Quality_Index', 0)
    ])

    # Simulate both paths
    notebook_result = simulate_notebook_prediction(
        player_data, player_feature_vector, ensemble_predictor, gs_g_data
    )

    debug_result = simulate_debug_prediction(
        player_data, pitcher_ensemble, gs_g_data
    )

    # Compare
    comparison = compare_predictions(test_player, notebook_result, debug_result)

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if comparison['diverges']:
        print(f"[FAIL] VALIDATION FAILED")
        print(f"  Notebook predicted: {comparison['notebook_prediction']:.3f} WAR")
        print(f"  Debug predicted: {comparison['debug_prediction']:.3f} WAR")
        print(f"  Difference: {comparison['prediction_diff']:.3f} WAR")
        print()
        print("ACTION REQUIRED: Fix notebook to match debug script path")
    else:
        print(f"[PASS] VALIDATION PASSED")
        print(f"  Both paths predict: {comparison['debug_prediction']:.3f} WAR")
        print()
        print("No action needed - notebook already aligned")

    print()


if __name__ == "__main__":
    main()
