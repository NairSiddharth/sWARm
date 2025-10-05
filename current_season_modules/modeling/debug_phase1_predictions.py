"""
Phase 1 Debug Script - Comprehensive Prediction Validation.

This script validates Phase 1 (rate-based three-path ensemble) predictions
by tracing the complete pipeline from features to final WAR predictions.

Usage:
    python -m current_season_modules.modeling.debug_phase1_predictions
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
from common_modules.logging import get_logger

logger = get_logger(__name__)

# Constants
QUALIFIED_IP = 162.0


def load_2024_holdout_data(pitcher_data_dict):
    """
    Extract 2024 holdout data from training dataset.

    Args:
        pitcher_data_dict: Full pitcher training data

    Returns:
        DataFrame: 2024 holdout pitcher data
    """
    if 'war' not in pitcher_data_dict:
        return None

    data = pitcher_data_dict['war']
    years = data['years']

    # Handle different year formats
    if isinstance(years, list):
        years_array = np.array(years)
    elif hasattr(years, 'values'):
        years_array = years.values
    else:
        years_array = years

    # Filter 2024 data
    mask_2024 = years_array == 2024
    holdout_indices = np.where(mask_2024)[0]

    # Convert to lists/arrays for safe indexing
    names_list = list(data['names'].values) if hasattr(data['names'], 'values') else list(data['names'])
    y_list = list(data['y'].values) if hasattr(data['y'], 'values') else list(data['y'])

    # Get IP from X DataFrame
    X_df = data['X']
    if hasattr(X_df, 'values'):
        # DataFrame with index - use iloc
        ip_list = list(X_df['IP'].values)
    else:
        ip_list = list(X_df['IP'])

    holdout_df = pd.DataFrame()
    holdout_df['Name'] = [names_list[i] for i in holdout_indices]
    holdout_df['WAR'] = [y_list[i] for i in holdout_indices]
    holdout_df['IP'] = [ip_list[i] for i in holdout_indices]

    return holdout_df


def load_2025_production_data():
    """
    Load 2025 first-half production data from CSV.

    Returns:
        DataFrame: 2025 pitcher data with actual WAR
    """
    csv_path = "MLB Player Data/FanGraphs_Data/pitchers/" \
               "fangraphs_pitchers_2025_firsthalf_standard.csv"

    if not Path(csv_path).exists():
        logger.warning(f"2025 CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Add actual WAR (will be filled manually for test set)
    # Using known values from FanGraphs
    actual_wars = {
        'Tarik Skubal': 4.79,
        'Garrett Crochet': 4.34,
        'Paul Skenes': 4.03,
        'Zack Wheeler': 3.71,
        'Logan Webb': 3.46
    }

    df['Actual_WAR'] = df['Name'].map(actual_wars)

    return df


def validate_pitcher_features(player_name, player_data, feature_vector):
    """
    Validate features passed to Phase 1 ensemble.

    Args:
        player_name: Player's name
        player_data: DataFrame row with player stats
        feature_vector: Numpy array of features for prediction

    Returns:
        dict: Validation results with status
    """
    print(f"\n{'=' * 80}")
    print(f"FEATURE VALIDATION - {player_name}")
    print(f"{'=' * 80}\n")

    # Expected feature ranges (reasonable MLB values)
    feature_ranges = {
        'BB%': (0, 20),
        'K%': (10, 45),
        'K-BB%': (-10, 40),
        'ERA': (0, 8),
        'damage_control_ratio': (0, 1),
        'Opportunity_Success': (0, 1),
        'Contact_Quality_Index': (20, 80),
        'HBP%': (0, 5),
        'Statcast_Launch_Quality_Index': (20, 80)
    }

    table = PrettyTable()
    table.field_names = ['Feature', 'Value', 'Range', 'Status']
    table.align['Feature'] = 'l'
    table.align['Value'] = 'r'
    table.align['Range'] = 'l'
    table.align['Status'] = 'c'

    validation_passed = True

    for i, feature_name in enumerate(PITCHER_RATE_FEATURES):
        value = feature_vector[i] if i < len(feature_vector) else None

        if value is None or np.isnan(value):
            status = '✗ MISSING'
            validation_passed = False
        else:
            min_val, max_val = feature_ranges.get(feature_name, (0, 100))
            range_str = f"{min_val}-{max_val}"

            if min_val <= value <= max_val:
                status = '✓ OK'
            else:
                status = '⚠ OUT OF RANGE'
                validation_passed = False

        table.add_row([
            feature_name,
            f"{value:.2f}" if value is not None else 'N/A',
            range_str,
            status
        ])

    print(table)
    print()

    return {
        'player_name': player_name,
        'feature_count': len(feature_vector),
        'expected_count': len(PITCHER_RATE_FEATURES),
        'validation_passed': validation_passed
    }


def validate_role_classification(player_name, player_data, phase1_result):
    """
    Validate pitcher role classification and routing.

    Args:
        player_name: Player's name
        player_data: DataFrame row with G, GS
        phase1_result: Phase 1 prediction result

    Returns:
        dict: Role validation with routing confirmation
    """
    print(f"\n{'=' * 80}")
    print(f"ROLE CLASSIFICATION - {player_name}")
    print(f"{'=' * 80}\n")

    G = player_data.get('G', 0)
    GS = player_data.get('GS', 0)
    gs_g_ratio = GS / G if G > 0 else 0

    # Expected role from GS/G
    if gs_g_ratio > 0.7:
        expected_role = 'starter'
    elif gs_g_ratio < 0.1:
        expected_role = 'reliever'
    else:
        expected_role = 'mixed'

    actual_role = phase1_result.get('role', 'unknown')

    table = PrettyTable()
    table.field_names = ['Metric', 'Value', 'Status']
    table.align['Metric'] = 'l'
    table.align['Value'] = 'r'
    table.align['Status'] = 'c'

    table.add_row(['Games (G)', G, '✓ OK' if G > 0 else '✗ MISSING'])
    table.add_row(['Games Started (GS)', GS, '✓ OK' if GS >= 0 else '✗ MISSING'])
    table.add_row(['GS/G Ratio', f"{gs_g_ratio:.3f}", '✓ OK'])
    table.add_row(['Expected Role', expected_role, ''])
    table.add_row(['Actual Role', actual_role,
                   '✓ MATCH' if actual_role == expected_role else '✗ MISMATCH'])

    print(table)
    print()

    return {
        'player_name': player_name,
        'G': G,
        'GS': GS,
        'gs_g_ratio': gs_g_ratio,
        'expected_role': expected_role,
        'actual_role': actual_role,
        'role_match': actual_role == expected_role
    }


def trace_prediction_pipeline(player_name, player_data, phase1_result, actual_war):
    """
    Trace complete prediction from features to final WAR.

    Args:
        player_name: Player's name
        player_data: DataFrame row with all stats
        phase1_result: Phase 1 prediction result
        actual_war: Actual WAR from FanGraphs

    Returns:
        dict: Step-by-step prediction breakdown
    """
    print(f"\n{'=' * 80}")
    print(f"PREDICTION PIPELINE - {player_name}")
    print(f"{'=' * 80}\n")

    IP = player_data.get('IP', 0)
    war_per_162 = phase1_result.get('war_per_162', 0)
    predicted_war = phase1_result.get('current_war', 0)
    denorm_factor = IP / QUALIFIED_IP if IP > 0 else 0

    # Calculate expected denormalized WAR manually
    expected_denorm = war_per_162 * denorm_factor

    error = predicted_war - actual_war if actual_war else None
    error_pct = (error / actual_war * 100) if actual_war and actual_war != 0 else None

    table = PrettyTable()
    table.field_names = ['Step', 'Value', 'Formula']
    table.align['Step'] = 'l'
    table.align['Value'] = 'r'
    table.align['Formula'] = 'l'

    table.add_row(['Input IP', f"{IP:.1f}", 'From CSV'])
    table.add_row(['Rate Prediction', f"{war_per_162:.3f}",
                   'Starter/Reliever model output'])
    table.add_row(['Denorm Factor', f"{denorm_factor:.3f}",
                   f"{IP:.1f} / {QUALIFIED_IP:.1f}"])
    table.add_row(['Expected Denorm', f"{expected_denorm:.3f}",
                   f"{war_per_162:.3f} * {denorm_factor:.3f}"])
    table.add_row(['Predicted WAR', f"{predicted_war:.3f}",
                   'Phase 1 output'])

    if actual_war:
        table.add_row(['Actual WAR', f"{actual_war:.2f}", 'FanGraphs first half'])
        table.add_row(['Error', f"{error:+.2f} ({error_pct:+.0f}%)",
                       f"{predicted_war:.3f} - {actual_war:.2f}"])

    print(table)
    print()

    return {
        'player_name': player_name,
        'IP': IP,
        'war_per_162': war_per_162,
        'denorm_factor': denorm_factor,
        'expected_denorm': expected_denorm,
        'predicted_war': predicted_war,
        'actual_war': actual_war,
        'error': error,
        'error_pct': error_pct
    }


def compare_test_vs_production(holdout_df, production_df):
    """
    Compare 2024 holdout data to 2025 production data.

    Args:
        holdout_df: 2024 holdout dataset
        production_df: 2025 first-half dataset

    Returns:
        dict: Statistical comparison of datasets
    """
    print(f"\n{'=' * 80}")
    print("DATA COMPARISON: 2024 Holdout vs 2025 Production")
    print(f"{'=' * 80}\n")

    table = PrettyTable()
    table.field_names = ['Metric', '2024 Test', '2025 Prod', 'Difference']
    table.align['Metric'] = 'l'
    table.align['2024 Test'] = 'r'
    table.align['2025 Prod'] = 'r'
    table.align['Difference'] = 'r'

    # Calculate statistics
    test_mean_ip = holdout_df['IP'].mean() if holdout_df is not None else 0
    prod_mean_ip = production_df['IP'].mean() if production_df is not None else 0

    test_count = len(holdout_df) if holdout_df is not None else 0
    prod_count = len(production_df) if production_df is not None else 0

    table.add_row(['Sample Size', test_count, prod_count, prod_count - test_count])
    table.add_row(['Mean IP', f"{test_mean_ip:.1f}", f"{prod_mean_ip:.1f}",
                   f"{prod_mean_ip - test_mean_ip:+.1f}"])

    # Calculate percentage of full season
    test_pct_full = (test_mean_ip / QUALIFIED_IP * 100) if test_mean_ip > 0 else 0
    prod_pct_full = (prod_mean_ip / QUALIFIED_IP * 100) if prod_mean_ip > 0 else 0

    table.add_row(['% Full Season', f"{test_pct_full:.0f}%", f"{prod_pct_full:.0f}%",
                   f"{prod_pct_full - test_pct_full:+.0f}%"])

    print(table)
    print()

    return {
        'test_mean_ip': test_mean_ip,
        'prod_mean_ip': prod_mean_ip,
        'test_pct_full': test_pct_full,
        'prod_pct_full': prod_pct_full
    }


def comprehensive_pitcher_test(pitcher_list, pitcher_ensemble,
                               production_df, gs_g_data):
    """
    Test Phase 1 on multiple 2025 pitchers.

    Args:
        pitcher_list: List of pitcher names
        pitcher_ensemble: Trained Phase 1 ensemble
        production_df: 2025 production DataFrame
        gs_g_data: Dict of (year, name) -> (G, GS)

    Returns:
        DataFrame: Comprehensive comparison table
    """
    print(f"\n{'=' * 80}")
    print("COMPREHENSIVE PITCHER COMPARISON - 2025 First Half")
    print(f"{'=' * 80}\n")

    results = []

    for pitcher_name in pitcher_list:
        # Find pitcher in data
        pitcher_row = production_df[production_df['Name'] == pitcher_name]

        if pitcher_row.empty:
            logger.warning(f"Pitcher not found: {pitcher_name}")
            continue

        pitcher_data = pitcher_row.iloc[0]

        # Get G and GS
        key = (2025, pitcher_name)
        if key in gs_g_data:
            G, GS = gs_g_data[key]
        else:
            G = pitcher_data.get('G', 0)
            GS = pitcher_data.get('GS', 0)

        IP = pitcher_data.get('IP', 0)
        actual_war = pitcher_data.get('Actual_WAR', None)

        # Get features (without IP)
        features = []
        for feat in PITCHER_RATE_FEATURES:
            features.append(pitcher_data.get(feat, 0))

        # Predict using Phase 1
        try:
            pred_result = pitcher_ensemble.predict(
                features=np.array(features),
                GS=int(GS),
                G=int(G),
                IP=float(IP),
                metric_type='war'
            )

            predicted_war = pred_result['current_war']
            error = predicted_war - actual_war if actual_war else None
            error_pct = (error / actual_war * 100) if actual_war and actual_war != 0 else None

            results.append({
                'Player': pitcher_name,
                'IP': IP,
                'GS': GS,
                'Actual': actual_war,
                'Predicted': predicted_war,
                'Error': error,
                'Error %': error_pct
            })

        except Exception as e:
            logger.error(f"Prediction failed for {pitcher_name}: {e}")
            continue

    # Create results table
    results_df = pd.DataFrame(results)

    table = PrettyTable()
    table.field_names = ['Player', 'IP', 'GS', 'Actual', 'Predicted', 'Error', 'Error %']
    table.align['Player'] = 'l'
    for col in ['IP', 'GS', 'Actual', 'Predicted', 'Error', 'Error %']:
        table.align[col] = 'r'

    for _, row in results_df.iterrows():
        table.add_row([
            row['Player'],
            f"{row['IP']:.0f}",
            row['GS'],
            f"{row['Actual']:.2f}" if row['Actual'] else 'N/A',
            f"{row['Predicted']:.3f}",
            f"{row['Error']:+.2f}" if row['Error'] else 'N/A',
            f"{row['Error %']:+.0f}%" if row['Error %'] else 'N/A'
        ])

    print(table)
    print()

    # Pattern analysis
    print("PATTERN ANALYSIS:")
    elite = results_df[results_df['Actual'] >= 4.0]
    mid_tier = results_df[(results_df['Actual'] >= 3.0) & (results_df['Actual'] < 4.0)]

    if not elite.empty:
        elite_avg_error_pct = elite['Error %'].mean()
        print(f"- Elite starters (>4.0 WAR): Avg error {elite_avg_error_pct:+.0f}%")

    if not mid_tier.empty:
        mid_avg_error_pct = mid_tier['Error %'].mean()
        print(f"- Mid-tier starters (3.0-4.0 WAR): Avg error {mid_avg_error_pct:+.0f}%")

    print()

    return results_df


def main():
    """Main debug function - runs all validation checks."""
    print("=" * 80)
    print("PHASE 1 DEBUG - COMPREHENSIVE VALIDATION")
    print("=" * 80)
    print()

    # Load training data and train Phase 1
    print("Loading training data (2016-2024)...")
    hitter_data, pitcher_data = prepare_data_for_kfold()

    # Load G/GS from CSV files
    print("Loading G/GS data from CSV files...")
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
        except Exception as e:
            continue

    print(f"Loaded G/GS data for {len(gs_g_data)} pitcher-seasons")

    # Prepare pitcher data for Phase 1
    print("Preparing pitcher data for Phase 1 training...")
    pitcher_dfs = {}

    for metric in ['war']:  # Just WAR for now
        if metric in pitcher_data:
            data = pitcher_data[metric]
            X_df = data['X']
            y_series = data['y']
            years_data = data['years']
            names_data = data['names']

            n_samples = len(X_df)

            # Convert years and names (handle nested structures)
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
            pitcher_df['WAR'] = y_series.values if hasattr(y_series, 'values') else y_series

            pitcher_dfs[metric] = pitcher_df

    # Train Phase 1
    print("\nTraining Phase 1 ensemble...")
    pitcher_ensemble = train_three_path_ensemble(pitcher_dfs, holdout_year=None)
    print("✓ Phase 1 ensemble trained")

    # Load test data
    holdout_df = load_2024_holdout_data(pitcher_data)
    production_df = load_2025_production_data()

    # Run comparison
    if holdout_df is not None and production_df is not None:
        compare_test_vs_production(holdout_df, production_df)

    # Run comprehensive pitcher test
    test_pitchers = [
        'Tarik Skubal',
        'Garrett Crochet',
        'Paul Skenes',
        'Zack Wheeler',
        'Logan Webb'
    ]

    if production_df is not None:
        comprehensive_pitcher_test(test_pitchers, pitcher_ensemble,
                                  production_df, gs_g_data)

    print("=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
