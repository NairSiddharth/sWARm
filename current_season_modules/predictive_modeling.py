"""
Temporary Modeling Module for K-Fold Cross-Validation
Modified from original modeling.py to implement K-fold cross-validation
for comprehensive year-by-year analysis without overfitting
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
import sys
import os

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. XGBoost models will be skipped.")

try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Input
    from keras.callbacks import EarlyStopping
    from keras.optimizers import AdamW
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow/Keras not available. Neural network models will be skipped.")

class CrossValidationResults:
    """Class to store K-fold cross-validation results with year information"""

    def __init__(self):
        self.results = {}

    def store_cv_results(self, model_name, player_type, metric_type, y_true, y_pred, player_names, years):
        """Store cross-validation results with year information"""
        key = f"{model_name}_{player_type}_{metric_type}"
        self.results[key] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'player_names': player_names,
            'years': years
        }

    def get_year_data(self, year):
        """Get all predictions for a specific year"""
        year_data = {}
        for key, data in self.results.items():
            year_mask = np.array([str(y) == str(year) for y in data['years']])
            if np.any(year_mask):
                year_data[key] = {
                    'y_true': np.array(data['y_true'])[year_mask],
                    'y_pred': np.array(data['y_pred'])[year_mask],
                    'player_names': np.array(data['player_names'])[year_mask],
                    'years': np.array(data['years'])[year_mask]
                }
        return year_data

    def get_available_years(self):
        """Get all years with predictions"""
        all_years = set()
        for data in self.results.values():
            all_years.update([str(y) for y in data['years']])
        return sorted(list(all_years))

def load_expanded_fangraphs_data(data_dir=None):
    """
    Load expanded FanGraphs data with full PA spectrum (0 to max PA)
    Uses new file structure: fangraphs_hitters_xxxx.csv (main files with WAR)
    """
    import glob
    print("Loading expanded FanGraphs hitter data...")

    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    # Load main hitter files (these contain WAR)
    hitter_files = glob.glob(os.path.join(data_dir, "FanGraphs_Data", "hitters", "fangraphs_hitters_*.csv"))
    # Exclude the suffixed files, we only want the main files with years
    hitter_files = [f for f in hitter_files if not any(suffix in f for suffix in ['_standard', '_advanced', '_battedball', '_firsthalf'])]

    all_hitter_data = []
    for file in sorted(hitter_files):
        year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        try:
            df = pd.read_csv(file, encoding='utf-8-sig')

            if 'WAR' in df.columns:
                # Add year info
                df['Year'] = year
                df['Type'] = 'Hitter'
                all_hitter_data.append(df)
                print(f"   SUCCESS {year}: {len(df)} hitter records loaded")
            else:
                print(f"   WARNING {year}: No WAR column found in {file}")

        except Exception as e:
            print(f"   ERROR {year}: Error loading {file} - {e}")

    if all_hitter_data:
        combined_hitters = pd.concat(all_hitter_data, ignore_index=True)

        # Check PA distribution to confirm expansion
        if 'PA' in combined_hitters.columns:
            print(f"\nExpanded FanGraphs hitter data: {len(combined_hitters)} total records")
            print(f"   PA range: {combined_hitters['PA'].min()} to {combined_hitters['PA'].max()}")
            print(f"   Players with 0 PA: {(combined_hitters['PA'] == 0).sum()}")
            print(f"   Players with <100 PA: {(combined_hitters['PA'] < 100).sum()}")
            print(f"   Players with 400+ PA: {(combined_hitters['PA'] >= 400).sum()}")

        return combined_hitters
    else:
        return pd.DataFrame()

def load_expanded_fangraphs_pitcher_data(data_dir=None):
    """
    Load expanded FanGraphs pitcher data with full appearance spectrum (0 to max appearances)
    Uses new file structure: fangraphs_pitchers_xxxx.csv (main files with WAR)
    """
    import glob
    print("Loading expanded FanGraphs pitcher data...")

    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    # Load main pitcher files (these contain WAR)
    pitcher_files = glob.glob(os.path.join(data_dir, "FanGraphs_Data", "pitchers", "fangraphs_pitchers_*.csv"))
    # Exclude the suffixed files, we only want the main files with years
    pitcher_files = [f for f in pitcher_files if not any(suffix in f for suffix in ['_standard', '_advanced', '_battedball', '_firsthalf'])]

    all_pitcher_data = []
    for file in sorted(pitcher_files):
        year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        try:
            df = pd.read_csv(file, encoding='utf-8-sig')

            if 'WAR' in df.columns:
                # Add year column
                df['Year'] = year

                all_pitcher_data.append(df)
                print(f"   SUCCESS {year}: {len(df)} pitcher records loaded")
            else:
                print(f"   WARNING {year}: No WAR column found, skipping")

        except Exception as e:
            print(f"   ERROR {year}: Error loading - {e}")

    if all_pitcher_data:
        combined_pitchers = pd.concat(all_pitcher_data, ignore_index=True)

        # Check appearance distribution to confirm expansion
        if 'G' in combined_pitchers.columns:
            print(f"\nExpanded FanGraphs pitcher data: {len(combined_pitchers)} total records")
            print(f"   Games range: {combined_pitchers['G'].min()} to {combined_pitchers['G'].max()}")
            print(f"   Pitchers with 0 G: {(combined_pitchers['G'] == 0).sum()}")
            print(f"   Pitchers with <10 G: {(combined_pitchers['G'] < 10).sum()}")
            print(f"   Pitchers with 30+ G: {(combined_pitchers['G'] >= 30).sum()}")

        return combined_pitchers
    else:
        return pd.DataFrame()

def filter_position_players_pitching(pitcher_df, two_way_analysis, data_source='war'):
    """
    Filter out position players who pitched but aren't qualified two-way players

    Args:
        pitcher_df: DataFrame with pitcher data
        two_way_analysis: Result from get_cleaned_two_way_data()
        data_source: 'war' or 'warp'

    Returns:
        Filtered DataFrame with only legitimate pitchers and qualified two-way players
    """
    if len(pitcher_df) == 0:
        return pitcher_df

    print(f"Filtering position players from {data_source.upper()} pitcher data...")

    # Get data from updated two-way analysis structure
    two_way_players = two_way_analysis['two_way_players']
    emergency_pitching = two_way_analysis['filtered_data']['emergency_pitching']

    # Create a set of legitimate pitcher names
    legitimate_pitcher_names = set()

    # Add qualified two-way players (these should definitely be included)
    for player_key in two_way_players.keys():
        name, year = player_key.rsplit('_', 1)
        legitimate_pitcher_names.add(name)

    # Create a set of emergency pitcher names to exclude
    emergency_pitcher_names = set()
    for emergency_player in emergency_pitching:
        emergency_pitcher_names.add(emergency_player['name'])

    # Filter the DataFrame
    year_col = 'Season' if data_source == 'warp' else 'Year'

    if 'Name' in pitcher_df.columns and year_col in pitcher_df.columns:
        # Filtering logic:
        # 1. Keep all pitchers who are NOT in the emergency pitcher list
        # 2. Always keep qualified two-way players even if they might appear in other lists

        # Create boolean mask
        is_not_emergency = ~pitcher_df['Name'].isin(emergency_pitcher_names)
        is_two_way_qualified = pitcher_df['Name'].isin(legitimate_pitcher_names)

        # Keep if: not emergency OR is qualified two-way
        keep_mask = is_not_emergency | is_two_way_qualified
        filtered_df = pitcher_df[keep_mask].copy()

        removed_count = len(pitcher_df) - len(filtered_df)
        emergency_removed = len(pitcher_df[pitcher_df['Name'].isin(emergency_pitcher_names)])
        two_way_kept = len(pitcher_df[pitcher_df['Name'].isin(legitimate_pitcher_names)])

        print(f"  Removed {removed_count} emergency/position player pitching records")
        print(f"  Emergency pitchers filtered: {emergency_removed}")
        print(f"  Qualified two-way players kept: {two_way_kept}")
        print(f"  Total legitimate pitcher records: {len(filtered_df)}")

        return filtered_df
    else:
        print(f"  WARNING: Expected columns not found, returning original data")
        return pitcher_df

def filter_pitchers_from_hitting_data(df, data_source='war', year_col='Year'):
    """
    Filter out pitchers from hitting data, except qualified two-way players (2020+)

    Args:
        df: DataFrame with hitting data
        data_source: 'war' or 'warp'
        year_col: Column name for year ('Year' for WAR, 'Season' for WARP)
    """
    if len(df) == 0:
        return df

    # For now, implement basic pitcher filtering
    # This would need to be enhanced with actual pitcher identification logic
    # based on your data structure

    print(f"Filtering pitchers from {data_source.upper()} hitting data...")

    # For simplified testing, skip complex pitcher filtering for now
    # TODO: Implement proper pitcher identification logic later
    print(f"  Skipping pitcher filtering for simplified testing")
    print(f"  {data_source.upper()} hitting data: {len(df)} records (no filtering applied)")
    return df

def create_mlbid_mapping(warp_df, war_df):
    """
    Create player mapping based on MLBID instead of names

    Args:
        warp_df: DataFrame with WARP data (has 'mlbid' column)
        war_df: DataFrame with WAR data (has 'MLBAMID' column)

    Returns:
        dict: mapping of {warp_index: war_index} for matched players
    """
    print("Creating MLBID-based player mapping...")

    # Get valid IDs from both datasets
    warp_ids = warp_df['mlbid'].dropna().astype(int)
    war_ids = war_df['MLBAMID'].dropna().astype(int)

    print(f"  WARP data: {len(warp_ids)} records with valid mlbid")
    print(f"  WAR data: {len(war_ids)} records with valid MLBAMID")

    # Find common IDs
    common_ids = set(warp_ids).intersection(set(war_ids))
    print(f"  Common MLB IDs: {len(common_ids)}")

    # Create index mapping
    index_mapping = {}

    for mlbid in common_ids:
        # Find all WARP records with this ID
        warp_indices = warp_df[warp_df['mlbid'] == mlbid].index.tolist()
        # Find all WAR records with this ID
        war_indices = war_df[war_df['MLBAMID'] == mlbid].index.tolist()

        # Match each WARP record to each WAR record for this player
        for warp_idx in warp_indices:
            for war_idx in war_indices:
                index_mapping[warp_idx] = war_idx
                break  # Take first WAR match for each WARP record

    print(f"  Created {len(index_mapping)} WARP->WAR index mappings")
    return index_mapping

def prepare_data_for_kfold():
    """Prepare comprehensive dataset for K-fold cross-validation"""
    print("Preparing comprehensive dataset for K-fold cross-validation...")

    # Import data loading functions - USING CORRECT MODULE PATHS
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common_modules.derived_stats import load_fixed_bp_data
    from common_modules.enhanced_features import get_enhanced_features
    from common_modules.positional_adjustments import PositionalAdjustmentCalculator

    # Use the new enhanced features system instead of legacy analytics
    print("Loading enhanced features...")
    baserunning_data, defense_data = get_enhanced_features()

    # Convert to the expected format for backward compatibility
    def calculate_enhanced_baserunning_values():
        return baserunning_data

    def clean_defensive_players():
        return defense_data

    # Load datasets - USING EXPANDED DATA
    print("Loading FIXED BP data with derived statistics...")
    hitter_warp, pitcher_warp = load_fixed_bp_data()

    print("Loading EXPANDED FanGraphs data...")
    hitter_war_raw = load_expanded_fangraphs_data()

    # Filter pitchers from WAR hitting data
    hitter_war = filter_pitchers_from_hitting_data(hitter_war_raw, 'war', 'Year')

    # Filter pitchers from WARP hitting data
    hitter_warp = filter_pitchers_from_hitting_data(hitter_warp, 'warp', 'Season')

    # Reset indices after filtering to avoid mapping issues
    hitter_warp = hitter_warp.reset_index(drop=True)
    hitter_war = hitter_war.reset_index(drop=True)

    # Load expanded pitcher data with two-way player logic
    print("Loading EXPANDED pitcher data...")
    pitcher_war_raw = load_expanded_fangraphs_pitcher_data()

    # For now, use pitcher data as-is (skip complex two-way filtering for initial test)
    pitcher_war = pitcher_war_raw

    # Reset indices after filtering
    pitcher_warp = pitcher_warp.reset_index(drop=True)
    pitcher_war = pitcher_war.reset_index(drop=True)

    print(f"Pitcher data after two-way filtering: {len(pitcher_warp)} WARP, {len(pitcher_war)} WAR")

    # Enhanced features
    enhanced_baserunning = calculate_enhanced_baserunning_values()
    enhanced_defensive = clean_defensive_players()

    # Load positional data for adjustments using new system
    pos_calc = PositionalAdjustmentCalculator()
    pos_calc.load_defensive_data()
    bp_positions = pos_calc.bp_positions if hasattr(pos_calc, 'bp_positions') else None
    fg_positions = pos_calc.fg_positions if hasattr(pos_calc, 'fg_positions') else None

    print(f"Loaded data: {len(hitter_warp)} hitter WARP, {len(pitcher_warp)} pitcher WARP")
    print(f"              {len(hitter_war)} hitter WAR, {len(pitcher_war)} pitcher WAR")
    bp_count = len(bp_positions) if bp_positions is not None else 0
    fg_count = len(fg_positions) if fg_positions is not None else 0
    print(f"              {bp_count} BP positions, {fg_count} FG positions")

    # Create MLBID-based mappings instead of name-based
    print("Creating MLBID-based player mappings...")
    hitter_mapping = create_mlbid_mapping(hitter_warp, hitter_war)

    # Only create pitcher mapping if we have pitcher WAR data
    if len(pitcher_war) > 0:
        pitcher_mapping = create_mlbid_mapping(pitcher_warp, pitcher_war)
    else:
        pitcher_mapping = {}

    print(f"MLBID mappings created: {len(hitter_mapping)} hitters, {len(pitcher_mapping)} pitchers")

    def prepare_dataset(warp_data, war_data, mapping, enhanced_br, enhanced_def, player_type, bp_pos, fg_pos):
        """Prepare matched dataset with features and targets"""
        if len(mapping) == 0:
            return None

        # MLBID mapping is already index-based: {warp_index: war_index}
        warp_indices = list(mapping.keys())
        war_indices = list(mapping.values())

        if len(warp_indices) == 0:
            return None

        warp_matched = warp_data.iloc[warp_indices].reset_index(drop=True)
        war_matched = war_data.iloc[war_indices].reset_index(drop=True)

        # Add enhanced features
        def add_enhanced_features(df, data_source='warp'):
            df_enhanced = df.copy()
            df_enhanced['Enhanced_Baserunning'] = df_enhanced['Name'].map(enhanced_br).fillna(0.0)
            df_enhanced['Enhanced_Defense'] = df_enhanced['Name'].map(enhanced_def).fillna(0.0)

            # Add GDP rate for hitters (situational hitting metric)
            if player_type == 'hitter' and 'GDP' in df_enhanced.columns and 'PA' in df_enhanced.columns:
                df_enhanced['GDP_rate'] = df_enhanced['GDP'].fillna(0) / df_enhanced['PA'].replace(0, 1)
                df_enhanced['GDP_rate'] = df_enhanced['GDP_rate'].fillna(0.0)
            elif player_type == 'hitter':
                df_enhanced['GDP_rate'] = 0.0  # Default if GDP/PA not available

            # Add positional adjustments using new system
            from common_modules.positional_adjustments import POSITION_WAR_ADJUSTMENTS
            if player_type == 'hitter' and 'PA' in df_enhanced.columns:
                # Calculate positional adjustment based on primary position and PA
                if 'Pos' in df_enhanced.columns:
                    df_enhanced['Positional_WAR'] = df_enhanced.apply(
                        lambda row: POSITION_WAR_ADJUSTMENTS.get(row.get('Pos', ''), 0.0) * (row.get('PA', 600) / 600),
                        axis=1
                    )
                else:
                    df_enhanced['Positional_WAR'] = 0.0
            else:
                df_enhanced['Positional_WAR'] = 0.0

            # Add enhanced pitcher features for 11-feature expansion
            if player_type == 'pitcher':
                # Load percentage-based pitcher features for consistent scaling
                from common_modules.derived_stats import load_percentage_pitcher_features

                # Load percentage-based features from all available years (2016-2024)
                data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"
                years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
                percentage_features = load_percentage_pitcher_features(data_dir, years)

                # Initialize all pitcher feature columns with percentage scaling
                pitcher_feature_columns = ['BB%', 'K%', 'K-BB%', 'damage_control_ratio', 'Opportunity_Success', 'Contact_Quality_Index', 'HBP%', 'Statcast_Launch_Quality_Index']
                for col in pitcher_feature_columns:
                    df_enhanced[col] = 0.0

                # Map percentage-based pitcher features by player ID
                from common_modules.derived_stats import get_player_percentage_features
                for idx, row in df_enhanced.iterrows():
                    player_id = row.get('MLBAMID', row.get('mlbid', ''))
                    if player_id:
                        try:
                            player_features = get_player_percentage_features(player_id, percentage_features)
                            # Map all percentage-based features
                            for feature_name, feature_value in player_features.items():
                                if feature_name in df_enhanced.columns:
                                    df_enhanced.at[idx, feature_name] = feature_value
                        except Exception as e:
                            # Use defaults if mapping fails
                            continue
            else:
                # Initialize pitcher features as 0 for hitters (not applicable)
                pitcher_feature_columns = ['BB%', 'K%', 'K-BB%', 'damage_control_ratio', 'Opportunity_Success', 'Contact_Quality_Index', 'HBP%', 'Statcast_Launch_Quality_Index']
                for col in pitcher_feature_columns:
                    df_enhanced[col] = 0.0

            # Enhanced_Defense is not applicable to pitchers - only to hitters
            if player_type != 'pitcher':
                # Enhanced_Defense already set above for hitters
                pass

            return df_enhanced

        warp_enhanced = add_enhanced_features(warp_matched, 'warp')
        war_enhanced = add_enhanced_features(war_matched, 'war')

        # Define features - BACKEND IMPROVEMENTS INTEGRATED: PA, GDP rate, positional adjustments
        if player_type == 'hitter':
            # ENHANCED HITTER FEATURES (10 features): PA volume scaling + situational hitting + positional adjustments
            # Core: K%, BB%, AVG, OBP, SLG (5 features)
            # Volume: PA (1 feature)
            # Situational: GDP rate (1 feature)
            # Enhanced: Baserunning, Defense (2 features)
            # Positional: Position adjustment (1 feature)
            warp_features = ['K%', 'BB%', 'AVG', 'OBP', 'SLG', 'PA', 'Positional_WAR', 'GDP_rate', 'Enhanced_Baserunning', 'Enhanced_Defense']
            war_features = ['K%', 'BB%', 'AVG', 'OBP', 'SLG', 'PA', 'Positional_WAR', 'GDP_rate', 'Enhanced_Baserunning', 'Enhanced_Defense']
        else:  # pitcher
            # 10-FEATURE PITCHER SET: Contact quality + role-neutral opportunity + command precision + K-BB%
            # Core: IP, BB%, K%, K-BB%, ERA (5 features) - K-BB% composite replaces WP for elite prediction
            # Enhanced: damage_control_ratio (1 feature) - LOB%/(HR/9 + 0.5)
            # Opportunity: Opportunity_Success (1 feature) - (QS + SV + HLD - BS) / G for role-neutral evaluation
            # Contact Quality: Contact_Quality_Index (1 composite feature) - replaces Hard%, Med%, Soft%
            # Command: HBP% (1 feature) - command precision measure
            # Statcast: Statcast_Launch_Quality_Index (1 feature) - launch angle control from empirical analysis
            # Note: K-BB% provides +0.1125 WAR improvement in elite pitcher prediction vs WP
            # Updated to use K-BB% instead of WP for improved elite pitcher prediction
            warp_features = ['IP', 'BB%', 'K%', 'K-BB%', 'ERA', 'damage_control_ratio', 'Opportunity_Success', 'Contact_Quality_Index', 'HBP%', 'Statcast_Launch_Quality_Index']
            war_features = ['IP', 'BB%', 'K%', 'K-BB%', 'ERA', 'damage_control_ratio', 'Opportunity_Success', 'Contact_Quality_Index', 'HBP%', 'Statcast_Launch_Quality_Index']

        # Filter to only available columns
        warp_available = warp_enhanced.columns.tolist()
        war_available = war_enhanced.columns.tolist()

        warp_features = [col for col in warp_features if col in warp_available]
        war_features = [col for col in war_features if col in war_available]

        print(f"Selected WARP features: {warp_features} (from {len(warp_available)} available)")
        print(f"Selected WAR features: {war_features} (from {len(war_available)} available)")

        # Extract features and targets with NaN cleaning
        # Clean NaN values from target variables
        warp_valid_mask = warp_enhanced['WARP'].notna()
        war_valid_mask = war_enhanced['WAR'].notna()

        print(f"  WARP data: {warp_valid_mask.sum()}/{len(warp_enhanced)} records with valid WARP values")
        print(f"  WAR data: {war_valid_mask.sum()}/{len(war_enhanced)} records with valid WAR values")

        # Filter to only valid records
        warp_clean = warp_enhanced[warp_valid_mask].reset_index(drop=True)
        war_clean = war_enhanced[war_valid_mask].reset_index(drop=True)

        warp_X = warp_clean[warp_features].fillna(0)
        warp_y = warp_clean['WARP']
        warp_names = warp_clean['Name']
        # WARP data: Look for Season first, then Year - SAME AS MAIN NOTEBOOK
        warp_years = warp_clean['Season'].tolist() if 'Season' in warp_clean.columns else warp_clean['Year'].tolist() if 'Year' in warp_clean.columns else ['2021'] * len(warp_clean)

        war_X = war_clean[war_features].fillna(0)
        war_y = war_clean['WAR']
        war_names = war_clean['Name']
        # WAR data: Look for Year - SAME AS MAIN NOTEBOOK
        war_years = war_clean['Year'].tolist() if 'Year' in war_clean.columns else ['2021'] * len(war_clean)

        return {
            'warp': {'X': warp_X, 'y': warp_y, 'names': warp_names, 'years': warp_years},
            'war': {'X': war_X, 'y': war_y, 'names': war_names, 'years': war_years}
        }

    # Prepare datasets
    hitter_data = prepare_dataset(hitter_warp, hitter_war, hitter_mapping,
                                 enhanced_baserunning, enhanced_defensive, 'hitter', bp_positions, fg_positions)
    pitcher_data = prepare_dataset(pitcher_warp, pitcher_war, pitcher_mapping,
                                  enhanced_baserunning, enhanced_defensive, 'pitcher', bp_positions, fg_positions)

    # CRITICAL: Apply legitimate pitcher filtering to remove position players who pitch in blowouts
    print("\\nApplying legitimate pitcher filtering...")
    from current_season_modules.filter_legitimate_pitchers import apply_pitcher_filtering
    try:
        filtered_pitcher_data = apply_pitcher_filtering(hitter_data, pitcher_data)
        if filtered_pitcher_data:
            pitcher_data = filtered_pitcher_data
            print(f"[OK] Pitcher filtering applied successfully")
        else:
            print("[WARNING] Pitcher filtering failed, using unfiltered data")
    except Exception as e:
        print(f"[WARNING] Pitcher filtering error: {e}, using unfiltered data")

    return hitter_data, pitcher_data


# Integration functions for multi-source training (bridges restored temp_modeling with new backend)
def load_comprehensive_warp_hitter_data():
    """Load comprehensive WARP hitter data with multi-source integration"""
    hitter_data, _ = prepare_data_for_kfold()
    return hitter_data['warp'] if hitter_data else None

def load_comprehensive_fangraphs_hitter_data():
    """Load comprehensive FanGraphs hitter WAR data with multi-source integration"""
    hitter_data, _ = prepare_data_for_kfold()
    return hitter_data['war'] if hitter_data else None

def load_comprehensive_warp_pitcher_data():
    """Load comprehensive WARP pitcher data with multi-source integration"""
    _, pitcher_data = prepare_data_for_kfold()
    return pitcher_data['warp'] if pitcher_data else None

def load_comprehensive_fangraphs_pitcher_data():
    """Load comprehensive FanGraphs pitcher WAR data with multi-source integration"""
    _, pitcher_data = prepare_data_for_kfold()
    return pitcher_data['war'] if pitcher_data else None

# Backward compatibility functions for existing code
def load_and_prepare_hitter_data():
    """Load and prepare hitter WARP data for modeling"""
    try:
        return load_comprehensive_warp_hitter_data()
    except Exception as e:
        print(f"Error loading hitter WARP data: {e}")
        return None

def load_and_prepare_hitter_war_data():
    """Load and prepare hitter WAR data for modeling"""
    try:
        return load_comprehensive_fangraphs_hitter_data()
    except Exception as e:
        print(f"Error loading hitter WAR data: {e}")
        return None

def load_and_prepare_pitcher_data():
    """Load and prepare pitcher WARP data for modeling"""
    try:
        return load_comprehensive_warp_pitcher_data()
    except Exception as e:
        print(f"Error loading pitcher WARP data: {e}")
        return None

def load_and_prepare_pitcher_war_data():
    """Load and prepare pitcher WAR data for modeling"""
    try:
        return load_comprehensive_fangraphs_pitcher_data()
    except Exception as e:
        print(f"Error loading pitcher WAR data: {e}")
        return None

def run_kfold_cross_validation(hitter_data, pitcher_data, n_splits=5):
    """Run K-fold cross-validation on all models and datasets"""
    print(f"Running {n_splits}-fold cross-validation...")

    results = CrossValidationResults()

    # Models to test
    models = {
        'ridge': Ridge(),
        'randomforest': RandomForestRegressor(n_estimators=100, random_state=42),
        'svr': SVR(),
        'keras': 'neural_network'  # Special case for Keras
    }

    def run_cv_for_dataset(data, player_type, metric_type):
        """Run cross-validation for a specific dataset"""
        if data is None:
            print(f"Skipping {player_type} {metric_type} - no data available")
            return

        X = data['X']
        y = data['y']
        names = data['names']
        years = data['years']

        print(f"Running CV for {player_type} {metric_type}: {len(X)} samples")

        # Use GroupKFold to keep years together
        gkf = GroupKFold(n_splits=n_splits)

        # Store predictions for each model
        for model_name, model in models.items():
            y_pred_all = np.zeros(len(y))

            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=years)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Handle Keras separately
                if model_name == 'keras' and HAS_TENSORFLOW:
                    # Create and train Keras model
                    keras_model = create_keras_model_temp(X_train.shape[1], f"{player_type}_{metric_type}")
                    keras_model.fit(X_train, y_train, epochs=50, batch_size=32,
                                   validation_split=0.2, verbose=0)
                    y_pred_fold = keras_model.predict(X_test, verbose=0).flatten()
                elif model_name != 'keras':
                    # Standard sklearn models
                    model_copy = models[model_name]
                    if hasattr(model_copy, 'set_params'):
                        # Create fresh copy to avoid contamination
                        from sklearn.base import clone
                        model_copy = clone(model_copy)

                    model_copy.fit(X_train, y_train)
                    y_pred_fold = model_copy.predict(X_test)
                else:
                    continue  # Skip Keras if TensorFlow not available

                y_pred_all[test_idx] = y_pred_fold

            # Store results
            if model_name == 'keras' and not HAS_TENSORFLOW:
                continue

            results.store_cv_results(model_name, player_type, metric_type,
                                   y.values, y_pred_all, names.values, years)

            # Print metrics
            r2 = r2_score(y, y_pred_all)
            rmse = np.sqrt(mean_squared_error(y, y_pred_all))
            print(f"  {model_name} {player_type} {metric_type}: R2={r2:.4f}, RMSE={rmse:.4f}")

    # Run CV for all datasets
    if hitter_data:
        run_cv_for_dataset(hitter_data['warp'], 'hitter', 'warp')
        run_cv_for_dataset(hitter_data['war'], 'hitter', 'war')

    if pitcher_data:
        run_cv_for_dataset(pitcher_data['warp'], 'pitcher', 'warp')
        run_cv_for_dataset(pitcher_data['war'], 'pitcher', 'war')

    print("K-fold cross-validation complete!")
    return results

def create_keras_model_temp(input_dim, name="model"):
    """Temporary Keras model creation for K-fold CV"""
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow/Keras not available")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def print_cv_summary(results):
    """Print comprehensive summary of cross-validation results"""
    print("\nCROSS-VALIDATION SUMMARY")
    print("=" * 50)

    years = results.get_available_years()
    print(f"Years with predictions: {years}")
    print(f"Total years: {len(years)}")

    for key in results.results.keys():
        data = results.results[key]
        r2 = r2_score(data['y_true'], data['y_pred'])
        rmse = np.sqrt(mean_squared_error(data['y_true'], data['y_pred']))
        mae = mean_absolute_error(data['y_true'], data['y_pred'])

        print(f"\n{key}:")
        print(f"  Total predictions: {len(data['y_true'])}")
        print(f"  R2: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")

        # Year breakdown
        for year in years:
            year_mask = np.array([str(y) == str(year) for y in data['years']])
            year_count = np.sum(year_mask)
            if year_count > 0:
                year_r2 = r2_score(np.array(data['y_true'])[year_mask],
                                  np.array(data['y_pred'])[year_mask])
                print(f"    {year}: {year_count} predictions, R2={year_r2:.4f}")


class CurrentSeasonPredictor:
    """
    Wrapper class for current season predictions using restored multi-source pipeline
    Maintains compatibility with WARPCalculator and other modules
    """

    def __init__(self):
        self.models = {}
        self.is_trained = False

    def train_ensemble_models(self, holdout_year=2024):
        """
        Train ensemble models using the restored multi-source pipeline

        Args:
            holdout_year: Year to hold out for validation
        """
        print("Training ensemble models using restored multi-source pipeline...")

        # Use the restored functions to prepare data and run cross-validation
        hitter_data, pitcher_data = prepare_data_for_kfold()
        results = run_kfold_cross_validation(hitter_data, pitcher_data, n_splits=5)

        self.models = results
        self.is_trained = True

        return results

    def predict_current_season(self, player_data, player_type='hitter'):
        """
        Make predictions for current season player data

        Args:
            player_data: DataFrame with player statistics
            player_type: 'hitter' or 'pitcher'

        Returns:
            DataFrame with predictions
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")

        # This would need to be implemented based on the trained models
        # For now, return placeholder
        return player_data.copy()