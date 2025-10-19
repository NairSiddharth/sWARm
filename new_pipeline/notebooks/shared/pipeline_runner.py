"""
Pipeline execution and data loading utilities for oWAR notebooks.

Provides high-level functions for:
- Loading current season and historical data
- Running sklearn pipelines
- Generating predictions with residuals and projections
- Splitting players by role/position
- Creating combined leaderboards (pitcher + hitter with two-way handling)
- Calculating model metrics
"""

from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from new_pipeline.common.transformers.pipeline_builder import build_pitcher_pipeline, build_hitter_pipeline
from new_pipeline.common.constants import (
    COL_MLBAMID,
    COL_NAME,
    COL_TWO_WAY_PLAYER,
    COL_IP,
    COL_PA,
    FANGRAPHS_PITCHER_DIR,
    FANGRAPHS_HITTER_DIR,
    PITCHER_STARTER_THRESHOLD,
    PITCHER_RELIEVER_THRESHOLD,
    PITCHER_MODEL_FEATURES,
    HITTER_MODEL_FEATURES,
    FULL_SEASON_GAMES,
    FULL_SEASON_PA,
    WAR_NORMALIZATION_IP_STARTER,
    WAR_NORMALIZATION_IP_RELIEVER,
    WAR_NORMALIZATION_IP_SWING
)
from new_pipeline.common.projections import get_team_games_from_data, calculate_remaining_usage


def load_current_season_data(player_type: str, year: int = 2025) -> pd.DataFrame:
    """
    Load current season data for pitchers or hitters.

    Args:
        player_type: 'pitcher' or 'hitter'
        year: Year to load (default: 2025)

    Returns:
        pd.DataFrame: Raw data with columns ['MLBAMID', 'Name', 'Team', 'Year', 'WAR', ...]

    Example:
        >>> df = load_current_season_data('pitcher', year=2025)
        >>> # Returns: DataFrame with all 2025 pitchers
    """
    if player_type not in ['pitcher', 'hitter']:
        raise ValueError(f"player_type must be 'pitcher' or 'hitter', got: {player_type}")

    # Determine data directory and file pattern
    if player_type == 'pitcher':
        data_dir = FANGRAPHS_PITCHER_DIR
        file_pattern = f"fangraphs_pitchers_{year}.csv"
    else:
        data_dir = FANGRAPHS_HITTER_DIR
        file_pattern = f"fangraphs_hitters_{year}.csv"

    # Check for partial season files first (firsthalf, quarter, etc.)
    # Look for base files (not advanced/standard/battedball/etc. splits)
    all_partial_files = sorted(data_dir.glob(f"fangraphs_{player_type}s_{year}_*.csv"))
    # Exclude files with secondary suffixes (advanced, standard, battedball, stuff, winprobability)
    partial_files = [
        f for f in all_partial_files
        if not any(suffix in f.name for suffix in ['_advanced', '_standard', '_battedball', '_stuff', '_winprobability'])
    ]

    if partial_files:
        # Use most recent partial season file
        csv_path = partial_files[0]  # Alphabetically first (e.g., "firsthalf" before "quarter")
        print(f"Loading partial season data: {csv_path.name}")
    else:
        csv_path = data_dir / file_pattern

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Expected either:\n"
            f"  - {data_dir / file_pattern}\n"
            f"  - {data_dir / f'fangraphs_{player_type}s_{year}_*_advanced.csv'}"
        )

    df = pd.read_csv(csv_path)

    # Add Year column if not present
    if 'Year' not in df.columns:
        df['Year'] = year

    return df


def load_historical_data(player_type: str, years: range) -> pd.DataFrame:
    """
    Load multiple years of historical data.

    Args:
        player_type: 'pitcher' or 'hitter'
        years: Years to load (e.g., range(2016, 2024))

    Returns:
        pd.DataFrame: Combined data from all years

    Example:
        >>> df = load_historical_data('hitter', years=range(2016, 2024))
        >>> # Returns: DataFrame with 2016-2023 hitters
    """
    if player_type not in ['pitcher', 'hitter']:
        raise ValueError(f"player_type must be 'pitcher' or 'hitter', got: {player_type}")

    dfs = []

    for year in years:
        try:
            df = load_current_season_data(player_type, year=year)
            dfs.append(df)
        except FileNotFoundError as e:
            print(f"Warning: Skipping {year} - {e}")
            continue

    if not dfs:
        raise ValueError(f"No data found for years {list(years)}")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def run_data_pipeline(df: pd.DataFrame, player_type: str) -> pd.DataFrame:
    """
    Execute complete sklearn pipeline on raw data.

    Args:
        df: Raw data from FanGraphs
        player_type: 'pitcher' or 'hitter'

    Returns:
        pd.DataFrame: Processed data with features added and WAR normalized

    Example:
        >>> df_processed = run_data_pipeline(df_raw, player_type='pitcher')
        >>> # Returns: DataFrame with 13 pitcher features + WAR_per_162
    """
    if player_type not in ['pitcher', 'hitter']:
        raise ValueError(f"player_type must be 'pitcher' or 'hitter', got: {player_type}")

    # Get unique years from data
    years = df['Year'].unique().tolist() if 'Year' in df.columns else [2025]

    # Build and run appropriate pipeline
    if player_type == 'pitcher':
        pipeline = build_pitcher_pipeline(years=years, include_validation=True)
    else:
        pipeline = build_hitter_pipeline(years=years, include_validation=True)

    # Transform data
    df_processed = pipeline.fit_transform(df)

    # Normalize player ID column name (handle PlayerId, playerid, MLBAMID variations)
    # This ensures consistency with CompleteProjectionGenerator and other components
    player_id_col = None
    for col in ['playerid', 'PlayerId', 'MLBAMID']:
        if col in df_processed.columns:
            player_id_col = col
            break

    # If not already 'playerid', create alias for consistency
    if player_id_col and player_id_col != 'playerid':
        df_processed = df_processed.copy()
        df_processed['playerid'] = df_processed[player_id_col]

    return df_processed


def generate_predictions(
    df: pd.DataFrame,
    model,
    player_type: str
) -> pd.DataFrame:
    """
    Generate predictions using trained model.

    Args:
        df: Processed data with features (output from run_data_pipeline)
        model: Trained model (sklearn estimator or custom ensemble)
        player_type: 'pitcher' or 'hitter'

    Returns:
        pd.DataFrame: Original data + predictions + residuals

    Columns Added:
        - Predicted_WAR_per_162 or Predicted_WAR_per_600 (rate prediction)
        - Residual (error in rate prediction)
        - Predicted_Current_WAR (predicted cumulative WAR to date)
        - Actual_Current_WAR (actual cumulative WAR from data, if available)
        - Current_Residual (error in current season prediction)
        - ROS_WAR (rest of season projection)
        - Total_Projected_WAR (predicted current + predicted ROS)

    Example:
        >>> predictions = generate_predictions(df_2025, model, player_type='pitcher')
    """
    if player_type not in ['pitcher', 'hitter']:
        raise ValueError(f"player_type must be 'pitcher' or 'hitter', got: {player_type}")

    df_result = df.copy()

    # Determine feature columns and WAR column based on player type
    if player_type == 'pitcher':
        feature_cols = PITCHER_MODEL_FEATURES
        war_col = 'WAR_per_162'
        pred_col = 'Predicted_WAR_per_162'
        usage_col = COL_IP
        full_season_usage = FULL_SEASON_GAMES  # Default for current WAR calculations
    else:
        feature_cols = HITTER_MODEL_FEATURES
        war_col = 'WAR_per_600'
        pred_col = 'Predicted_WAR_per_600'
        usage_col = COL_PA
        full_season_usage = FULL_SEASON_PA

    # Make predictions using exact model features
    X = df_result[feature_cols]

    # Handle role-based models (pitcher ensembles use duck typing)
    try:
        # Pitcher ensembles: Need role-aware predictions
        roles = df_result.apply(lambda row: _get_pitcher_role(row), axis=1).values

        # For pitchers, predict role by role with role-specific season progress
        if hasattr(model, 'set_season_progress') and usage_col in df_result.columns:
            # Role-specific IP targets
            role_ip_targets = {'starter': 162, 'reliever': 70, 'swing': 110}

            # Initialize predictions array
            predictions = np.zeros(len(df_result))

            # Predict each role separately with correct season progress
            for role, ip_target in role_ip_targets.items():
                role_mask = roles == role
                if not role_mask.any():
                    continue

                # Calculate role-specific season progress
                role_avg_ip = df_result.loc[role_mask, usage_col].mean()
                role_season_pct = min(role_avg_ip / ip_target, 1.0)

                # Set season progress for this role
                model.set_season_progress(role_season_pct)
                print(f"  {role.capitalize()}s: {role_season_pct:.1%} season ({role_avg_ip:.1f} avg IP, target={ip_target})")

                # Predict for this role only
                X_role = X.iloc[role_mask] if hasattr(X, 'iloc') else X[role_mask]
                roles_role = roles[role_mask]
                predictions[role_mask] = model.predict(X_role, roles_role)
        else:
            # No season progress support or no usage data - predict all at once
            predictions = model.predict(X, roles)
    except TypeError:
        # Standard model without role parameter (hitter ensembles)
        # Detect season progress from usage data for dynamic threshold calculation
        if hasattr(model, 'set_season_progress') and usage_col in df_result.columns:
            avg_usage = df_result[usage_col].mean()
            season_pct = min(avg_usage / full_season_usage, 1.0)
            model.set_season_progress(season_pct)
            print(f"  Detected season progress: {season_pct:.1%} ({avg_usage:.1f} avg {usage_col})")

        predictions = model.predict(X)

    df_result[pred_col] = predictions

    # Calculate residuals (if actual WAR rate exists)
    if war_col in df_result.columns:
        df_result['Residual'] = df_result[war_col] - predictions
    else:
        df_result['Residual'] = np.nan

    # Calculate Predicted Current WAR (cumulative based on current usage)
    if usage_col in df_result.columns:
        current_usage = df_result[usage_col].values

        # For pitchers, use role-specific denominators
        if player_type == 'pitcher':
            # Get role-specific denominators for each pitcher
            role_denominators = df_result.apply(
                lambda row: _get_role_specific_denominator(_get_pitcher_role(row)),
                axis=1
            ).values
            predicted_current_war = predictions * (current_usage / role_denominators)
        else:
            # For hitters, use single denominator
            predicted_current_war = predictions * (current_usage / full_season_usage)

        df_result['Predicted_Current_WAR'] = predicted_current_war
    else:
        df_result['Predicted_Current_WAR'] = 0.0

    # Calculate Actual Current WAR (for validation when actual WAR exists)
    if 'WAR' in df_result.columns:
        df_result['Actual_Current_WAR'] = df_result['WAR']

        # Calculate current season prediction error
        df_result['Current_Residual'] = df_result['Actual_Current_WAR'] - df_result['Predicted_Current_WAR']
    else:
        df_result['Actual_Current_WAR'] = np.nan
        df_result['Current_Residual'] = np.nan

    # Calculate ROS (Rest of Season) projection
    # Uses IP/G approach for pitchers, G/team_G for hitters
    if usage_col in df_result.columns:
        # Get team games context from data
        team_games_dict, league_median_games = get_team_games_from_data(df_result)

        # Calculate remaining usage (IP for pitchers, games for hitters)
        remaining_usage = df_result.apply(
            lambda row: calculate_remaining_usage(
                row,
                player_type,
                team_games_dict,
                league_median_games,
                season_length=162
            ),
            axis=1
        )

        # Prorated prediction for remaining usage
        # For pitchers: Use role-specific denominators (162/48.2/110)
        # For hitters: WAR_per_600 * (remaining_PA / 600)
        if player_type == 'pitcher':
            # Get role-specific denominators for each pitcher
            role_denominators = df_result.apply(
                lambda row: _get_role_specific_denominator(_get_pitcher_role(row)),
                axis=1
            ).values
            ros_war = predictions * (remaining_usage / role_denominators)
        else:
            ros_war = predictions * (remaining_usage / full_season_usage)

        df_result['ROS_WAR'] = ros_war
    else:
        df_result['ROS_WAR'] = predictions

    # Total projection = predicted current + predicted ROS (consistent projection)
    df_result['Total_Projected_WAR'] = df_result['Predicted_Current_WAR'] + df_result['ROS_WAR']

    return df_result


def split_by_role(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split pitchers by role (starter/reliever/swing).

    Args:
        df: Pitcher data with 'GS_per_G' column or 'GS'/'G' columns

    Returns:
        dict: {'Starter': df1, 'Reliever': df2, 'Swing': df3}

    Thresholds:
        - Starter: GS/G > 0.7
        - Reliever: GS/G < 0.1
        - Swing: 0.1 <= GS/G <= 0.7

    Example:
        >>> splits = split_by_role(df_pitchers)
        >>> # Returns: {'Starter': 234 rows, 'Reliever': 289 rows, 'Swing': 45 rows}
    """
    df_copy = df.copy()

    # Calculate GS_per_G if not present
    if 'GS_per_G' not in df_copy.columns:
        if 'GS' in df_copy.columns and 'G' in df_copy.columns:
            df_copy['GS_per_G'] = df_copy['GS'] / df_copy['G'].replace(0, 1)
        else:
            raise ValueError("DataFrame must have either 'GS_per_G' or both 'GS' and 'G' columns")

    # Split by thresholds
    starter_mask = df_copy['GS_per_G'] > PITCHER_STARTER_THRESHOLD
    reliever_mask = df_copy['GS_per_G'] < PITCHER_RELIEVER_THRESHOLD
    swing_mask = (df_copy['GS_per_G'] >= PITCHER_RELIEVER_THRESHOLD) & (df_copy['GS_per_G'] <= PITCHER_STARTER_THRESHOLD)

    splits = {
        'Starter': df_copy[starter_mask].copy(),
        'Reliever': df_copy[reliever_mask].copy(),
        'Swing': df_copy[swing_mask].copy()
    }

    return splits


def split_by_position(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split hitters by position group.

    Args:
        df: Hitter data with 'Primary_Position' column

    Returns:
        dict: {'IF': df1, 'OF': df2, 'C': df3, 'DH': df4}

    Position Mapping:
        - IF: 1B, 2B, 3B, SS
        - OF: LF, CF, RF, OF
        - C: C
        - DH: DH

    Example:
        >>> splits = split_by_position(df_hitters)
        >>> # Returns: {'IF': 145 rows, 'OF': 118 rows, 'C': 45 rows, 'DH': 19 rows}
    """
    if 'Primary_Position' not in df.columns:
        raise ValueError("DataFrame must have 'Primary_Position' column")

    df_copy = df.copy()

    # Define position groups
    if_positions = ['1B', '2B', '3B', 'SS']
    of_positions = ['LF', 'CF', 'RF', 'OF']
    c_positions = ['C']
    dh_positions = ['DH']

    # Split by position group
    splits = {
        'IF': df_copy[df_copy['Primary_Position'].isin(if_positions)].copy(),
        'OF': df_copy[df_copy['Primary_Position'].isin(of_positions)].copy(),
        'C': df_copy[df_copy['Primary_Position'].isin(c_positions)].copy(),
        'DH': df_copy[df_copy['Primary_Position'].isin(dh_positions)].copy()
    }

    return splits


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard regression metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        dict: {'MAE': float, 'RMSE': float, 'R²': float}

    Example:
        >>> metrics = calculate_metrics(y_val, y_pred)
        >>> # Returns: {'MAE': 0.52, 'RMSE': 0.71, 'R²': 0.83}
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    }


def create_combined_leaderboard(
    pitcher_df: pd.DataFrame,
    hitter_df: pd.DataFrame,
    top_n: int = 100,
    war_col_pitcher: str = 'WAR_per_162',
    war_col_hitter: str = 'WAR_per_600'
) -> pd.DataFrame:
    """
    Merge pitcher and hitter predictions into unified rankings.

    Handles two-way players by combining their pitcher and hitter WAR.
    Pure pitchers and hitters show only their respective WAR component.

    Two-way detection: Players appearing in BOTH datasets (by MLBAMID) are
    automatically identified as two-way players.

    Args:
        pitcher_df: Pitcher predictions with MLBAMID, Name, WAR_per_162
        hitter_df: Hitter predictions with MLBAMID, Name, WAR_per_600
        top_n: Number of top players to return (default: 100)
        war_col_pitcher: Column name for pitcher WAR (default: 'WAR_per_162')
        war_col_hitter: Column name for hitter WAR (default: 'WAR_per_600')

    Returns:
        pd.DataFrame: Combined leaderboard sorted by total_WAR

        Columns:
            - MLBAMID: Player ID
            - Name: Player name
            - Type: 'Pitcher', 'Hitter', or 'Two-Way'
            - total_WAR: Combined WAR (pitcher + hitter for two-way)
            - pitcher_WAR: Pitcher component
            - hitter_WAR: Hitter component

    Example:
        >>> leaderboard = create_combined_leaderboard(pitcher_preds, hitter_preds, top_n=6)
        >>> # Returns:
        >>> #   Rank  Name               Type      total_WAR  pitcher_WAR  hitter_WAR
        >>> #   1     Shohei Ohtani      Two-Way   11.7       2.5          9.2
        >>> #   2     Aaron Judge        Hitter    8.2        0.0          8.2
        >>> #   3     Juan Soto          Hitter    7.1        0.0          7.1
        >>> #   4     Tarik Skubal       Pitcher   6.8        6.8          0.0
    """
    # Validate required columns
    # Note: COL_TWO_WAY_PLAYER not required - we detect two-way via MLBAMID cross-check
    required_pitcher_cols = [COL_MLBAMID, war_col_pitcher]
    required_hitter_cols = [COL_MLBAMID, war_col_hitter]

    missing_pitcher = [col for col in required_pitcher_cols if col not in pitcher_df.columns]
    missing_hitter = [col for col in required_hitter_cols if col not in hitter_df.columns]

    if missing_pitcher:
        raise ValueError(f"pitcher_df missing required columns: {missing_pitcher}")
    if missing_hitter:
        raise ValueError(f"hitter_df missing required columns: {missing_hitter}")

    # Find two-way players (appear in both datasets)
    # Cross-check MLBAMIDs instead of relying on two_way_player flag
    # (flag can't be set correctly when pipelines run separately)
    pitcher_ids = set(pitcher_df[COL_MLBAMID].dropna())
    hitter_ids = set(hitter_df[COL_MLBAMID].dropna())
    two_way_ids = pitcher_ids & hitter_ids

    combined_rows = []

    # Two-way players: Merge predictions
    for mlbamid in two_way_ids:
        p_row = pitcher_df[pitcher_df[COL_MLBAMID] == mlbamid].iloc[0]
        h_row = hitter_df[hitter_df[COL_MLBAMID] == mlbamid].iloc[0]

        combined_rows.append({
            COL_MLBAMID: mlbamid,
            COL_NAME: p_row.get(COL_NAME, h_row.get(COL_NAME, 'Unknown')),
            'Type': 'Two-Way',
            'total_WAR': p_row[war_col_pitcher] + h_row[war_col_hitter],
            'pitcher_WAR': p_row[war_col_pitcher],
            'hitter_WAR': h_row[war_col_hitter]
        })

    # Pure pitchers
    pure_pitchers = pitcher_df[~pitcher_df[COL_MLBAMID].isin(two_way_ids)]
    for _, row in pure_pitchers.iterrows():
        combined_rows.append({
            COL_MLBAMID: row[COL_MLBAMID],
            COL_NAME: row.get(COL_NAME, 'Unknown'),
            'Type': 'Pitcher',
            'total_WAR': row[war_col_pitcher],
            'pitcher_WAR': row[war_col_pitcher],
            'hitter_WAR': 0.0
        })

    # Pure hitters
    pure_hitters = hitter_df[~hitter_df[COL_MLBAMID].isin(two_way_ids)]
    for _, row in pure_hitters.iterrows():
        combined_rows.append({
            COL_MLBAMID: row[COL_MLBAMID],
            COL_NAME: row.get(COL_NAME, 'Unknown'),
            'Type': 'Hitter',
            'total_WAR': row[war_col_hitter],
            'pitcher_WAR': 0.0,
            'hitter_WAR': row[war_col_hitter]
        })

    # Combine and rank
    combined = pd.DataFrame(combined_rows)
    combined = combined.sort_values('total_WAR', ascending=False).head(top_n)

    # Add rank column
    combined.insert(0, 'Rank', range(1, len(combined) + 1))

    return combined.reset_index(drop=True)


def _get_pitcher_role(row: pd.Series) -> str:
    """
    Helper: Determine pitcher role from row data.

    Args:
        row: DataFrame row with GS and G columns

    Returns:
        str: 'starter', 'reliever', or 'swing'
    """
    if 'GS_per_G' in row.index:
        gs_per_g = row['GS_per_G']
    elif 'GS' in row.index and 'G' in row.index:
        gs_per_g = row['GS'] / max(row['G'], 1)
    else:
        return 'starter'  # Default if data missing

    if gs_per_g > PITCHER_STARTER_THRESHOLD:
        return 'starter'
    elif gs_per_g < PITCHER_RELIEVER_THRESHOLD:
        return 'reliever'
    else:
        return 'swing'


def _get_role_specific_denominator(role: str) -> float:
    """
    Helper: Get WAR normalization denominator for pitcher role.

    Args:
        role: 'starter', 'reliever', or 'swing'

    Returns:
        float: Normalization denominator (IP for full season equivalent)
    """
    role_denominators = {
        'starter': WAR_NORMALIZATION_IP_STARTER,    # 162
        'reliever': WAR_NORMALIZATION_IP_RELIEVER,  # 48.2
        'swing': WAR_NORMALIZATION_IP_SWING         # 110
    }
    return role_denominators.get(role, WAR_NORMALIZATION_IP_STARTER)
