"""
Data Preparation for Future Season Projections

Load historical data and prepare longitudinal sequences for model training.
Uses new pipeline infrastructure exclusively - NO custom FanGraphs loading.

See FUTURE_PROJECTIONS_DATA_PREPARATION.md for detailed specifications.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from new_pipeline.notebooks.shared.pipeline_runner import load_historical_data
from new_pipeline.models.future_season.constants import (
    FUTURE_HITTER_MODEL_FEATURES,
    FUTURE_PITCHER_MODEL_FEATURES
)
from new_pipeline.models.future_season.transformers import (
    FuturePitcherTransformer,
    FutureHitterTransformer
)
from new_pipeline.models.future_season.injury_feature_builder import InjuryFeatureBuilder
from new_pipeline.common.transformers.age_enricher import AgeEnricher


def load_historical_player_data(
    player_type: str,
    years: range = range(2016, 2025),
    min_usage_threshold: Optional[int] = None
) -> pd.DataFrame:
    """
    Load historical player data with new pipeline features.

    Uses new_pipeline loaders and transformers to ensure feature consistency
    with current season and ROS projection systems.

    Args:
        player_type: 'hitter' or 'pitcher'
        years: Range of years to load (default: 2016-2024)
        min_usage_threshold: Minimum PA/IP to include (None = use defaults)
            Default: 75 PA for hitters, 20 IP for pitchers

    Returns:
        pd.DataFrame with columns:
            - playerid (MLBAMID)
            - Name
            - Team
            - Year
            - Age
            - WAR
            - Model features (19 for hitters, 21 for pitchers)
            - Usage metric (PA for hitters, IP for pitchers)
    """
    if player_type not in ['hitter', 'pitcher']:
        raise ValueError(f"player_type must be 'hitter' or 'pitcher', got {player_type}")

    # Set default thresholds
    if min_usage_threshold is None:
        min_usage_threshold = 75 if player_type == 'hitter' else 20

    # Handle both list and range objects
    if isinstance(years, range):
        year_str = f"{years.start}-{years.stop-1}"
    else:
        year_str = f"{min(years)}-{max(years)}"

    print(f"Loading historical {player_type} data for years {year_str}...")

    # Load raw historical data using new pipeline
    df_raw = load_historical_data(player_type, years=years)
    print(f"  Loaded {len(df_raw)} raw player-seasons")

    # Add injury features BEFORE pipeline processing (Phase 1: Injury Feature Engineering)
    print("  Adding injury features to raw data...")
    injury_builder = InjuryFeatureBuilder(
        injury_data_dir=project_root / "MLB Player Data" / "FanGraphs_Data" / "injuries"
    )
    injury_builder.load_injury_data(years=range(2020, 2026))
    df_raw = injury_builder.add_injury_features_to_historical_data(df_raw)

    # Add Age column from Baseball Prospectus data
    print("  Adding age data from Baseball Prospectus...")
    age_enricher = AgeEnricher(years=list(years), player_type=player_type)
    df_raw = age_enricher.fit_transform(df_raw)

    # Ensure required columns for transformers (rename playerid to MLBAMID if needed)
    if 'playerid' in df_raw.columns and 'MLBAMID' not in df_raw.columns:
        df_raw = df_raw.rename(columns={'playerid': 'MLBAMID'})

    # Process through future season transformers to get features
    if player_type == 'hitter':
        transformer = FutureHitterTransformer(years=list(years))
        model_features = FUTURE_HITTER_MODEL_FEATURES
    else:
        transformer = FuturePitcherTransformer(years=list(years))
        model_features = FUTURE_PITCHER_MODEL_FEATURES

    df_processed = transformer.fit_transform(df_raw)
    print(f"  Processed {len(df_processed)} player-seasons through future season pipeline")

    # Debug: Check what columns exist after transformation
    print(f"  DEBUG: Checking for new features after transformation:")
    for feat in ['ISO', 'GB%', 'HR/FB', 'Hard%', 'Pull%']:
        if feat in df_processed.columns:
            non_nan = df_processed[feat].notna().sum()
            total = len(df_processed)
            print(f"    {feat}: {non_nan}/{total} non-NaN values")
        else:
            print(f"    {feat}: MISSING from DataFrame")

    # Rename MLBAMID back to playerid for consistency with rest of codebase
    if 'MLBAMID' in df_processed.columns:
        df_processed = df_processed.rename(columns={'MLBAMID': 'playerid'})

    usage_col = 'PA' if player_type == 'hitter' else 'IP'

    required_cols = ['playerid', 'Name', 'Team', 'Year', 'Age', 'WAR', usage_col] + model_features

    # Filter to columns that exist
    available_cols = [col for col in required_cols if col in df_processed.columns]
    print(f"  DEBUG: Features in available_cols: {[c for c in available_cols if c in ['ISO', 'GB%', 'HR/FB', 'Hard%', 'Pull%']]}")
    df_filtered = df_processed[available_cols].copy()

    # Apply usage threshold
    df_filtered = df_filtered[df_filtered[usage_col] >= min_usage_threshold].copy()
    print(f"  Filtered to {len(df_filtered)} player-seasons with {usage_col} >= {min_usage_threshold}")

    # Sort by player and year for sequence building
    df_filtered = df_filtered.sort_values(['playerid', 'Year']).reset_index(drop=True)

    return df_filtered


def build_longitudinal_sequences(
    df: pd.DataFrame,
    player_type: str,
    min_sequence_length: int = 2
) -> pd.DataFrame:
    """
    Create year-to-year training sequences: Year N features → Year N+1 WAR.

    For each player with consecutive seasons, create sequences where:
    - Features from Year N are inputs
    - WAR from Year N+1 is the target

    Args:
        df: Historical player data from load_historical_player_data()
        player_type: 'hitter' or 'pitcher'
        min_sequence_length: Minimum career length to include (default: 2 years)

    Returns:
        pd.DataFrame with columns:
            - playerid
            - year_n: Year N (feature year)
            - year_n_plus_1: Year N+1 (target year)
            - age_n: Age in Year N
            - war_n: WAR in Year N
            - [feature]_n: Each model feature from Year N (e.g. 'K%_n', 'BB%_n')
            - war_n_plus_1: WAR in Year N+1 (TARGET)
    """
    model_features = FUTURE_HITTER_MODEL_FEATURES if player_type == 'hitter' else FUTURE_PITCHER_MODEL_FEATURES

    sequences = []

    # Group by player
    for playerid, player_df in df.groupby('playerid'):
        player_df = player_df.sort_values('Year').reset_index(drop=True)

        # Need at least 2 consecutive seasons
        if len(player_df) < min_sequence_length:
            continue

        # Create sequences for each consecutive year pair
        for i in range(len(player_df) - 1):
            year_n = player_df.iloc[i]
            year_n_plus_1 = player_df.iloc[i + 1]

            # Check if years are consecutive
            if year_n_plus_1['Year'] != year_n['Year'] + 1:
                continue

            # Build sequence dictionary
            sequence = {
                'playerid': playerid,
                'year_n': int(year_n['Year']),
                'year_n_plus_1': int(year_n_plus_1['Year']),
                'age_n': float(year_n['Age']),
                'war_n': float(year_n['WAR']),
                'war_n_plus_1': float(year_n_plus_1['WAR'])  # TARGET
            }

            # Add model features from Year N
            for feature in model_features:
                if feature in year_n:
                    sequence[f'{feature}_n'] = float(year_n[feature])

            sequences.append(sequence)

    sequences_df = pd.DataFrame(sequences)

    # Debug: Check what _n feature columns exist in sequences
    print(f"  DEBUG: Checking for _n feature columns in sequences:")
    for feat in ['ISO_n', 'GB%_n', 'HR/FB_n', 'Hard%_n', 'Pull%_n']:
        if feat in sequences_df.columns:
            non_nan = sequences_df[feat].notna().sum()
            total = len(sequences_df)
            print(f"    {feat}: {non_nan}/{total} non-NaN values")
        else:
            print(f"    {feat}: MISSING from sequences DataFrame")

    print(f"Created {len(sequences_df)} training sequences from {df['playerid'].nunique()} players")
    print(f"  Year range: {sequences_df['year_n'].min()}-{sequences_df['year_n'].max()}")
    print(f"  Predicting: {sequences_df['year_n_plus_1'].min()}-{sequences_df['year_n_plus_1'].max()}")

    # Add age and career context features
    sequences_df = add_age_context_features(sequences_df, player_type)

    return sequences_df


def add_age_context_features(
    sequences_df: pd.DataFrame,
    player_type: str
) -> pd.DataFrame:
    """
    Add age and career context features to sequences.

    Adds:
    - age_squared: age^2 for non-linear aging patterns
    - years_from_peak: Distance from position-specific peak age
    - age_group: Categorical age bins (one-hot encoded)
    - career_war: Cumulative WAR up to Year N
    - seasons_played: Number of seasons up to Year N
    - peak_war: Best single-season WAR up to Year N

    Args:
        sequences_df: Output from build_longitudinal_sequences()
        player_type: 'hitter' or 'pitcher'

    Returns:
        sequences_df with additional age/career context features
    """
    df = sequences_df.copy()

    # Age features
    df['age_squared'] = df['age_n'] ** 2

    # Position-specific peak ages (from future_projections.py lines 55-73)
    # For now, use general peak age - can be refined with position data
    peak_age = 27 if player_type == 'pitcher' else 28
    df['years_from_peak'] = df['age_n'] - peak_age

    # Age groups (one-hot encoding)
    df['age_group_young'] = (df['age_n'] < 26).astype(int)
    df['age_group_prime'] = ((df['age_n'] >= 26) & (df['age_n'] <= 30)).astype(int)
    df['age_group_veteran'] = (df['age_n'] > 30).astype(int)

    # Career context - requires loading full player history
    # For initial implementation, we'll use simplified version
    # TODO: Implement full career WAR tracking when we add survival model
    df['career_war'] = df['war_n']  # Placeholder - use Year N WAR as proxy
    df['seasons_played'] = 1  # Placeholder
    df['peak_war'] = df['war_n']  # Placeholder

    print(f"Added age/career context features:")
    print(f"  Age range: {df['age_n'].min():.1f} - {df['age_n'].max():.1f}")
    print(f"  Years from peak: {df['years_from_peak'].min():.0f} to {df['years_from_peak'].max():.0f}")

    return df


def create_temporal_splits(
    sequences_df: pd.DataFrame,
    train_years: Optional[range] = None,
    val_years: Optional[range] = None,
    test_years: Optional[range] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create temporal train/validation/test splits.

    Ensures no future data leakage by splitting on Year N (not Year N+1).

    Default splits (if not specified):
    - Train: 2016-2021 (predicting 2017-2022)
    - Validation: 2022 (predicting 2023)
    - Test: 2023 (predicting 2024)

    Args:
        sequences_df: Output from build_longitudinal_sequences()
        train_years: Years to use for training (Year N values)
        val_years: Years to use for validation (Year N values)
        test_years: Years to use for testing (Year N values)

    Returns:
        (train_df, val_df, test_df)
    """
    # Default splits if not provided
    if train_years is None:
        train_years = range(2016, 2022)  # 2016-2021
    if val_years is None:
        val_years = range(2022, 2023)  # 2022
    if test_years is None:
        test_years = range(2023, 2024)  # 2023

    # Split on Year N (feature year)
    train_df = sequences_df[sequences_df['year_n'].isin(train_years)].copy()
    val_df = sequences_df[sequences_df['year_n'].isin(val_years)].copy()
    test_df = sequences_df[sequences_df['year_n'].isin(test_years)].copy()

    print("Temporal splits created:")
    print(f"  Train: {len(train_df)} sequences from years {list(train_years)} -> predicting {train_df['year_n_plus_1'].min()}-{train_df['year_n_plus_1'].max()}")
    print(f"  Val:   {len(val_df)} sequences from years {list(val_years)} -> predicting {val_df['year_n_plus_1'].min()}-{val_df['year_n_plus_1'].max()}")
    print(f"  Test:  {len(test_df)} sequences from years {list(test_years)} -> predicting {test_df['year_n_plus_1'].min()}-{test_df['year_n_plus_1'].max()}")

    return train_df, val_df, test_df
