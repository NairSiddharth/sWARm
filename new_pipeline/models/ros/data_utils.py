"""
Data conversion utilities for ROS models.

Handles conversion between different time series formats:
- sktime: MultiIndex Series/DataFrames for DirectROSForecaster
- Darts: TimeSeries objects for temporal models (TCN, TSMixer, AutoARIMA)
- numpy: Standard arrays for MultiQuantileHistGB baseline
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from darts import TimeSeries


def convert_to_sktime_format(
    historical_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'WAR_per_600'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert historical data to sktime time series format.

    Creates time-indexed series per player for DirectROSForecaster.
    DirectROSForecaster will automatically create lag features from y,
    and use X as exogenous variables.

    For panel data (multiple players), sktime requires y as DataFrame with MultiIndex,
    not Series with MultiIndex.

    Args:
        historical_df: DataFrame with columns:
            - player_id: Player identifier
            - Year: Year (will be converted to datetime)
            - WAR_per_600 or WAR_per_162: Target variable
            - All feature columns
        feature_columns: List of feature column names (all ROS features)
        target_column: Target column name (default: 'WAR_per_600')

    Returns:
        Tuple of (y, X) where:
        - y: pd.DataFrame with MultiIndex (player_id, datetime) and single column (target)
        - X: pd.DataFrame with same MultiIndex and feature columns

    Example:
        >>> y, X = convert_to_sktime_format(hist_df, ROS_HITTER_FEATURES)
        >>> y.index
        MultiIndex([(12345, '2020-12-31'),
                    (12345, '2021-12-31'),
                    (12345, '2022-12-31'),
                    (67890, '2019-12-31'),
                    ...])
        >>> y.shape
        (1234, 1)  # DataFrame with 1 column
    """
    # Validate required columns
    required_cols = ['playerid', 'Year', target_column]
    missing = [col for col in required_cols if col not in historical_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create datetime index from Year
    df = historical_df.copy()
    df['datetime'] = pd.to_datetime(df['Year'], format='%Y') + pd.offsets.YearEnd(0)

    # Sort by player and time
    df = df.sort_values(['playerid', 'datetime'])

    # Create MultiIndex
    df = df.set_index(['playerid', 'datetime'])

    # Fill missing years per player to regularize time series
    # This ensures sktime's DirectReducer can determine frequency (avoids freq=None errors)
    # Similar to Darts' fill_missing_dates=True approach
    filled_dfs = []
    for player_id in df.index.get_level_values(0).unique():
        player_df = df.loc[player_id]

        # Check if player has duplicate datetime values
        # (happens in training with multiple split points per year: 0.25, 0.5, 0.75)
        if player_df.index.duplicated().any():
            # Training scenario: duplicates present, skip gap-filling
            # Time series is already dense (3 obs/year), no need to fill gaps
            player_df_to_add = player_df.copy()
            player_df_to_add['_playerid'] = player_id
            filled_dfs.append(player_df_to_add)
        else:
            # Inference scenario: no duplicates, fill gaps for regular time series
            # Get year range for this player
            min_year = player_df.index.min().year
            max_year = player_df.index.max().year

            # Create complete date range (annual frequency, year-end)
            full_range = pd.date_range(
                f'{min_year}-12-31',
                f'{max_year}-12-31',
                freq='YE'
            )

            # Reindex to fill gaps with NaN (only works when no duplicates)
            player_df_filled = player_df.reindex(full_range)

            # Add player_id back for MultiIndex reconstruction
            player_df_filled['_playerid'] = player_id
            filled_dfs.append(player_df_filled)

    # Combine all players
    df = pd.concat(filled_dfs)
    df = df.set_index('_playerid', append=True)
    df = df.swaplevel(0, 1)  # Back to (playerid, datetime) order
    df.index.names = ['playerid', 'datetime']  # Restore index names

    # Extract target as DataFrame (not Series) for panel data compatibility
    # sktime requires DataFrame with MultiIndex for panel data, not Series
    y = df[[target_column]]  # Double brackets → DataFrame with 1 column

    # Extract feature DataFrame
    # Only include features that exist in the DataFrame
    available_features = [col for col in feature_columns if col in df.columns]
    if len(available_features) < len(feature_columns):
        missing_features = set(feature_columns) - set(available_features)
        # Don't error - just use available features
        # print(f"Warning: {len(missing_features)} features not found in data")

    X = df[available_features]

    # Rebuild MultiIndex with frequency-aware DatetimeIndex as FINAL step
    # This is required for sktime's DirectReducer (avoids TypeError: 'int' and 'NoneType')
    # Must be done AFTER all dataframe operations to prevent loss during concat
    #
    # Strategy: Use pd.infer_freq() + pd.date_range() to preserve original dates
    # while setting frequency. This is more reliable than PeriodIndex conversion
    # which changes dates (Dec 31 -> Jan 1).

    # Check if frequency is already set
    if y.index.levels[1].freq is not None:
        # Already has frequency, no need to fix
        return y, X

    # Get unique dates from level 1
    unique_dates = y.index.levels[1]

    # Reconstruct with explicit frequency using date_range
    # This creates a new DatetimeIndex with frequency set while preserving dates
    min_date = unique_dates.min()
    max_date = unique_dates.max()

    try:
        # Try to infer frequency first
        inferred_freq = pd.infer_freq(unique_dates)
        if inferred_freq is None:
            # Manual frequency setting for year-end December
            inferred_freq = 'YE-DEC'

        # Create new freq-aware DatetimeIndex
        # This preserves the original dates (e.g., Dec 31) while setting frequency
        freq_dates = pd.date_range(start=min_date, end=max_date, freq=inferred_freq)

        # Update the level
        y.index = y.index.set_levels(freq_dates, level=1)
        X.index = X.index.set_levels(freq_dates, level=1)

    except (ValueError, TypeError) as e:
        # If all else fails, raise informative error
        raise ValueError(
            f"Failed to set frequency on datetime index: {e}. "
            "This will cause TypeError during DirectROSForecaster prediction."
        )

    # Final verification
    if y.index.levels[1].freq is None:
        raise ValueError(
            "Failed to set frequency on datetime index after multiple attempts. "
            "This will cause TypeError during DirectROSForecaster prediction."
        )

    return y, X


def convert_to_darts_format(
    historical_df: pd.DataFrame,
    target_column: str = 'WAR_per_600',
    min_length: int = 4
) -> List[TimeSeries]:
    """
    Convert historical data to Darts TimeSeries format (WAR only, no covariates).

    Creates one TimeSeries per player for temporal models.
    No covariates - temporal models learn "clean career arcs" from WAR history only.

    Args:
        historical_df: DataFrame with columns:
            - playerid: Player identifier
            - Year: Year
            - WAR_per_600 or WAR_per_162: Target variable
        target_column: Target column name
        min_length: Minimum number of years required (default: 4, required for Darts TCN/TSMixer)

    Returns:
        List of TimeSeries objects (one per player with sufficient history)

    Example:
        >>> series_list = convert_to_darts_format(hist_df)
        >>> len(series_list)
        245  # Players with 4+ years of history
        >>> series_list[0].values().shape
        (7, 1)  # 7 years of WAR data
    """
    # Validate required columns
    required_cols = ['playerid', 'Year', target_column]
    missing = [col for col in required_cols if col not in historical_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    series_list = []

    # Group by player
    for player_id, player_data in historical_df.groupby('playerid'):
        # Sort by year
        player_data = player_data.sort_values('Year')

        # Handle duplicate years (multi-position players)
        # Aggregate using innings-weighted average to get single value per year
        if player_data['Year'].duplicated().any():
            if 'Inn' in player_data.columns:
                # Innings-weighted aggregation
                def innings_weighted_avg(group):
                    total_inn = group['Inn'].sum()
                    if total_inn > 0 and target_column in group.columns:
                        # Reconstruct actual WAR from rate: WAR_per_600 * Inn / 600
                        actual_war = (group[target_column] * group['Inn'] / 600).sum()
                        # Re-rate by total innings
                        return (actual_war / total_inn) * 600
                    else:
                        return group[target_column].mean()  # Fallback

                # Aggregate duplicate years
                player_data = player_data.groupby('Year').agg({
                    target_column: innings_weighted_avg
                }).reset_index()
            else:
                # Fallback to simple mean if Inn column not available
                player_data = player_data.groupby('Year', as_index=False)[target_column].mean()

        # Skip if insufficient history (check AFTER aggregation to get unique year count)
        if len(player_data) < min_length:
            continue

        # Create datetime index (MLB season ends in September)
        dates = pd.to_datetime(player_data['Year'].astype(str) + '-09-30')

        # Create DataFrame for Darts (needs DatetimeIndex)
        ts_df = pd.DataFrame({
            target_column: player_data[target_column].values
        }, index=dates)

        # Convert to TimeSeries
        try:
            ts = TimeSeries.from_dataframe(
                ts_df,
                value_cols=target_column,
                fill_missing_dates=True,  # Fill structural gaps (career interruptions)
                freq='YE-SEP',  # Year End - September (MLB season end, pandas 2.0+ format)
                static_covariates=pd.DataFrame({'playerid': [player_id]})  # Set playerid for later access
            )
            # Note: Missing years are filled with NaN (no interpolation)
            # This preserves real trajectories while handling career gaps (injury, COVID, etc.)
            series_list.append(ts)
        except Exception as e:
            # Log conversion failures for debugging
            print(f"Warning: Failed to convert player {player_id} to TimeSeries: {e}")
            continue

    return series_list


def prepare_player_data(
    historical_df: pd.DataFrame,
    player_id: str,
    current_year: int,
    target_column: str = 'WAR_per_600'
) -> pd.DataFrame:
    """
    Prepare time series data for a single player.

    Args:
        historical_df: Full historical data
        player_id: Player to extract
        current_year: Current year (for prediction)
        target_column: Target column name

    Returns:
        DataFrame with player's historical data, sorted by year

    Example:
        >>> player_ts = prepare_player_data(hist_df, '12345', 2025)
        >>> player_ts
           Year  WAR_per_600  age  wOBA  ...
        0  2019         4.2   26  .395  ...
        1  2020         5.1   27  .410  ...
        2  2021         4.8   28  .398  ...
    """
    # Filter to player
    player_data = historical_df[historical_df['playerid'] == player_id].copy()

    if len(player_data) == 0:
        raise ValueError(f"No historical data found for player {player_id}")

    # Sort by year
    player_data = player_data.sort_values('Year')

    # Filter to years before current year (don't include current season in history)
    player_data = player_data[player_data['Year'] < current_year]

    return player_data


def validate_time_series_data(
    y: pd.DataFrame,
    X: Optional[pd.DataFrame] = None,
    min_length: int = 3
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Validate and clean time series data.

    Args:
        y: Target DataFrame with MultiIndex (player_id, datetime) and single target column
        X: Feature DataFrame with same index (optional)
        min_length: Minimum length per player

    Returns:
        Tuple of (cleaned_y, cleaned_X)

    Removes:
    - Players with insufficient history
    - NaN values in target
    - Non-chronological data
    """
    if not isinstance(y.index, pd.MultiIndex):
        raise ValueError("y must have MultiIndex (player_id, datetime)")

    # Check for NaN in target (DataFrame requires .any().any())
    if y.isna().any().any():
        # Remove NaN rows
        # For DataFrame, need to check if any column has NaN in each row
        valid_idx = ~y.isna().any(axis=1)
        y = y[valid_idx]
        if X is not None:
            X = X[valid_idx]

    # Group by player and filter by length
    player_counts = y.groupby(level=0).size()
    valid_players = player_counts[player_counts >= min_length].index

    # Filter to valid players
    y = y[y.index.get_level_values(0).isin(valid_players)]
    if X is not None:
        X = X[X.index.get_level_values(0).isin(valid_players)]

    # Verify chronological order
    if not y.index.is_monotonic_increasing:
        y = y.sort_index()
        if X is not None:
            X = X.sort_index()

    return y, X


def split_features_target(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = 'target_secondhalf_WAR'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split DataFrame into features (X) and target (y) arrays.

    Simple helper for baseline model (no time series conversion).

    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names
        target_column: Target column name

    Returns:
        Tuple of (X, y) as numpy arrays
    """
    # Only include features that exist
    available_features = [col for col in feature_columns if col in df.columns]

    X = df[available_features].values
    y = df[target_column].values

    return X, y


def create_prediction_data(
    current_stats: pd.DataFrame,
    historical_data: pd.DataFrame,
    feature_columns: List[str]
) -> Tuple[np.ndarray, pd.Series, pd.DataFrame, List[TimeSeries]]:
    """
    Prepare all data formats needed for prediction.

    Args:
        current_stats: Current season stats (firsthalf)
        historical_data: Historical data for all players
        feature_columns: List of ROS feature columns

    Returns:
        Tuple of:
        - X_baseline: Feature matrix for baseline model (n_players, n_features)
        - y_sktime: Target series for DirectROSForecaster (MultiIndex)
        - X_sktime: Feature DataFrame for DirectROSForecaster (MultiIndex)
        - series_darts: List of TimeSeries for Darts models

    Example:
        >>> X_base, y_skt, X_skt, series = create_prediction_data(
        ...     current_2025, historical_2016_2024, ROS_HITTER_FEATURES
        ... )
    """
    # Baseline model: current season features only
    available_features = [col for col in feature_columns if col in current_stats.columns]
    X_baseline = current_stats[available_features].values

    # sktime format: combine historical + current as time series
    # (This would need historical + current year combined)
    y_sktime, X_sktime = convert_to_sktime_format(
        historical_data,
        feature_columns
    )

    # Darts format: historical WAR only
    series_darts = convert_to_darts_format(historical_data)

    return X_baseline, y_sktime, X_sktime, series_darts


def filter_splits_to_nearest(
    historical_splits: pd.DataFrame,
    current_completion_pct: float,
    available_splits: List[float] = None
) -> pd.DataFrame:
    """
    Filter historical split data to the split point nearest to current timing.

    This ensures apples-to-apples comparison when creating lag features:
    - If predicting at 50% of season, use historical 0.5 splits for lags
    - If predicting at 35% of season, use nearest split (0.25) for lags

    Args:
        historical_splits: DataFrame with 'split_point' column (from training)
        current_completion_pct: Current season completion (0.0-1.0)
        available_splits: List of split points in data (default: [0.25, 0.5, 0.75])

    Returns:
        Filtered DataFrame containing only rows matching the nearest split point

    Example:
        >>> # Predicting at firsthalf (50% of season)
        >>> filtered = filter_splits_to_nearest(historical_splits, 0.5)
        >>> filtered['split_point'].unique()
        [0.5]  # Only 0.5 splits returned

        >>> # Predicting at 35% of season
        >>> filtered = filter_splits_to_nearest(historical_splits, 0.35)
        >>> filtered['split_point'].unique()
        [0.25]  # Nearest split is 0.25
    """
    if available_splits is None:
        available_splits = [0.25, 0.5, 0.75]

    # Validate split_point column exists
    if 'split_point' not in historical_splits.columns:
        raise ValueError("historical_splits must have 'split_point' column")

    # Find nearest split point
    nearest_split = min(available_splits, key=lambda x: abs(x - current_completion_pct))

    # Filter to only that split
    filtered = historical_splits[historical_splits['split_point'] == nearest_split].copy()

    if len(filtered) == 0:
        raise ValueError(
            f"No rows found for split_point={nearest_split}. "
            f"Available splits in data: {sorted(historical_splits['split_point'].unique())}"
        )

    return filtered
