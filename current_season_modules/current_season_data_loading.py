"""Data Loading Module for oWAR Analysis.

This module handles all data loading operations including:
- Primary dataset loading (hitters, pitchers, fielding, etc.)
- Cache management for mappings and computed data
- Specialized data loaders for external datasets (WARP, OAA, framing)
"""

# Standard library imports
import os
import json
from typing import Dict, Optional

# Third-party imports
import pandas as pd

# Local application imports
from common_modules.config import DATA_DIR, CACHE_DIR
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True, parents=True)

__all__ = [
    'load_primary_datasets',
    'load_mapping_from_file',
    'clear_mapping_cache',
    'clear_all_cache',
    'load_official_oaa_data',
    'load_yearly_bp_baserunning_data',
    'load_yearly_catcher_framing_data',
    'get_primary_dataframes',
    'calculate_team_games_from_hitters'
]

# Global variables to store loaded data
_primary_dataframes = {}


def load_primary_datasets():
    """
    Load all primary CSV datasets used in oWAR analysis.

    Returns:
        dict: Dictionary containing all primary dataframes
    """
    global _primary_dataframes

    logger.info("Loading primary datasets...")

    try:
        _primary_dataframes = {
            'hitter_by_game_df': pd.read_csv(
                DATA_DIR /
                "Original_Data" /
                "game_by_game" /
                "hittersByGame(player_offense_data).csv",
                low_memory=False),
            'pitcher_by_game_df': pd.read_csv(
                DATA_DIR /
                "Original_Data" /
                "game_by_game" /
                "pitchersByGame(pitcher_data).csv",
                low_memory=False),
            'baserunning_by_game_df': pd.read_csv(
                DATA_DIR /
                "Original_Data" /
                "game_by_game" /
                "baserunningNotes(player_offense_data).csv"),
            'fielding_by_game_df': pd.read_csv(
                DATA_DIR /
                "Original_Data" /
                "game_by_game" /
                "fieldingNotes(player_defensive_data).csv"),
            'warp_hitter_df': pd.read_csv(
                DATA_DIR /
                "BP_Data" /
                "hitters" /
                "bp_hitters_2021.csv"),
            'warp_pitcher_df': pd.read_csv(
                DATA_DIR /
                "BP_Data" /
                "pitchers" /
                "bp_pitchers_2021.csv"),
            'oaa_hitter_df': pd.read_csv(
                DATA_DIR /
                "Statcast_Data" /
                "outs_above_average.csv"),
            'fielding_df': pd.read_csv(
                DATA_DIR /
                "Original_Data" /
                "game_by_game" /
                "fieldingNotes(player_defensive_data).csv"),
            'baserunning_df': pd.read_csv(
                DATA_DIR /
                "Original_Data" /
                "game_by_game" /
                "baserunningNotes(player_offense_data).csv"),
            'war_df': pd.read_csv(
                DATA_DIR /
                "FanGraphs_Data" /
                "FanGraphs Leaderboard.csv")}

        logger.info(f"Successfully loaded {len(_primary_dataframes)} primary datasets:")
        for name, df in _primary_dataframes.items():
            logger.info(f"  {name}: {len(df):,} rows")

        return _primary_dataframes

    except FileNotFoundError as e:
        logger.error(f"Could not find required data file - {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading primary datasets: {e}", exc_info=True)
        return {}


def get_primary_dataframes():
    """
    Get the loaded primary dataframes. Loads them if not already loaded.

    Returns:
        dict: Dictionary containing all primary dataframes
    """
    global _primary_dataframes

    if not _primary_dataframes:
        return load_primary_datasets()

    return _primary_dataframes


def load_mapping_from_file(source_names, target_names) -> Optional[Dict]:
    """
    Load name mapping from persistent file

    Args:
        source_names: List of source names
        target_names: List of target names

    Returns:
        dict or None: Cached mapping if valid, None otherwise
    """
    # Generate cache filename locally to avoid circular imports
    import hashlib
    source_str = '|'.join(sorted([str(x) for x in source_names if pd.notna(x)]))
    target_str = '|'.join(sorted([str(x) for x in target_names if pd.notna(x)]))
    combined = f"{source_str}||{target_str}"
    hash_obj = hashlib.md5(combined.encode())
    hash_str = hash_obj.hexdigest()[:16]
    filename = f"name_mapping_{hash_str}.json"
    filepath = CACHE_DIR / filename

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # Validate cache is still current
        metadata = cache_data.get('metadata', {})
        source_count = len([x for x in source_names if pd.notna(x)])
        target_count = len([x for x in target_names if pd.notna(x)])

        if (metadata.get('source_count') == source_count and
                metadata.get('target_count') == target_count):

            logger.info(f"Loaded cached mapping from {filename}")
            logger.debug(f"   Created: {metadata.get('created_timestamp', 'Unknown')}")
            logger.debug(f"   Mappings: {metadata.get('mapping_count', 0)}")
            return cache_data['mapping']
        else:
            logger.info(f"Cache invalid (data changed), will regenerate mapping")
            return None

    except Exception as e:
        logger.warning(f"Could not load mapping cache: {e}")
        return None


def clear_mapping_cache() -> None:
    """Clear all cached name mappings (useful when data changes)"""
    try:
        cache_files = list(CACHE_DIR.glob("name_mapping_*.json"))
        for file in cache_files:
            file.unlink()
        logger.info(f"Cleared {len(cache_files)} cached mapping files")
    except Exception as e:
        logger.warning(f"Could not clear cache: {e}")


def clear_all_cache() -> None:
    """Clear all cached data (mappings, baserunning, defensive)"""
    try:
        cache_files = list(CACHE_DIR.glob("*.json"))
        for file in cache_files:
            file.unlink()
        logger.info(f"Cleared {len(cache_files)} cached files")
    except Exception as e:
        logger.warning(f"Could not clear cache: {e}")


def load_official_oaa_data() -> Dict:
    """
    Load and clean the official OAA data for comparison

    Returns:
        dict: {player_name: {official_oaa, position, fielding_runs_prevented}}
    """
    # Get OAA dataframe from primary datasets
    dataframes = get_primary_dataframes()
    oaa_hitter_df = dataframes.get('oaa_hitter_df')

    if oaa_hitter_df is None:
        logger.warning("OAA hitter dataframe not loaded")
        return {}

    oaa_data = {}

    for _, row in oaa_hitter_df.iterrows():
        last_name = str(row.get('last_name', '')).strip()
        first_name = str(row.get(' first_name', '')).strip()

        if last_name == 'nan' or first_name == 'nan' or not last_name or not first_name:
            continue

        player_name = f"{first_name} {last_name}"
        oaa_value = row.get('outs_above_average', 0)
        position = str(row.get('primary_pos_formatted', '')).strip()

        if pd.notna(oaa_value):
            oaa_data[player_name] = {
                'official_oaa': float(oaa_value),
                'position': position,
                'fielding_runs_prevented': row.get('fielding_runs_prevented', 0)
            }

    return oaa_data

# Note: load_comprehensive_fangraphs_data has been moved to legacy_modules.historical_data_loading (deprecated)

# Note: load_yearly_bp_data has been moved to legacy_modules.historical_data_loading (deprecated)


def load_yearly_catcher_framing_data() -> Dict:
    """
    Load and unify catcher framing data from 2016-2021 with yearly breakdown

    Returns:
        dict: {player_name_year: framing_runs} e.g. {'Buster Posey_2016': 31.0}
    """
    cache_file = os.path.join(
        CACHE_DIR, "yearly_catcher_framing_data.json"
    )

    # Check cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            logger.info(
                f"Loaded cached yearly catcher framing data "
                f"({len(cached_data)} player-seasons)"
            )
            return cached_data
        except Exception:
            pass

    logger.info("=== LOADING YEARLY CATCHER FRAMING DATA (2016-2021) ===")

    yearly_framing_data = {}

    # Data format patterns by year
    years_with_formats = {
        # Format 1: separate last_name/first_name columns, player_id, runs_extra_strikes
        2016: 'format1',
        2017: 'format1',
        # Format 2: single name column (id), rv_tot for run value
        2018: 'format2',
        2019: 'format2',
        2020: 'format2',
        2021: 'format1'  # But with fielder_2 instead of player_id
    }

    for year in range(2016, 2022):
        filename = os.path.join(DATA_DIR, "Statcast_Data", f'catcher_framing_{year}.csv')
        if not os.path.exists(filename):
            logger.warning(f"  Missing file: {filename}")
            continue

        try:
            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip()  # Clean column names
            logger.info(f"Processing {year}: {len(df)} records")

            format_type = years_with_formats.get(year, 'format1')

            if format_type == 'format1':
                # Format with separate name columns
                for _, row in df.iterrows():
                    if year == 2021:
                        # 2021 uses 'fielder_2' instead of 'player_id'
                        last_name = str(row.get('fielder_2', '')).strip()
                        first_name = str(row.get('first_name', '')).strip()
                    else:
                        last_name = str(row.get('last_name', '')).strip()
                        first_name = str(row.get('first_name', '')).strip()

                    framing_runs = row.get('runs_extra_strikes', 0)

                    if last_name and first_name and pd.notna(framing_runs):
                        player_name = f"{first_name} {last_name}"
                        key = f"{player_name}_{year}"
                        yearly_framing_data[key] = float(framing_runs)

            elif format_type == 'format2':
                # Format with single name column
                for _, row in df.iterrows():
                    player_name = str(row.get('id', '')).strip()
                    framing_runs = row.get('rv_tot', 0)

                    if player_name and pd.notna(framing_runs) and player_name != 'nan':
                        key = f"{player_name}_{year}"
                        yearly_framing_data[key] = float(framing_runs)

            count = len([k for k in yearly_framing_data.keys() if k.endswith(f'_{year}')])
            logger.info(f"  {year}: Added {count} player records")

        except Exception as e:
            logger.error(f"  Error processing {filename}: {e}")

    # Cache the results
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(yearly_framing_data, f, indent=2)
        logger.info(f"Cached yearly catcher framing data to {cache_file}")
    except Exception as e:
        logger.info(f"Warning: Could not cache framing data: {e}")

    logger.info(f"Loaded yearly catcher framing data: {len(yearly_framing_data)} player-seasons")
    return yearly_framing_data


def load_yearly_bp_baserunning_data() -> Dict:
    """
    Load and unify Baseball Prospectus baserunning data from 2016-2024
    Combined with Statcast running splits data for speed-adjusted calculations

    Returns:
        dict: {
            'baserunning': {player_name_year: baserunning_stats_dict},
            'running_speed': {player_name_year: speed_metrics_dict}
        }
    """
    cache_file = os.path.join(CACHE_DIR, "yearly_bp_baserunning_data.json")

    # Check cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            logger.info(f"Loaded cached BP baserunning data:")
            logger.info(f"  Baserunning: {len(cached_data.get('baserunning', {}))} player-seasons")
            logger.info(f"  Running Speed: {len(cached_data.get('running_speed', {}))} player-seasons")
            return cached_data
        except Exception:
            pass

    logger.info("=== LOADING BP BASERUNNING DATA (2016-2024) ===")

    bp_baserunning_data = {'baserunning': {}, 'running_speed': {}}
    missing_data_log = {'no_speed_data': [], 'no_baserunning_data': []}

    # Load BP baserunning data (2016-2024)
    for year in range(2016, 2025):  # 2016-2024
        filename = os.path.join(DATA_DIR, "BP_Data", "baserunning", f'bp_baserunning_{year}.csv')
        if not os.path.exists(filename):
            logger.info(f"  Missing BP baserunning file: {year}")
            continue

        try:
            df = pd.read_csv(filename, encoding='utf-8-sig')  # Handle BOM
            df.columns = df.columns.str.strip().str.strip('"')

            for _, row in df.iterrows():
                player_name = str(row.get('Name', '')).strip()
                mlbid = row.get('mlbid', None)

                if player_name and player_name != 'nan':
                    key = f"{player_name}_{year}"

                    # Store BP baserunning metrics
                    bp_baserunning_data['baserunning'][key] = {
                        'name': player_name,
                        'year': year,
                        'mlbid': mlbid,
                        'age': row.get('Age', None),
                        'SB': row.get('SB', 0),
                        'CS': row.get('CS', 0),
                        'SB_pct': row.get('SB%', None),
                        'PO': row.get('PO', 0),
                        'XBT_pct': row.get('XBT%', None),
                        'TRAA': row.get('TRAA', 0),  # BP's baserunning runs above average
                        'SRAA': row.get('SRAA', 0),  # BP's stolen base runs above average
                        'DRB': row.get('DRB', 0)     # BP's overall baserunning metric
                    }

            count = len([
                k for k in bp_baserunning_data['baserunning'].keys()
                if k.endswith(f'_{year}')
            ])
            logger.info(f"  {year} baserunning: {count} players loaded")

        except Exception as e:
            logger.info(f"  Error loading {filename}: {e}")

    # Load Statcast running splits data (2016-2024)
    for year in range(2016, 2025):  # 2016-2024
        filename = os.path.join(
            DATA_DIR,
            "Statcast_Data",
            "running_splits",
            f'running_splits_statcast_{year}.csv')
        if not os.path.exists(filename):
            logger.info(f"  Missing Statcast running splits file: {year}")
            continue

        try:
            df = pd.read_csv(filename, encoding='utf-8-sig')
            df.columns = df.columns.str.strip().str.strip('"')

            for _, row in df.iterrows():
                # Parse player name from "Last, First" format
                name_field = str(row.get('last_name, first_name', '')).strip()
                if name_field and name_field != 'nan' and ',' in name_field:
                    parts = name_field.split(',')
                    if len(parts) >= 2:
                        last_name = parts[0].strip()
                        first_name = parts[1].strip()
                        player_name = f"{first_name} {last_name}"
                    else:
                        continue
                else:
                    continue

                player_id = row.get('player_id', None)

                key = f"{player_name}_{year}"

                # Calculate speed: 90ft / time_to_first_base (in ft/sec)
                time_to_first = row.get('seconds_since_hit_090', None)  # 90ft = first base
                if time_to_first and pd.notna(time_to_first) and float(time_to_first) > 0:
                    speed_ft_per_sec = 90.0 / float(time_to_first)
                else:
                    speed_ft_per_sec = None

                bp_baserunning_data['running_speed'][key] = {
                    'name': player_name,
                    'year': year,
                    'player_id': player_id,
                    'age': row.get('age', None),
                    'time_to_first': time_to_first,
                    'speed_ft_per_sec': speed_ft_per_sec
                }

            count = len([
                k for k in bp_baserunning_data['running_speed'].keys()
                if k.endswith(f'_{year}')
            ])
            logger.info(f"  {year} running speed: {count} players loaded")

        except Exception as e:
            logger.info(f"  Error loading {filename}: {e}")

    # Log missing data matches
    baserunning_players = set(bp_baserunning_data['baserunning'].keys())
    speed_players = set(bp_baserunning_data['running_speed'].keys())

    missing_data_log['no_speed_data'] = list(baserunning_players - speed_players)
    missing_data_log['no_baserunning_data'] = list(speed_players - baserunning_players)

    if missing_data_log['no_speed_data']:
        count = len(missing_data_log['no_speed_data'])
        logger.warning(
            f"{count} players have baserunning data but no speed data"
        )
    if missing_data_log['no_baserunning_data']:
        count = len(missing_data_log['no_baserunning_data'])
        logger.warning(
            f"{count} players have speed data but no baserunning data"
        )

    # Cache the results
    try:
        cache_data = bp_baserunning_data.copy()
        cache_data['missing_data_log'] = missing_data_log

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, default=str)
        logger.info(f"Cached BP baserunning data to {cache_file}")
    except Exception as e:
        logger.info(f"Warning: Could not cache BP baserunning data: {e}")

    baserunning_count = len(bp_baserunning_data['baserunning'])
    speed_count = len(bp_baserunning_data['running_speed'])
    logger.info(
        f"Loaded BP baserunning data: {baserunning_count} baserunning player-seasons, "
        f"{speed_count} speed player-seasons"
    )
    return bp_baserunning_data


def calculate_team_games_from_hitters(
    year: int, data_source: str = 'fangraphs'
) -> Dict[str, int]:
    """
    Calculate games played by each team using hitter data.

    Uses the maximum games played by any hitter on each team as a proxy for
    team games played. This is more accurate than using pitcher games since
    everyday position players participate in most team games.

    Args:
        year: Year to calculate team games for (2016-2025)
        data_source: 'fangraphs' or 'bp' for data source

    Returns:
        Dictionary mapping team abbreviation to games played

    Raises:
        ValueError: If year is invalid or data source not recognized
        FileNotFoundError: If data file doesn't exist
    """
    if year < 2016 or year > 2025:
        raise ValueError(f"Year {year} outside valid range (2016-2025)")

    if data_source not in ['fangraphs', 'bp']:
        raise ValueError(f"Invalid data source: {data_source}. Must be 'fangraphs' or 'bp'")

    try:
        # Construct file path based on year and data source
        if data_source == 'fangraphs':
            if year == 2025:
                file_path = DATA_DIR / "FanGraphs_Data" / "hitters" / "fangraphs_hitters_2025_firsthalf.csv"
            else:
                file_path = DATA_DIR / "FanGraphs_Data" / "hitters" / f"fangraphs_hitters_{year}.csv"
        else:  # bp
            file_path = DATA_DIR / "BP_Data" / "hitters" / f"bp_hitters_{year}_standard.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load hitter data
        hitter_df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(hitter_df)} hitters from {data_source} for {year}")

        # Identify team and games columns based on data source
        team_col = 'Team' if 'Team' in hitter_df.columns else 'team'
        games_col = 'G' if 'G' in hitter_df.columns else 'games'

        if team_col not in hitter_df.columns:
            cols_preview = list(hitter_df.columns)[:10]
            raise ValueError(
                f"Team column not found in data. Available columns: {cols_preview}..."
            )
        if games_col not in hitter_df.columns:
            cols_preview = list(hitter_df.columns)[:10]
            raise ValueError(
                f"Games column not found in data. Available columns: {cols_preview}..."
            )

        # Group by team and get max games played
        team_games = hitter_df.groupby(team_col)[games_col].max().to_dict()

        # Log results
        avg_team_games = sum(team_games.values()) / len(team_games) if team_games else 0
        logger.info(f"Calculated team games for {len(team_games)} teams")
        logger.info(f"Average team games: {avg_team_games:.1f}")
        logger.info(f"Range: {min(team_games.values())} to {max(team_games.values())} games")

        return team_games

    except Exception as e:
        logger.error(f"Error calculating team games from hitters: {e}")
        raise
