"""
Future Season Pitcher Feature Loaders

Optimized for year-to-year prediction with high-correlation features.
Removes: ERA, LOB%, HR/FB%
Adds: Contact%, O-Swing%, Zone%, O-Contact%, F-Strike%
"""

from typing import Dict, List, Tuple

# Reuse helpers from common
from new_pipeline.common.loaders.helpers import (
    _load_fangraphs_feature,
    _convert_decimal_to_percentage,
    validate_percentage_scale
)

# Reuse existing loaders for retained features
from new_pipeline.common.loaders.pitcher_loaders import (
    load_bb_pct_all_years,             # REUSE
    load_k_pct_all_years,              # REUSE
    load_swstr_all_years,              # REUSE
    load_gb_pct_park_adjusted,         # REUSE
    load_wpa_li_all_years,             # REUSE
    load_running_control_all_years,    # REUSE
    load_sd_all_years,                 # REUSE (for SD_MD_Net composite)
    load_md_all_years,                 # REUSE (for SD_MD_Net composite)
    load_hard_pct_all_years,           # REUSE (for composites)
    load_statcast_data                 # REUSE (for Launch Quality Index)
)


# ============================================================================
# NEW LOADERS - Pitcher Plate Discipline Metrics
# ============================================================================

def load_contact_pct_all_years(years: List[int]) -> Dict[Tuple[int, int], float]:
    """
    Load Contact% (contact rate on swings) for pitchers - year-specific.

    Contact% measures how often batters make contact when they swing.
    Lower is better for pitchers (more whiffs).

    Source: FanGraphs_Data/pitchers/fangraphs_pitchers_YYYY_stuff.csv
    Column: Contact%
    Scale: FanGraphs stores as decimal (0.752 = 75.2%), convert to percentage

    Args:
        years: Years to load

    Returns:
        dict: {(MLBAMID, Year): Contact% in percentage format}

    Example:
        Spencer Strider 2023: Contact% = 67.2% (elite bat-missing)
        Kyle Hendricks 2023: Contact% = 84.1% (weak stuff, lots of contact)
    """
    # Load raw data (decimals) - Contact% is in stuff file
    contact_raw = _load_fangraphs_feature(
        years,
        'stuff',  # File type
        'Contact%',  # Column name
        player_type='pitchers'
    )

    # Convert decimal → percentage (now with year tuples)
    contact_pct = {
        (pid, year): _convert_decimal_to_percentage(val)
        for (pid, year), val in contact_raw.items()
    }

    # Validate
    if contact_pct:
        # Extract values without year for validation
        validate_percentage_scale(
            {pid: val for (pid, year), val in contact_pct.items()},
            'Contact% (Pitchers)',
            expected_range=(60, 90)
        )

    return contact_pct


def load_o_swing_pct_all_years(years: List[int]) -> Dict[Tuple[int, int], float]:
    """
    Load O-Swing% (outside zone swing rate) for pitchers - year-specific.

    O-Swing% measures how often batters chase pitches outside the strike zone.
    Higher is better for pitchers (more chases).

    Source: FanGraphs_Data/pitchers/fangraphs_pitchers_YYYY_stuff.csv
    Column: O-Swing%
    Scale: FanGraphs stores as decimal (0.315 = 31.5%), convert to percentage

    Args:
        years: Years to load

    Returns:
        dict: {(MLBAMID, Year): O-Swing% in percentage format}

    Example:
        Sandy Alcantara 2022: O-Swing% = 35.1% (sinker + slider induced chases)
        Dallas Keuchel 2023: O-Swing% = 24.8% (lost deception, batters laid off)
    """
    # Load raw data (decimals) - O-Swing% is in stuff file
    o_swing_raw = _load_fangraphs_feature(
        years,
        'stuff',
        'O-Swing%',
        player_type='pitchers'
    )

    # Convert decimal → percentage
    o_swing_pct = {
        (pid, year): _convert_decimal_to_percentage(val)
        for (pid, year), val in o_swing_raw.items()
    }

    # Validate
    if o_swing_pct:
        validate_percentage_scale(
            {pid: val for (pid, year), val in o_swing_pct.items()},
            'O-Swing% (Pitchers)',
            expected_range=(20, 40)
        )

    return o_swing_pct


def load_zone_pct_all_years(years: List[int]) -> Dict[Tuple[int, int], float]:
    """
    Load Zone% (strike zone rate) for pitchers - year-specific.

    Zone% measures how often pitchers throw strikes.
    Higher is better for pitchers (more strikes, better command).

    Source: FanGraphs_Data/pitchers/fangraphs_pitchers_YYYY_stuff.csv
    Column: Zone%
    Scale: FanGraphs stores as decimal (0.456 = 45.6%), convert to percentage

    Args:
        years: Years to load

    Returns:
        dict: {(MLBAMID, Year): Zone% in percentage format}

    Example:
        Zack Greinke 2023: Zone% = 51.2% (extreme control)
        Grayson Rodriguez 2023: Zone% = 39.4% (struggled with command)
    """
    # Load raw data (decimals) - Zone% is in stuff file
    zone_raw = _load_fangraphs_feature(
        years,
        'stuff',
        'Zone%',
        player_type='pitchers'
    )

    # Convert decimal → percentage
    zone_pct = {
        (pid, year): _convert_decimal_to_percentage(val)
        for (pid, year), val in zone_raw.items()
    }

    # Validate
    if zone_pct:
        validate_percentage_scale(
            {pid: val for (pid, year), val in zone_pct.items()},
            'Zone% (Pitchers)',
            expected_range=(35, 55)
        )

    return zone_pct


def load_o_contact_pct_all_years(years: List[int]) -> Dict[Tuple[int, int], float]:
    """
    Load O-Contact% (outside zone contact rate) for pitchers - year-specific.

    O-Contact% measures contact rate on swings outside the strike zone.
    Lower is better for pitchers (more whiffs on chase pitches).

    Source: FanGraphs_Data/pitchers/fangraphs_pitchers_YYYY_stuff.csv
    Column: O-Contact%
    Scale: FanGraphs stores as decimal (0.612 = 61.2%), convert to percentage

    Args:
        years: Years to load

    Returns:
        dict: {(MLBAMID, Year): O-Contact% in percentage format}

    Example:
        Emmanuel Clase 2023: O-Contact% = 51.7% (cutter whiffs on chases)
        Kyle Gibson 2023: O-Contact% = 68.9% (batters made contact on chases)
    """
    # Load raw data (decimals) - O-Contact% is in stuff file
    o_contact_raw = _load_fangraphs_feature(
        years,
        'stuff',
        'O-Contact%',
        player_type='pitchers'
    )

    # Convert decimal → percentage
    o_contact_pct = {
        (pid, year): _convert_decimal_to_percentage(val)
        for (pid, year), val in o_contact_raw.items()
    }

    # Validate
    if o_contact_pct:
        validate_percentage_scale(
            {pid: val for (pid, year), val in o_contact_pct.items()},
            'O-Contact% (Pitchers)',
            expected_range=(45, 75)
        )

    return o_contact_pct


def load_f_strike_pct_all_years(years: List[int]) -> Dict[Tuple[int, int], float]:
    """
    Load F-Strike% (first pitch strike rate) for pitchers - year-specific.

    F-Strike% measures how often the first pitch is a strike.
    Higher is better for pitchers (ahead in counts, better outcomes).

    Source: FanGraphs_Data/pitchers/fangraphs_pitchers_YYYY_stuff.csv
    Column: F-Strike%
    Scale: FanGraphs stores as decimal (0.618 = 61.8%), convert to percentage

    Args:
        years: Years to load

    Returns:
        dict: {(MLBAMID, Year): F-Strike% in percentage format}

    Example:
        Aaron Nola 2023: F-Strike% = 65.8% (aggressive, gets ahead)
        Patrick Corbin 2023: F-Strike% = 54.1% (falls behind in counts)
    """
    # Load raw data (decimals) - F-Strike% is in stuff file
    f_strike_raw = _load_fangraphs_feature(
        years,
        'stuff',
        'F-Strike%',
        player_type='pitchers'
    )

    # Convert decimal → percentage
    f_strike_pct = {
        (pid, year): _convert_decimal_to_percentage(val)
        for (pid, year), val in f_strike_raw.items()
    }

    # Validate
    if f_strike_pct:
        validate_percentage_scale(
            {pid: val for (pid, year), val in f_strike_pct.items()},
            'F-Strike% (Pitchers)',
            expected_range=(50, 70)
        )

    return f_strike_pct
