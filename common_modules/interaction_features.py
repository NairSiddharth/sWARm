"""
Interaction feature calculations for pitcher WAR prediction.

Based on FanGraphs SIERA research:
- High K% pitchers induce weak contact
- GB% + control = contact management pathway
- K% value increases with lower BB%
"""

from typing import Dict
import numpy as np
from .logging import get_logger

logger = get_logger(__name__)


def calculate_strikeout_efficiency(
    k_pct_data: Dict[int, float],
    bb_pct_data: Dict[int, float]
) -> Dict[int, float]:
    """
    Calculate K% × (100 - BB%) interaction feature.

    Captures: Strikeout ability is MORE valuable with low walk rates.

    Args:
        k_pct_data: Dictionary mapping player_id to K% (decimal format: 0.22 for 22%)
        bb_pct_data: Dictionary mapping player_id to BB% (decimal format: 0.08 for 8%)

    Returns:
        Dictionary mapping player_id to strikeout efficiency value
    """
    strikeout_efficiency = {}
    all_players = set(k_pct_data.keys()) | set(bb_pct_data.keys())

    for player_id in all_players:
        k_pct_raw = k_pct_data.get(player_id, 0.232)  # League average K% (decimal)
        bb_pct_raw = bb_pct_data.get(player_id, 0.082)  # League average BB% (decimal)

        # Convert from decimal to percentage format
        k_pct = k_pct_raw * 100.0
        bb_pct = bb_pct_raw * 100.0

        efficiency = k_pct * (100.0 - bb_pct)
        strikeout_efficiency[player_id] = efficiency

    logger.debug(f"Calculated strikeout efficiency for {len(strikeout_efficiency)} players")
    return strikeout_efficiency


def calculate_contact_management(
    gb_pct_data: Dict[int, float],
    bb_pct_data: Dict[int, float]
) -> Dict[int, float]:
    """
    Calculate GB% × (100 - BB%) interaction feature.

    Captures: Ground ball ability is MORE valuable with low walk rates.
    Alternative pathway to strikeouts (Maddux-type pitchers).

    Args:
        gb_pct_data: Dictionary mapping player_id to GB% (decimal format: 0.44 for 44%)
        bb_pct_data: Dictionary mapping player_id to BB% (decimal format: 0.08 for 8%)

    Returns:
        Dictionary mapping player_id to contact management value
    """
    contact_management = {}
    all_players = set(gb_pct_data.keys()) | set(bb_pct_data.keys())

    for player_id in all_players:
        gb_pct_raw = gb_pct_data.get(player_id, 0.426)  # League average GB% (decimal)
        bb_pct_raw = bb_pct_data.get(player_id, 0.082)   # League average BB% (decimal)

        # Convert from decimal to percentage format
        gb_pct = gb_pct_raw * 100.0
        bb_pct = bb_pct_raw * 100.0

        management = gb_pct * (100.0 - bb_pct)
        contact_management[player_id] = management

    logger.debug(f"Calculated contact management for {len(contact_management)} players")
    return contact_management


def calculate_strikeout_contact_quality(
    k_pct_data: Dict[int, float],
    hard_pct_data: Dict[int, float]
) -> Dict[int, float]:
    """
    Calculate K% × (100 - Hard%) interaction feature.

    Captures: SIERA finding that high-K pitchers induce weak contact.
    Dual-threat metric (strikeouts + contact quality).

    Args:
        k_pct_data: Dictionary mapping player_id to K% (decimal format: 0.22 for 22%)
        hard_pct_data: Dictionary mapping player_id to Hard% (decimal format: 0.35 for 35%)

    Returns:
        Dictionary mapping player_id to strikeout contact quality value
    """
    strikeout_contact = {}
    all_players = set(k_pct_data.keys()) | set(hard_pct_data.keys())

    for player_id in all_players:
        k_pct_raw = k_pct_data.get(player_id, 0.232)     # League average K% (decimal)
        hard_pct_raw = hard_pct_data.get(player_id, 0.308)  # League average Hard% (decimal)

        # Convert from decimal to percentage format
        k_pct = k_pct_raw * 100.0
        hard_pct = hard_pct_raw * 100.0

        contact_quality = k_pct * (100.0 - hard_pct)
        strikeout_contact[player_id] = contact_quality

    logger.debug(f"Calculated strikeout contact quality for {len(strikeout_contact)} players")
    return strikeout_contact


def calculate_all_interaction_features(
    k_pct_data: Dict[int, float],
    bb_pct_data: Dict[int, float],
    gb_pct_data: Dict[int, float],
    hard_pct_data: Dict[int, float]
) -> Dict[str, Dict[int, float]]:
    """
    Calculate all interaction features at once.

    Args:
        k_pct_data: K% data
        bb_pct_data: BB% data
        gb_pct_data: GB% data
        hard_pct_data: Hard% data

    Returns:
        Dictionary with keys:
        - 'strikeout_efficiency': K% × (100 - BB%)
        - 'contact_management': GB% × (100 - BB%)
        - 'strikeout_contact_quality': K% × (100 - Hard%)
    """
    return {
        'strikeout_efficiency': calculate_strikeout_efficiency(k_pct_data, bb_pct_data),
        'contact_management': calculate_contact_management(gb_pct_data, bb_pct_data),
        'strikeout_contact_quality': calculate_strikeout_contact_quality(k_pct_data, hard_pct_data)
    }
