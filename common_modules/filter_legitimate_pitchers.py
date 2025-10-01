#!/usr/bin/env python3
"""
Filter Legitimate Pitchers Module
==================================

Cross-reference pitcher and hitter lists to filter out position players
who occasionally pitch, while keeping:
1. Pure pitchers (not in hitter dataset)
2. Legitimate two-way players (meet substantial criteria for both)

Two-way criteria (both must be met):
- Pitcher: >= 20 IP or >= 10 games pitched
- Hitter: >= 100 PA or >= 50 games played
"""

__version__ = '2.0.0'
__author__ = 'oWAR Development Team'

# Standard library imports
import sys
from typing import Dict, Optional, Tuple, Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from .logging import get_logger

logger = get_logger(__name__)

# Constants for two-way player criteria
MIN_PITCHER_IP = 20.0  # Minimum IP for legitimate pitcher
MIN_PITCHER_GAMES = 10  # Minimum games pitched
MIN_HITTER_PA = 100  # Minimum PA for legitimate hitter
MIN_HITTER_GAMES = 50  # Minimum games played

# Public API exports
__all__ = [
    'apply_pitcher_filtering',
    'apply_hitter_filtering',
    'analyze_pitcher_hitter_overlap',
    'identify_player_types',
    'test_realistic_war_prediction'
]


def analyze_pitcher_hitter_overlap() -> Optional[Dict[str, Any]]:
    """
    Analyze overlap between pitcher and hitter datasets.

    Returns:
        Dictionary with overlap analysis results or None if error

    Raises:
        ImportError: If required modules cannot be imported
        ValueError: If data validation fails
    """
    logger.info("Analyzing pitcher-hitter overlap")

    try:
        from current_season_modules.modeling import prepare_data_for_kfold

        # Load data
        hitter_data, pitcher_data = prepare_data_for_kfold()

        if not pitcher_data or 'war' not in pitcher_data:
            logger.error("Failed to load pitcher data")
            raise ValueError("Pitcher data is missing or incomplete")

        if not hitter_data or 'war' not in hitter_data:
            logger.error("Failed to load hitter data")
            raise ValueError("Hitter data is missing or incomplete")

        # Extract pitcher info
        pitcher_war_X = pitcher_data['war']['X']
        pitcher_war_y = pitcher_data['war']['y']

        # Extract hitter info
        hitter_war_X = hitter_data['war']['X']
        hitter_war_y = hitter_data['war']['y']

        logger.info(f"Total pitchers in dataset: {len(pitcher_war_y)}")
        logger.info(f"Total hitters in dataset: {len(hitter_war_y)}")

        # Get feature names
        if hasattr(pitcher_war_X, 'columns'):
            pitcher_features = pitcher_war_X.columns.tolist()
            pitcher_X_array = pitcher_war_X.values
        else:
            pitcher_features = [
                'IP',
                'BB%',
                'K%',
                'ERA',
                'damage_control_ratio',
                'Opportunity_Success',
                'Hard%',
                'Med%',
                'Soft%',
                'HBP',
                'WP']
            pitcher_X_array = pitcher_war_X

        if hasattr(hitter_war_X, 'columns'):
            hitter_features = hitter_war_X.columns.tolist()
            hitter_X_array = hitter_war_X.values
        else:
            hitter_features = [
                'K%',
                'BB%',
                'AVG',
                'OBP',
                'SLG',
                'PA',
                'Positional_WAR',
                'GDP_rate',
                'Enhanced_Baserunning',
                'Enhanced_Defense']
            hitter_X_array = hitter_war_X

        logger.debug(f"Pitcher features: {pitcher_features}")
        logger.debug(f"Hitter features: {hitter_features}")

        # Analyze pitcher activity levels
        if 'IP' in pitcher_features:
            ip_idx = pitcher_features.index('IP')
            ip_values = pitcher_X_array[:, ip_idx]

            logger.info(f"Pitcher activity analysis:")
            logger.info(f"IP range: {ip_values.min():.1f} to {ip_values.max():.1f}")

            # Create activity categories
            categories = {
                'Very Low (1-5 IP)': (ip_values >= 1) & (ip_values < 5),
                'Low (5-10 IP)': (ip_values >= 5) & (ip_values < 10),
                'Light (10-20 IP)': (ip_values >= 10) & (ip_values < 20),
                'Moderate (20-50 IP)': (ip_values >= 20) & (ip_values < 50),
                'Substantial (50-100 IP)': (ip_values >= 50) & (ip_values < 100),
                'High (100+ IP)': ip_values >= 100
            }

            for category, mask in categories.items():
                count = mask.sum()
                pct = 100 * count / len(ip_values)
                avg_war = pitcher_war_y[mask].mean() if count > 0 else 0
                logger.info(
                    f"  {category}: {count:4d} pitchers ({pct:4.1f}%) - Avg WAR: {avg_war:+.3f}")

        # Analyze hitter activity levels
        if 'PA' in hitter_features:
            pa_idx = hitter_features.index('PA')
            pa_values = hitter_X_array[:, pa_idx]

            logger.info("Hitter activity analysis:")
            logger.info(f"PA range: {pa_values.min():.0f} to {pa_values.max():.0f}")

            # Create PA categories
            pa_categories = {
                'Very Low (1-50 PA)': (pa_values >= 1) & (pa_values < 50),
                'Low (50-100 PA)': (pa_values >= 50) & (pa_values < 100),
                'Light (100-200 PA)': (pa_values >= 100) & (pa_values < 200),
                'Moderate (200-400 PA)': (pa_values >= 200) & (pa_values < 400),
                'Substantial (400-600 PA)': (pa_values >= 400) & (pa_values < 600),
                'High (600+ PA)': pa_values >= 600
            }

            for category, mask in pa_categories.items():
                count = mask.sum()
                pct = 100 * count / len(pa_values)
                avg_war = hitter_war_y[mask].mean() if count > 0 else 0
                logger.info(
                    f"  {category}: {count:4d} hitters ({pct:4.1f}%) - Avg WAR: {avg_war:+.3f}")

        return {
            'pitcher_X': pitcher_X_array,
            'pitcher_y': pitcher_war_y,
            'pitcher_features': pitcher_features,
            'hitter_X': hitter_X_array,
            'hitter_y': hitter_war_y,
            'hitter_features': hitter_features
        }

    except ImportError as e:
        logger.error(f"Import error: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error analyzing overlap: {str(e)}", exc_info=True)
        raise


def identify_player_types(data_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Identify pure pitchers, pure hitters, and two-way players.

    Args:
        data_dict: Dictionary with pitcher and hitter data

    Returns:
        Dictionary with player type classifications or None if error

    Raises:
        ValueError: If required data fields are missing
    """
    logger.info("Identifying player types")

    pitcher_X = data_dict['pitcher_X']
    pitcher_y = data_dict['pitcher_y']
    pitcher_features = data_dict['pitcher_features']
    hitter_X = data_dict['hitter_X']
    hitter_y = data_dict['hitter_y']
    hitter_features = data_dict['hitter_features']

    # Use module constants for thresholds
    PITCHER_IP_THRESHOLD = MIN_PITCHER_IP
    HITTER_PA_THRESHOLD = MIN_HITTER_PA

    logger.info(f"Two-way player criteria:")
    logger.info(f"  Pitcher threshold: >= {PITCHER_IP_THRESHOLD} IP")
    logger.info(f"  Hitter threshold: >= {HITTER_PA_THRESHOLD} PA")

    # Get IP and PA values
    ip_idx = pitcher_features.index('IP') if 'IP' in pitcher_features else None
    pa_idx = hitter_features.index('PA') if 'PA' in hitter_features else None

    if ip_idx is None:
        logger.error("No IP data found in pitcher features")
        raise ValueError("IP feature not found in pitcher data")

    if pa_idx is None:
        logger.error("No PA data found in hitter features")
        raise ValueError("PA feature not found in hitter data")

    ip_values = pitcher_X[:, ip_idx]
    pa_values = hitter_X[:, pa_idx]

    # Classify pitchers
    legitimate_pitchers = ip_values >= PITCHER_IP_THRESHOLD
    occasional_pitchers = ip_values < PITCHER_IP_THRESHOLD

    # Classify hitters
    legitimate_hitters = pa_values >= HITTER_PA_THRESHOLD
    occasional_hitters = pa_values < HITTER_PA_THRESHOLD

    logger.info(f"Pitcher classification:")
    logger.info(
        f"  Legitimate pitchers (>={PITCHER_IP_THRESHOLD} IP): {
            legitimate_pitchers.sum()} / {
            len(ip_values)} ({
                100 *
                legitimate_pitchers.sum() /
            len(ip_values):.1f}%)")
    logger.info(
        f"  Occasional pitchers (<{PITCHER_IP_THRESHOLD} IP): {
            occasional_pitchers.sum()} / {
            len(ip_values)} ({
                100 *
                occasional_pitchers.sum() /
            len(ip_values):.1f}%)")

    logger.info(f"Hitter classification:")
    logger.info(
        f"  Legitimate hitters (>={HITTER_PA_THRESHOLD} PA): {
            legitimate_hitters.sum()} / {
            len(pa_values)} ({
                100 *
                legitimate_hitters.sum() /
            len(pa_values):.1f}%)")
    logger.info(
        f"  Occasional hitters (<{HITTER_PA_THRESHOLD} PA): {
            occasional_hitters.sum()} / {
            len(pa_values)} ({
                100 *
                occasional_hitters.sum() /
            len(pa_values):.1f}%)")

    # Create filtered dataset
    logger.info("Filtering logic:")
    logger.info(f"  KEEP: All legitimate pitchers (>={PITCHER_IP_THRESHOLD} IP)")
    logger.info(
        f"  REMOVE: Occasional pitchers (<{PITCHER_IP_THRESHOLD} IP) who are likely position players")

    # Apply filter
    filtered_pitcher_X = pitcher_X[legitimate_pitchers]
    filtered_pitcher_y = pitcher_y[legitimate_pitchers]

    logger.info("Filtering results:")
    logger.info(f"  Original pitchers: {len(pitcher_y)}")
    logger.info(f"  Filtered pitchers: {len(filtered_pitcher_y)}")
    logger.info(f"  Removed: {len(pitcher_y) - len(filtered_pitcher_y)} likely position players")
    logger.info(f"  Retention rate: {100 * len(filtered_pitcher_y) / len(pitcher_y):.1f}%")

    # Analyze removed players
    removed_pitcher_X = pitcher_X[occasional_pitchers]
    removed_pitcher_y = pitcher_y[occasional_pitchers]

    logger.debug("Removed players analysis:")
    logger.debug(f"  Count: {len(removed_pitcher_y)}")
    logger.debug(f"  IP range: {removed_pitcher_X[:,
                                                  ip_idx].min():.1f} to {removed_pitcher_X[:,
                                                                                           ip_idx].max():.1f}")
    logger.debug(f"  WAR range: {removed_pitcher_y.min():.3f} to {removed_pitcher_y.max():.3f}")
    logger.debug(f"  Average WAR: {removed_pitcher_y.mean():.3f}")

    # Analyze kept players
    logger.debug("Kept players analysis:")
    logger.debug(f"  Count: {len(filtered_pitcher_y)}")
    logger.debug(f"  IP range: {filtered_pitcher_X[:,
                                                   ip_idx].min():.1f} to {filtered_pitcher_X[:,
                                                                                             ip_idx].max():.1f}")
    logger.debug(f"  WAR range: {filtered_pitcher_y.min():.3f} to {filtered_pitcher_y.max():.3f}")
    logger.debug(f"  Average WAR: {filtered_pitcher_y.mean():.3f}")

    return {
        'filtered_pitcher_X': filtered_pitcher_X,
        'filtered_pitcher_y': filtered_pitcher_y,
        'pitcher_features': pitcher_features,
        'removed_count': len(removed_pitcher_y),
        'kept_count': len(filtered_pitcher_y)
    }


def test_realistic_war_prediction(filtered_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Test WAR prediction with properly filtered pitcher data.

    Args:
        filtered_data: Dictionary with filtered pitcher data

    Returns:
        Dictionary with model test results or None if error

    Raises:
        ImportError: If required ML libraries cannot be imported
    """
    logger.info("Testing realistic WAR prediction")

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        pitcher_X = filtered_data['filtered_pitcher_X']
        pitcher_y = filtered_data['filtered_pitcher_y']
        features = filtered_data['pitcher_features']

        logger.info(f"Testing with {len(pitcher_y)} legitimate pitchers")
        logger.debug(f"Features: {features}")

        # Test different model configurations
        models = {
            'RF_default': RandomForestRegressor(
                n_estimators=100,
                random_state=42),
            'RF_regularized': RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=10,
                random_state=42),
            'RF_simple': RandomForestRegressor(
                n_estimators=20,
                max_depth=5,
                min_samples_split=20,
                random_state=42)}

        logger.info("Cross-validation results:")
        results = {}

        for name, model in models.items():
            try:
                scores = cross_val_score(model, pitcher_X, pitcher_y, cv=5, scoring='r2')
                results[name] = {
                    'r2_mean': scores.mean(),
                    'r2_std': scores.std()
                }
                logger.info(f"  {name}: R² = {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                logger.error(f"  {name}: Failed - {e}")
                results[name] = None

        # Test without IP feature to avoid IP-WAR correlation
        logger.info("Test without IP feature (to avoid IP-WAR correlation):")

        if 'IP' in features:
            ip_idx = features.index('IP')
            pitcher_X_no_ip = np.delete(pitcher_X, ip_idx, axis=1)
            features_no_ip = [f for i, f in enumerate(features) if i != ip_idx]

            logger.debug(f"Features without IP: {features_no_ip}")

            for name, model in models.items():
                if results[name] is not None:  # Only test if original worked
                    try:
                        scores = cross_val_score(
                            model, pitcher_X_no_ip, pitcher_y, cv=5, scoring='r2')
                        logger.info(
                            f"  {name} (no IP): R² = {
                                scores.mean():.4f} ± {
                                scores.std():.4f}")
                    except Exception as e:
                        logger.error(f"  {name} (no IP): Failed - {e}")

        return results

    except ImportError as e:
        logger.error(f"Import error: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error analyzing overlap: {str(e)}", exc_info=True)
        raise


def apply_pitcher_filtering(hitter_data: Dict, pitcher_data: Dict) -> Optional[Dict]:
    """
    Apply filtering to remove position players who pitch in blowouts.

    Criteria:
    - Remove pitchers who also appear in hitter dataset with substantial hitting stats
    - Keep legitimate two-way players (meet both pitcher and hitter thresholds)
    - Pitcher criteria: >= 20 IP OR >= 10 games pitched
    - Hitter criteria: >= 100 PA OR >= 50 games played

    Returns:
        dict: Filtered pitcher data with same structure as input
    """
    if not pitcher_data or not hitter_data:
        raise ValueError("Pitcher or hitter data is missing")

    if 'war' not in pitcher_data or 'war' not in hitter_data:
        raise ValueError("WAR data is missing from input")

    logger.info(f"Before filtering: {len(pitcher_data['war']['y'])} pitchers")

    try:
        import pandas as pd
        import numpy as np

        # Get pitcher features
        pitcher_X = pitcher_data['war']['X']
        pitcher_y = pitcher_data['war']['y']

        # Look for IP column to filter by innings pitched
        if 'IP' in pitcher_X.columns:
            # Filter out very low innings pitched (position players pitching in blowouts)
            min_ip_threshold = MIN_PITCHER_IP  # Use module constant

            # Simple boolean mask approach
            ip_mask = pitcher_X['IP'] >= min_ip_threshold

            # Filter DataFrames and arrays using boolean mask directly
            filtered_X = pitcher_X[ip_mask].copy()

            # Handle different types for pitcher_y
            if isinstance(pitcher_y, (list, tuple)):
                # Convert list/tuple to numpy array, apply mask, convert back to list
                filtered_y = np.array(pitcher_y)[ip_mask].tolist()
            else:
                # pandas Series/DataFrame or numpy array
                filtered_y = pitcher_y[ip_mask].copy()

            # Update other arrays if they exist
            filtered_data = {
                'war': {
                    'X': filtered_X,
                    'y': filtered_y
                }
            }

            # Copy other data if present
            if 'names' in pitcher_data['war']:
                names_array = pitcher_data['war']['names']
                if isinstance(names_array, (list, tuple)):
                    names_np = np.array(names_array)
                    # Check if it's an array matching the data length
                    if len(names_np.shape) > 0 and len(names_np) == len(pitcher_y):
                        # Array matching data length - apply mask
                        filtered_data['war']['names'] = names_np[ip_mask].tolist()
                    else:
                        # Keep original if shape doesn't match
                        filtered_data['war']['names'] = names_array
                else:
                    # Handle numpy arrays or other array-like objects
                    try:
                        filtered_data['war']['names'] = names_array[ip_mask].copy()
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Could not apply mask to names array: {e}")
                        filtered_data['war']['names'] = names_array
            if 'years' in pitcher_data['war']:
                years_array = pitcher_data['war']['years']
                # Handle tuple-wrapped lists (common from data processing)
                if isinstance(years_array, tuple) and len(years_array) == 1:
                    years_array = years_array[0]

                if isinstance(years_array, (list, tuple)):
                    years_np = np.array(years_array)
                    # Check if it's a single value or an array matching the data length
                    if years_np.shape == () or (len(years_np.shape) == 1 and len(years_np) == 1):
                        # Single year value - keep it as is
                        filtered_data['war']['years'] = years_array
                    elif len(years_np) == len(pitcher_y):
                        # Array matching data length - apply mask
                        filtered_data['war']['years'] = years_np[ip_mask].tolist()
                    else:
                        # Unexpected shape - keep original
                        logger.warning(f"Unexpected years array shape: {years_np.shape}, expected length {len(pitcher_y)}")
                        filtered_data['war']['years'] = years_array
                else:
                    # Handle numpy arrays or other array-like objects
                    try:
                        filtered_data['war']['years'] = years_array[ip_mask].copy()
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Could not apply mask to years array: {e}")
                        filtered_data['war']['years'] = years_array

            # Copy MLB IDs if present
            if 'mlbids' in pitcher_data['war']:
                mlbids_array = pitcher_data['war']['mlbids']
                if isinstance(mlbids_array, (list, tuple)):
                    mlbids_np = np.array(mlbids_array)
                    if len(mlbids_np) == len(pitcher_y):
                        filtered_data['war']['mlbids'] = mlbids_np[ip_mask].tolist()
                    else:
                        filtered_data['war']['mlbids'] = mlbids_array
                else:
                    try:
                        filtered_data['war']['mlbids'] = mlbids_array[ip_mask].copy()
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Could not apply mask to mlbids array: {e}")
                        filtered_data['war']['mlbids'] = mlbids_array

            # Copy WARP data if present
            if 'warp' in pitcher_data:
                # Apply same filtering to WARP data
                if len(pitcher_data['warp']['X']) == len(pitcher_X):
                    warp_y = pitcher_data['warp']['y']
                    if isinstance(warp_y, (list, tuple)):
                        filtered_warp_y = np.array(warp_y)[ip_mask].tolist()
                    else:
                        filtered_warp_y = warp_y[ip_mask].copy()

                    filtered_data['warp'] = {
                        'X': pitcher_data['warp']['X'][ip_mask].copy(),
                        'y': filtered_warp_y
                    }
                    if 'names' in pitcher_data['warp']:
                        warp_names = pitcher_data['warp']['names']
                        if isinstance(warp_names, (list, tuple)):
                            warp_names_np = np.array(warp_names)
                            if len(warp_names_np.shape) > 0 and len(warp_names_np) == len(pitcher_y):
                                filtered_data['warp']['names'] = warp_names_np[ip_mask].tolist()
                            else:
                                filtered_data['warp']['names'] = warp_names
                        else:
                            filtered_data['warp']['names'] = warp_names[ip_mask].copy()
                    if 'years' in pitcher_data['warp']:
                        warp_years = pitcher_data['warp']['years']
                        if isinstance(warp_years, (list, tuple)):
                            warp_years_np = np.array(warp_years)
                            if warp_years_np.shape == () or len(warp_years_np) == 1:
                                filtered_data['warp']['years'] = warp_years
                            elif len(warp_years_np) == len(pitcher_y):
                                filtered_data['warp']['years'] = warp_years_np[ip_mask].tolist()
                            else:
                                filtered_data['warp']['years'] = warp_years
                        else:
                            filtered_data['warp']['years'] = warp_years[ip_mask].copy()

                    if 'mlbids' in pitcher_data['warp']:
                        warp_mlbids = pitcher_data['warp']['mlbids']
                        if isinstance(warp_mlbids, (list, tuple)):
                            filtered_data['warp']['mlbids'] = np.array(warp_mlbids)[ip_mask].tolist()
                        else:
                            filtered_data['warp']['mlbids'] = warp_mlbids[ip_mask].copy()
                else:
                    # Keep original WARP data if sizes don't match
                    filtered_data['warp'] = pitcher_data['warp']

            removed_count = len(pitcher_y) - len(filtered_y)
            logger.info(
                f"After filtering: {
                    len(filtered_y)} pitchers (removed {removed_count} position players)")

            if removed_count > 0:
                removed_ip = pitcher_X['IP'][~ip_mask]
                kept_ip = pitcher_X['IP'][ip_mask]
                logger.debug(
                    f"Removed pitchers IP range: {
                        removed_ip.min():.1f} to {
                        removed_ip.max():.1f}")
                logger.debug(f"Kept pitchers IP range: {kept_ip.min():.1f} to {kept_ip.max():.1f}")

            return filtered_data

        else:
            logger.warning("No IP column found, cannot apply innings-based filtering")
            return pitcher_data

    except Exception as e:
        logger.error(f"Error in pitcher filtering: {e}", exc_info=True)
        raise


def apply_hitter_filtering(hitter_data: Dict, pitcher_data: Dict) -> Optional[Dict]:
    """
    Apply filtering to remove pitchers who were forced to hit (NL pre-DH).

    Criteria:
    - Remove hitters who also appear in pitcher dataset with substantial pitching stats
      but minimal hitting stats (forced to hit due to rules)
    - Keep legitimate two-way players (meet both pitcher and hitter thresholds)
    - Keep pure hitters (not in pitcher dataset)

    Parameters:
        hitter_data: Dictionary with 'war' and optionally 'warp' keys containing hitting data
        pitcher_data: Dictionary with 'war' and optionally 'warp' keys containing pitching data

    Returns:
        dict: Filtered hitter data with same structure as input
    """
    if not hitter_data or not pitcher_data:
        raise ValueError("Hitter or pitcher data is missing")

    if 'war' not in hitter_data:
        raise ValueError("WAR data is missing from hitter input")

    logger.info(f"Before filtering: {len(hitter_data['war']['y'])} hitters")

    try:
        import pandas as pd
        import numpy as np

        # Get hitter features
        hitter_X = hitter_data['war']['X']
        hitter_y = hitter_data['war']['y']

        # Get pitcher features for cross-reference
        pitcher_X = pitcher_data['war']['X'] if 'war' in pitcher_data else None

        # Look for PA column to filter by plate appearances
        if 'PA' in hitter_X.columns and pitcher_X is not None:
            # Get MLB IDs for cross-reference from the data dictionary
            hitter_mlbids = hitter_data['war'].get('mlbids', [])
            pitcher_mlbids = pitcher_data['war'].get('mlbids', [])

            # Create boolean mask - start with all True
            keep_mask = np.ones(len(hitter_y), dtype=bool)

            if hitter_mlbids and pitcher_mlbids and 'IP' in pitcher_X.columns:
                # Build a dictionary of pitcher IDs with their IP values
                pitcher_ids_with_ip = {}
                for idx in range(len(pitcher_mlbids)):
                    pitcher_id = pitcher_mlbids[idx]
                    if pd.notna(pitcher_id):
                        pitcher_ip = pitcher_X.iloc[idx]['IP']
                        pitcher_ids_with_ip[int(pitcher_id)] = pitcher_ip

                logger.debug(f"Found {len(pitcher_ids_with_ip)} pitchers with valid IDs for cross-reference")

                # Check each hitter
                removed_count_detail = 0
                kept_two_way = 0
                hitter_names = hitter_data['war'].get('names', [])

                for idx in range(len(hitter_mlbids)):
                    hitter_id = hitter_mlbids[idx]
                    if pd.notna(hitter_id):
                        hitter_id = int(hitter_id)
                        pa = hitter_X.iloc[idx]['PA']

                        # If this hitter is also in the pitcher dataset
                        if hitter_id in pitcher_ids_with_ip:
                            pitcher_ip = pitcher_ids_with_ip[hitter_id]

                            # If high IP and low PA, this is a pitcher forced to hit
                            if pitcher_ip >= MIN_PITCHER_IP and pa < MIN_HITTER_PA:
                                keep_mask[idx] = False
                                removed_count_detail += 1
                                # Get name if available for logging
                                name = hitter_names[idx] if idx < len(hitter_names) else f"ID:{hitter_id}"
                                logger.debug(f"Removing forced hitter: {name} (IP: {pitcher_ip:.1f}, PA: {pa})")
                            # If both high, this is a two-way player - keep them
                            elif pitcher_ip >= MIN_PITCHER_IP and pa >= MIN_HITTER_PA:
                                kept_two_way += 1
                                name = hitter_names[idx] if idx < len(hitter_names) else f"ID:{hitter_id}"
                                logger.debug(f"Keeping two-way player: {name} (IP: {pitcher_ip:.1f}, PA: {pa})")

                if removed_count_detail > 0:
                    logger.info(f"Removed {removed_count_detail} pitchers forced to hit")
                if kept_two_way > 0:
                    logger.info(f"Kept {kept_two_way} legitimate two-way players")
            else:
                # Fallback: Simple PA-based filtering
                min_pa_threshold = MIN_HITTER_PA
                keep_mask = hitter_X['PA'] >= min_pa_threshold
                logger.info("Using simple PA-based filtering (no name cross-reference available)")

            # Apply filtering
            filtered_X = hitter_X[keep_mask].copy()

            # Handle different types for hitter_y
            if isinstance(hitter_y, (list, tuple)):
                filtered_y = np.array(hitter_y)[keep_mask].tolist()
            else:
                filtered_y = hitter_y[keep_mask].copy()

            # Create filtered data structure
            filtered_data = {
                'war': {
                    'X': filtered_X,
                    'y': filtered_y
                }
            }

            # Copy other data if present
            if 'names' in hitter_data['war']:
                names_array = hitter_data['war']['names']
                if isinstance(names_array, (list, tuple)):
                    names_np = np.array(names_array)
                    if len(names_np.shape) > 0 and len(names_np) == len(hitter_y):
                        filtered_data['war']['names'] = names_np[keep_mask].tolist()
                    else:
                        filtered_data['war']['names'] = names_array
                else:
                    try:
                        filtered_data['war']['names'] = names_array[keep_mask].copy()
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Could not apply mask to names array: {e}")
                        filtered_data['war']['names'] = names_array

            if 'years' in hitter_data['war']:
                years_array = hitter_data['war']['years']
                # Handle tuple-wrapped lists (common from data processing)
                if isinstance(years_array, tuple) and len(years_array) == 1:
                    years_array = years_array[0]

                if isinstance(years_array, (list, tuple)):
                    years_np = np.array(years_array)
                    if years_np.shape == () or (len(years_np.shape) == 1 and len(years_np) == 1):
                        filtered_data['war']['years'] = years_array
                    elif len(years_np) == len(hitter_y):
                        filtered_data['war']['years'] = years_np[keep_mask].tolist()
                    else:
                        logger.warning(f"Unexpected years array shape: {years_np.shape}, expected length {len(hitter_y)}")
                        filtered_data['war']['years'] = years_array
                else:
                    try:
                        filtered_data['war']['years'] = years_array[keep_mask].copy()
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Could not apply mask to years array: {e}")
                        filtered_data['war']['years'] = years_array

            # Copy MLB IDs if present
            if 'mlbids' in hitter_data['war']:
                mlbids_array = hitter_data['war']['mlbids']
                if isinstance(mlbids_array, (list, tuple)):
                    mlbids_np = np.array(mlbids_array)
                    if len(mlbids_np) == len(hitter_y):
                        filtered_data['war']['mlbids'] = mlbids_np[keep_mask].tolist()
                    else:
                        filtered_data['war']['mlbids'] = mlbids_array
                else:
                    filtered_data['war']['mlbids'] = mlbids_array

            # Copy WARP data if present
            if 'warp' in hitter_data:
                if 'X' in hitter_data['warp'] and 'y' in hitter_data['warp']:
                    warp_X = hitter_data['warp']['X']
                    warp_y = hitter_data['warp']['y']

                    if len(warp_X) == len(hitter_X):
                        filtered_data['warp'] = {
                            'X': warp_X[keep_mask].copy(),
                            'y': warp_y[keep_mask].copy() if not isinstance(warp_y, (list, tuple)) else np.array(warp_y)[keep_mask].tolist()
                        }

                        if 'names' in hitter_data['warp']:
                            warp_names = hitter_data['warp']['names']
                            if isinstance(warp_names, (list, tuple)):
                                filtered_data['warp']['names'] = np.array(warp_names)[keep_mask].tolist()
                            else:
                                filtered_data['warp']['names'] = warp_names[keep_mask].copy()

                        if 'years' in hitter_data['warp']:
                            warp_years = hitter_data['warp']['years']
                            if isinstance(warp_years, (list, tuple)):
                                years_np = np.array(warp_years)
                                if years_np.shape == () or len(years_np) == 1:
                                    filtered_data['warp']['years'] = warp_years
                                elif len(years_np) == len(hitter_y):
                                    filtered_data['warp']['years'] = years_np[keep_mask].tolist()
                                else:
                                    filtered_data['warp']['years'] = warp_years
                            else:
                                filtered_data['warp']['years'] = warp_years[keep_mask].copy()

                        if 'mlbids' in hitter_data['warp']:
                            warp_mlbids = hitter_data['warp']['mlbids']
                            if isinstance(warp_mlbids, (list, tuple)):
                                filtered_data['warp']['mlbids'] = np.array(warp_mlbids)[keep_mask].tolist()
                            else:
                                filtered_data['warp']['mlbids'] = warp_mlbids[keep_mask].copy()
                    else:
                        filtered_data['warp'] = hitter_data['warp']

            removed_count = len(hitter_y) - len(filtered_y)
            logger.info(
                f"After filtering: {len(filtered_y)} hitters (removed {removed_count} forced hitters)")

            if removed_count > 0:
                removed_pa = hitter_X['PA'][~keep_mask]
                kept_pa = hitter_X['PA'][keep_mask]
                if len(removed_pa) > 0:
                    logger.debug(f"Removed hitters PA range: {removed_pa.min():.0f} to {removed_pa.max():.0f}")
                logger.debug(f"Kept hitters PA range: {kept_pa.min():.0f} to {kept_pa.max():.0f}")

            return filtered_data

        else:
            logger.warning("No PA column found or no pitcher data for cross-reference, cannot apply PA-based filtering")
            return hitter_data

    except Exception as e:
        logger.error(f"Error in hitter filtering: {e}", exc_info=True)
        raise
