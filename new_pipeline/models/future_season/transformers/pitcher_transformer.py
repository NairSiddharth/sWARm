"""
Future Season Pitcher Feature Transformer

Loads pitcher features optimized for year-to-year prediction.
Excludes: ERA, LOB%, HR/FB%, damage_control_ratio, Opportunity_Success
"""

from typing import Optional, List
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from ..loaders.pitcher_loaders import (
    # Base features (reused)
    load_bb_pct_all_years,
    load_k_pct_all_years,
    load_swstr_all_years,
    load_gb_pct_park_adjusted,
    load_wpa_li_all_years,
    load_running_control_all_years,
    load_sd_all_years,
    load_md_all_years,
    load_hard_pct_all_years,
    load_statcast_data,
    # New features
    load_contact_pct_all_years,
    load_o_swing_pct_all_years,
    load_zone_pct_all_years,
    load_o_contact_pct_all_years,
    load_f_strike_pct_all_years
)

from ..constants import (
    COL_MLBAMID,
    COL_YEAR,
    FUTURE_PITCHER_BASE_FEATURES,
    FUTURE_PITCHER_COMPOSITE_FEATURES
)

from new_pipeline.common.logging_config import get_logger

logger = get_logger(__name__)


class FuturePitcherTransformer(BaseEstimator, TransformerMixin):
    """
    Load pitcher features for future season projections.

    Optimized for year-to-year prediction with high-correlation features.

    Features loaded (16 total before injury):
    - Base (11): BB%, K%, GB%, SwStr%, WPA/LI, Running_Control,
                 Contact%, O-Swing%, Zone%, O-Contact%, F-Strike%
    - Composites (5): strikeout_efficiency, contact_management,
                     strikeout_contact_quality, Statcast_Launch_Quality_Index, SD_MD_Net

    Excluded features:
    - ERA (r=0.38, replaced by component stats)
    - LOB% (r=0.22, unreliable)
    - HR/FB% (r=0.29, unreliable)
    - damage_control_ratio (uses LOB%, HR/FB%)
    - Opportunity_Success (uses LOB%)
    """

    def __init__(self, years: Optional[List[int]] = None):
        """
        Args:
            years: Years to load features for. If None, infer from DataFrame.
        """
        self.years = years

    def fit(self, X: pd.DataFrame, y=None):
        """Sklearn compatibility - nothing to fit."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Load all future season pitcher features and add to DataFrame.

        Args:
            X: DataFrame with MLBAMID and Year columns

        Returns:
            DataFrame with 16 pitcher features added (before injury features)

        Raises:
            ValueError: If MLBAMID or Year columns missing
        """
        if COL_MLBAMID not in X.columns or COL_YEAR not in X.columns:
            raise ValueError(f"DataFrame must have {COL_MLBAMID} and {COL_YEAR} columns")

        logger.info("Loading future season pitcher features...")

        # Get years to load
        years = self.years or X[COL_YEAR].unique().tolist()

        result = X.copy()

        # ==================================================================
        # 1. Load Base Features
        # ==================================================================

        logger.debug("Loading base features...")

        # Load reused features
        bb_pct = load_bb_pct_all_years(years)
        k_pct = load_k_pct_all_years(years)
        swstr = load_swstr_all_years(years)
        gb_pct = load_gb_pct_park_adjusted(years)
        wpa_li = load_wpa_li_all_years(years)
        running_control = load_running_control_all_years(years)
        sd = load_sd_all_years(years)
        md = load_md_all_years(years)
        hard_pct = load_hard_pct_all_years(years)
        statcast = load_statcast_data(years)

        # Load new features
        contact_pct = load_contact_pct_all_years(years)
        o_swing_pct = load_o_swing_pct_all_years(years)
        zone_pct = load_zone_pct_all_years(years)
        o_contact_pct = load_o_contact_pct_all_years(years)
        f_strike_pct = load_f_strike_pct_all_years(years)

        # Map to DataFrame
        result['BB%'] = result.apply(lambda row: bb_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['K%'] = result.apply(lambda row: k_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['SwStr%'] = result.apply(lambda row: swstr.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['GB%'] = result.apply(lambda row: gb_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['WPA/LI'] = result.apply(lambda row: wpa_li.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['Running_Control'] = result.apply(lambda row: running_control.get(row[COL_MLBAMID], np.nan), axis=1)
        result['Contact%'] = result.apply(lambda row: contact_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['O-Swing%'] = result.apply(lambda row: o_swing_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['Zone%'] = result.apply(lambda row: zone_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['O-Contact%'] = result.apply(lambda row: o_contact_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['F-Strike%'] = result.apply(lambda row: f_strike_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['Hard%'] = result.apply(lambda row: hard_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)

        # SD and MD (for composite)
        result['SD'] = result.apply(lambda row: sd.get((row[COL_MLBAMID], row[COL_YEAR]), 0), axis=1)
        result['MD'] = result.apply(lambda row: md.get((row[COL_MLBAMID], row[COL_YEAR]), 0), axis=1)

        # Statcast components (for Launch Quality Index)
        result['avg_hit_angle'] = result.apply(
            lambda row: statcast.get(row[COL_MLBAMID], {}).get('avg_hit_angle', np.nan), axis=1
        )
        result['anglesweetspotpercent'] = result.apply(
            lambda row: statcast.get(row[COL_MLBAMID], {}).get('anglesweetspotpercent', np.nan), axis=1
        )

        # ==================================================================
        # 2. Calculate Composite Features
        # ==================================================================

        logger.debug("Calculating composite features...")

        # Composite 1: strikeout_efficiency = K% × (100 - BB%)
        result['strikeout_efficiency'] = result['K%'] * (100.0 - result['BB%'])

        # Composite 2: contact_management = GB% × (100 - BB%)
        result['contact_management'] = result['GB%'] * (100.0 - result['BB%'])

        # Composite 3: strikeout_contact_quality = K% × (100 - Hard%)
        result['strikeout_contact_quality'] = result['K%'] * (100.0 - result['Hard%'])

        # Composite 4: Statcast_Launch_Quality_Index
        # Formula from pitcher_composites.py
        OPTIMAL_LAUNCH_ANGLE = 14.2
        ANGLE_WEIGHT = -0.056
        SWEET_SPOT_WEIGHT = 0.659

        result['Statcast_Launch_Quality_Index'] = (
            (OPTIMAL_LAUNCH_ANGLE - result['avg_hit_angle']) * ANGLE_WEIGHT +
            result['anglesweetspotpercent'] * SWEET_SPOT_WEIGHT
        )

        # Composite 5: SD_MD_Net = SD - MD
        result['SD_MD_Net'] = result['SD'] - result['MD']

        # ==================================================================
        # 3. Drop Intermediate Columns
        # ==================================================================

        # Drop SD, MD, avg_hit_angle, anglesweetspotpercent, Hard%
        # (only needed for composites, not model features)
        result = result.drop(columns=['SD', 'MD', 'avg_hit_angle', 'anglesweetspotpercent', 'Hard%'], errors='ignore')

        logger.info(f"Loaded {len(FUTURE_PITCHER_BASE_FEATURES)} base + {len(FUTURE_PITCHER_COMPOSITE_FEATURES)} composite features")

        return result
