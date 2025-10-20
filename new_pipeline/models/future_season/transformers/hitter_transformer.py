"""
Future Season Hitter Feature Transformer

Loads hitter features with additional power and batted ball metrics.
No composites for hitters (all base features).
"""

from typing import Optional, List
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from ..loaders.hitter_loaders import (
    # Reused features
    load_k_pct_all_years,
    load_bb_pct_all_years,
    load_pa_all_years,
    load_gdp_all_years,
    load_avg_park_adjusted,
    load_obp_park_adjusted,
    load_slg_park_adjusted,
    load_positional_war,
    load_positions_all_years,
    load_enhanced_baserunning,
    load_enhanced_defense,
    # New features
    load_iso_all_years,
    load_gb_pct_all_years,
    load_hr_fb_pct_all_years,
    load_hard_pct_all_years,
    load_pull_pct_all_years
)

from ..constants import (
    COL_MLBAMID,
    COL_YEAR,
    FUTURE_HITTER_BASE_FEATURES
)

from new_pipeline.common.logging_config import get_logger

logger = get_logger(__name__)


class FutureHitterTransformer(BaseEstimator, TransformerMixin):
    """
    Load hitter features for future season projections.

    Features loaded (14 total before injury):
    - Base features (14): K%, BB%, AVG, OBP, SLG, GDP, Positional_WAR,
                         Enhanced_Baserunning, Enhanced_Defense,
                         ISO, GB%, HR/FB, Hard%, Pull%

    No composite features for hitters.
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
        Load all future season hitter features and add to DataFrame.

        Args:
            X: DataFrame with MLBAMID and Year columns

        Returns:
            DataFrame with 14 hitter features added (before injury features)

        Raises:
            ValueError: If MLBAMID or Year columns missing
        """
        if COL_MLBAMID not in X.columns or COL_YEAR not in X.columns:
            raise ValueError(f"DataFrame must have {COL_MLBAMID} and {COL_YEAR} columns")

        logger.info("Loading future season hitter features...")

        # Get years to load
        years = self.years or X[COL_YEAR].unique().tolist()

        result = X.copy()

        # ==================================================================
        # Load All Base Features
        # ==================================================================

        logger.debug("Loading base features...")

        # Load reused features
        k_pct = load_k_pct_all_years(years)
        bb_pct = load_bb_pct_all_years(years)
        pa = load_pa_all_years(years)
        gdp = load_gdp_all_years(years)
        avg = load_avg_park_adjusted(years)
        obp = load_obp_park_adjusted(years)
        slg = load_slg_park_adjusted(years)
        positional_war = load_positional_war(years)
        enhanced_baserunning = load_enhanced_baserunning(years)
        enhanced_defense = load_enhanced_defense(years)

        # Load new features
        iso = load_iso_all_years(years)
        gb_pct = load_gb_pct_all_years(years)
        hr_fb = load_hr_fb_pct_all_years(years)
        hard_pct = load_hard_pct_all_years(years)
        pull_pct = load_pull_pct_all_years(years)

        # Map to DataFrame
        result['K%'] = result.apply(lambda row: k_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['BB%'] = result.apply(lambda row: bb_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['PA'] = result.apply(lambda row: pa.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['GDP'] = result.apply(lambda row: gdp.get((row[COL_MLBAMID], row[COL_YEAR]), 0), axis=1)
        result['AVG'] = result.apply(lambda row: avg.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['OBP'] = result.apply(lambda row: obp.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['SLG'] = result.apply(lambda row: slg.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['Positional_WAR'] = result.apply(lambda row: positional_war.get(row[COL_MLBAMID], 0.0), axis=1)
        result['Enhanced_Baserunning'] = result.apply(lambda row: enhanced_baserunning.get((row[COL_MLBAMID], row[COL_YEAR]), 0.0), axis=1)
        result['Enhanced_Defense'] = result.apply(lambda row: enhanced_defense.get((row[COL_MLBAMID], row[COL_YEAR]), 0.0), axis=1)
        result['ISO'] = result.apply(lambda row: iso.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['GB%'] = result.apply(lambda row: gb_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['HR/FB'] = result.apply(lambda row: hr_fb.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['Hard%'] = result.apply(lambda row: hard_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)
        result['Pull%'] = result.apply(lambda row: pull_pct.get((row[COL_MLBAMID], row[COL_YEAR]), np.nan), axis=1)

        logger.info(f"Loaded {len(FUTURE_HITTER_BASE_FEATURES)} hitter features")

        return result
