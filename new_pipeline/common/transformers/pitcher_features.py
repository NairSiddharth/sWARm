"""
Pitcher feature transformer for the oWAR pipeline.

Loads all 11 pitcher features and maps them to DataFrame columns.
"""
from typing import Optional, List

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from ..constants import (
    COL_MLBAMID,
    COL_YEAR,
    COL_BB_PCT,
    COL_K_PCT,
    COL_SWSTR_PCT,
    COL_WPA_LI,
    COL_SD,
    COL_MD,
    COL_LOB_PCT,
    COL_HARD_PCT,
    COL_ERA,
    COL_GB_PCT,
    COL_HR_FB_PCT,
    COL_AVG_HIT_ANGLE,
    COL_ANGLE_SWEET_SPOT_PCT,
    COL_RUNNING_CONTROL
)
from ..exceptions import MissingColumnError, InvalidDataTypeError
from ..logging_config import get_logger
from ..loaders.pitcher_loaders import (
    load_bb_pct_all_years,
    load_k_pct_all_years,
    load_swstr_all_years,
    load_wpa_li_all_years,
    load_sd_all_years,
    load_md_all_years,
    load_lob_pct_all_years,
    load_hard_pct_all_years,
    load_era_park_adjusted,
    load_gb_pct_park_adjusted,
    load_hr_fb_pct_park_adjusted,
    load_statcast_data,
    load_running_control_all_years
)

logger = get_logger(__name__)


class PitcherFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Load and combine all 13 pitcher features.

    This transformer takes raw pitcher data (with MLBAMID column)
    and adds all pitcher features by loading them from the loaders
    and mapping via MLBAMID.
    """

    def __init__(self, years: Optional[List[int]] = None):
        """
        Args:
            years: Years to load features for. If None, use years from data.
        """
        self.years = years

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PitcherFeatureTransformer':
        """Nothing to fit - features are loaded from raw data."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Load all pitcher features and map to DataFrame.

        Args:
            X: Data with MLBAMID column

        Returns:
            Data with all 13 pitcher features added

        Raises:
            InvalidDataTypeError: If X is not a DataFrame
            MissingColumnError: If required columns not found
            ValueError: If years cannot be determined
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        if COL_MLBAMID not in X.columns:
            raise MissingColumnError([COL_MLBAMID], X.columns.tolist())

        logger.info("Loading pitcher features...")

        # Get years to load
        years = self.years
        if years is None and COL_YEAR in X.columns:
            years = X[COL_YEAR].unique().tolist()
        elif years is None:
            raise ValueError(f"Must provide years parameter or have '{COL_YEAR}' column in data")

        # Start with copy of input data
        result = X.copy()

        # Load each feature and map to DataFrame
        # 1. BB%
        logger.debug(f"Loading {COL_BB_PCT}...")
        bb_pct_dict = load_bb_pct_all_years(years)
        result[COL_BB_PCT] = result.apply(lambda row: bb_pct_dict.get((row[COL_MLBAMID], row[COL_YEAR])), axis=1)

        # 2. K%
        logger.debug(f"Loading {COL_K_PCT}...")
        k_pct_dict = load_k_pct_all_years(years)
        result[COL_K_PCT] = result.apply(lambda row: k_pct_dict.get((row[COL_MLBAMID], row[COL_YEAR])), axis=1)

        # 3. SwStr%
        logger.debug(f"Loading {COL_SWSTR_PCT}...")
        swstr_dict = load_swstr_all_years(years)
        result[COL_SWSTR_PCT] = result.apply(lambda row: swstr_dict.get((row[COL_MLBAMID], row[COL_YEAR])), axis=1)

        # 4. WPA/LI
        logger.debug(f"Loading {COL_WPA_LI}...")
        wpa_li_dict = load_wpa_li_all_years(years)
        result[COL_WPA_LI] = result.apply(lambda row: wpa_li_dict.get((row[COL_MLBAMID], row[COL_YEAR])), axis=1)

        # 5. SD (for SD_MD_Net composite)
        logger.debug(f"Loading {COL_SD}...")
        sd_dict = load_sd_all_years(years)
        result[COL_SD] = result.apply(lambda row: sd_dict.get((row[COL_MLBAMID], row[COL_YEAR])), axis=1)

        # 6. MD (for SD_MD_Net composite)
        logger.debug(f"Loading {COL_MD}...")
        md_dict = load_md_all_years(years)
        result[COL_MD] = result.apply(lambda row: md_dict.get((row[COL_MLBAMID], row[COL_YEAR])), axis=1)

        # 7. LOB%
        logger.debug(f"Loading {COL_LOB_PCT}...")
        lob_pct_dict = load_lob_pct_all_years(years)
        result[COL_LOB_PCT] = result.apply(lambda row: lob_pct_dict.get((row[COL_MLBAMID], row[COL_YEAR])), axis=1)

        # 8. Hard%
        logger.debug(f"Loading {COL_HARD_PCT}...")
        hard_pct_dict = load_hard_pct_all_years(years)
        result[COL_HARD_PCT] = result.apply(lambda row: hard_pct_dict.get((row[COL_MLBAMID], row[COL_YEAR])), axis=1)

        # 9. ERA (park-adjusted)
        logger.debug(f"Loading {COL_ERA} (park-adjusted)...")
        era_dict = load_era_park_adjusted(years)
        result[COL_ERA] = result.apply(lambda row: era_dict.get((row[COL_MLBAMID], row[COL_YEAR])), axis=1)

        # 10. GB% (park-adjusted)
        logger.debug(f"Loading {COL_GB_PCT} (park-adjusted)...")
        gb_pct_dict = load_gb_pct_park_adjusted(years)
        result[COL_GB_PCT] = result.apply(lambda row: gb_pct_dict.get((row[COL_MLBAMID], row[COL_YEAR])), axis=1)

        # 11. HR/FB% (park-adjusted)
        logger.debug(f"Loading {COL_HR_FB_PCT} (park-adjusted)...")
        hr_fb_pct_dict = load_hr_fb_pct_park_adjusted(years)
        result[COL_HR_FB_PCT] = result.apply(lambda row: hr_fb_pct_dict.get((row[COL_MLBAMID], row[COL_YEAR])), axis=1)

        # 12. Statcast Launch Quality Index
        logger.debug("Loading Statcast Launch Quality Index...")
        statcast_dict = load_statcast_data(years)

        # Statcast returns nested dict, need to extract components
        if statcast_dict:
            result[COL_AVG_HIT_ANGLE] = result[COL_MLBAMID].map(
                lambda pid: statcast_dict.get(pid, {}).get(COL_AVG_HIT_ANGLE)
            )
            result[COL_ANGLE_SWEET_SPOT_PCT] = result[COL_MLBAMID].map(
                lambda pid: statcast_dict.get(pid, {}).get(COL_ANGLE_SWEET_SPOT_PCT)
            )

        # 13. Running_Control
        logger.debug(f"Loading {COL_RUNNING_CONTROL}...")
        running_control_dict = load_running_control_all_years(years)
        result[COL_RUNNING_CONTROL] = result[COL_MLBAMID].map(running_control_dict)

        logger.info(f"Loaded 13 pitcher feature sets ({len(result.columns)} total columns)")

        return result
