"""
Hitter feature transformer for the oWAR pipeline.

Loads all 10 hitter features and maps them to DataFrame columns.
"""
from typing import Optional, List

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from ..constants import (
    COL_MLBAMID,
    COL_YEAR,
    COL_K_PCT,
    COL_BB_PCT,
    COL_PA,
    COL_GDP,
    COL_AVG,
    COL_OBP,
    COL_SLG,
    COL_POSITIONAL_WAR,
    COL_ENHANCED_BASERUNNING,
    COL_ENHANCED_DEFENSE
)
from ..exceptions import MissingColumnError, InvalidDataTypeError
from ..logging_config import get_logger
from ..loaders.hitter_loaders import (
    load_k_pct_all_years,
    load_bb_pct_all_years,
    load_pa_all_years,
    load_gdp_all_years,
    load_avg_park_adjusted,
    load_obp_park_adjusted,
    load_slg_park_adjusted,
    load_positional_war,
    load_enhanced_baserunning,
    load_enhanced_defense
)

logger = get_logger(__name__)


class HitterFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Load and combine all 10 hitter features.

    This transformer takes raw hitter data (with MLBAMID column)
    and adds all hitter features by loading them from the loaders
    and mapping via MLBAMID.
    """

    def __init__(self, years: Optional[List[int]] = None):
        """
        Args:
            years: Years to load features for. If None, use years from data.
        """
        self.years = years

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'HitterFeatureTransformer':
        """Nothing to fit - features are loaded from raw data."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Load all hitter features and map to DataFrame.

        Args:
            X: Data with MLBAMID column

        Returns:
            Data with all 10 hitter features added

        Raises:
            InvalidDataTypeError: If X is not a DataFrame
            MissingColumnError: If required columns not found
            ValueError: If years cannot be determined
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        if COL_MLBAMID not in X.columns:
            raise MissingColumnError([COL_MLBAMID], X.columns.tolist())

        logger.info("Loading hitter features...")

        # Get years to load
        years = self.years
        if years is None and COL_YEAR in X.columns:
            years = X[COL_YEAR].unique().tolist()
        elif years is None:
            raise ValueError(f"Must provide years parameter or have '{COL_YEAR}' column in data")

        # Start with copy of input data
        result = X.copy()

        # Load each feature and map to DataFrame
        # 1. K%
        logger.debug(f"Loading {COL_K_PCT}...")
        k_pct_dict = load_k_pct_all_years(years)
        result[COL_K_PCT] = result[COL_MLBAMID].map(k_pct_dict)

        # 2. BB%
        logger.debug(f"Loading {COL_BB_PCT}...")
        bb_pct_dict = load_bb_pct_all_years(years)
        result[COL_BB_PCT] = result[COL_MLBAMID].map(bb_pct_dict)

        # 3. PA
        logger.debug(f"Loading {COL_PA}...")
        pa_dict = load_pa_all_years(years)
        result[COL_PA] = result[COL_MLBAMID].map(pa_dict)

        # 4. GDP
        logger.debug(f"Loading {COL_GDP}...")
        gdp_dict = load_gdp_all_years(years)
        result[COL_GDP] = result[COL_MLBAMID].map(gdp_dict)

        # 5. AVG (park-adjusted)
        logger.debug(f"Loading {COL_AVG} (park-adjusted)...")
        avg_dict = load_avg_park_adjusted(years)
        result[COL_AVG] = result[COL_MLBAMID].map(avg_dict)

        # 6. OBP (park-adjusted)
        logger.debug(f"Loading {COL_OBP} (park-adjusted)...")
        obp_dict = load_obp_park_adjusted(years)
        result[COL_OBP] = result[COL_MLBAMID].map(obp_dict)

        # 7. SLG (park-adjusted)
        logger.debug(f"Loading {COL_SLG} (park-adjusted)...")
        slg_dict = load_slg_park_adjusted(years)
        result[COL_SLG] = result[COL_MLBAMID].map(slg_dict)

        # 8. Positional_WAR
        logger.debug(f"Loading {COL_POSITIONAL_WAR}...")
        pos_war_dict = load_positional_war(years)
        result[COL_POSITIONAL_WAR] = result[COL_MLBAMID].map(pos_war_dict)

        # 9. Enhanced_Baserunning
        logger.debug(f"Loading {COL_ENHANCED_BASERUNNING}...")
        baserunning_dict = load_enhanced_baserunning(years)
        result[COL_ENHANCED_BASERUNNING] = result[COL_MLBAMID].map(baserunning_dict)

        # 10. Enhanced_Defense
        logger.debug(f"Loading {COL_ENHANCED_DEFENSE}...")
        defense_dict = load_enhanced_defense(years)
        result[COL_ENHANCED_DEFENSE] = result[COL_MLBAMID].map(defense_dict)

        logger.info(f"Loaded 10 hitter feature sets ({len(result.columns)} total columns)")

        return result
