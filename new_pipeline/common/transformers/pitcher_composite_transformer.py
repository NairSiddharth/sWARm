"""
Pitcher Composite Feature Transformer.

Applies composite feature calculations to pitcher DataFrames.
"""
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from ..constants import (
    COL_MLBAMID,
    COL_BB_PCT,
    COL_K_PCT,
    COL_GB_PCT,
    COL_HARD_PCT,
    COL_LOB_PCT,
    COL_HR_FB_PCT,
    COL_AVG_HIT_ANGLE,
    COL_ANGLE_SWEET_SPOT_PCT
)
from ..exceptions import MissingColumnError, InvalidDataTypeError
from ..logging_config import get_logger
from .pitcher_composites import calculate_all_pitcher_composites

logger = get_logger(__name__)


class PitcherCompositeTransformer(BaseEstimator, TransformerMixin):
    """
    Calculate all pitcher composite features.

    Takes DataFrame with base features and adds 6 composite columns:
    - damage_control_ratio
    - Opportunity_Success
    - strikeout_efficiency
    - contact_management
    - strikeout_contact_quality
    - Statcast_Launch_Quality_Index

    Requires base features from PitcherFeatureTransformer to be present.
    """

    def __init__(self):
        """Initialize composite transformer."""
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PitcherCompositeTransformer':
        """Nothing to fit - composites are deterministic calculations."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and add composite features to DataFrame.

        Args:
            X: DataFrame with base pitcher features and MLBAMID column

        Returns:
            DataFrame with 6 additional composite feature columns

        Raises:
            InvalidDataTypeError: If X is not a DataFrame
            MissingColumnError: If required base features are missing
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        # Check for required columns
        required_cols = [
            COL_MLBAMID, COL_BB_PCT, COL_K_PCT, COL_GB_PCT, COL_HARD_PCT,
            COL_LOB_PCT, COL_HR_FB_PCT, COL_AVG_HIT_ANGLE, COL_ANGLE_SWEET_SPOT_PCT
        ]
        missing = [col for col in required_cols if col not in X.columns]
        if missing:
            raise MissingColumnError(missing, X.columns.tolist())

        logger.info("Calculating pitcher composite features...")

        # Convert DataFrame columns to dicts for composite functions
        bb_pct_dict = X.set_index(COL_MLBAMID)[COL_BB_PCT].to_dict()
        k_pct_dict = X.set_index(COL_MLBAMID)[COL_K_PCT].to_dict()
        gb_pct_dict = X.set_index(COL_MLBAMID)[COL_GB_PCT].to_dict()
        hard_pct_dict = X.set_index(COL_MLBAMID)[COL_HARD_PCT].to_dict()
        lob_pct_dict = X.set_index(COL_MLBAMID)[COL_LOB_PCT].to_dict()
        hr_fb_pct_dict = X.set_index(COL_MLBAMID)[COL_HR_FB_PCT].to_dict()

        # Prepare Statcast nested dict
        statcast_dict = {}
        for idx, row in X.iterrows():
            mlbamid = row[COL_MLBAMID]
            if pd.notna(row[COL_AVG_HIT_ANGLE]) and pd.notna(row[COL_ANGLE_SWEET_SPOT_PCT]):
                statcast_dict[mlbamid] = {
                    COL_AVG_HIT_ANGLE: row[COL_AVG_HIT_ANGLE],
                    COL_ANGLE_SWEET_SPOT_PCT: row[COL_ANGLE_SWEET_SPOT_PCT]
                }

        # Calculate all composites
        composites = calculate_all_pitcher_composites(
            bb_pct=bb_pct_dict,
            k_pct=k_pct_dict,
            gb_pct=gb_pct_dict,
            hard_pct=hard_pct_dict,
            lob_pct=lob_pct_dict,
            hr_fb_pct=hr_fb_pct_dict,
            statcast_data=statcast_dict
        )

        # Add composite columns to DataFrame
        result = X.copy()
        for composite_name, composite_values in composites.items():
            result[composite_name] = result[COL_MLBAMID].map(composite_values)

        logger.info(f"Added {len(composites)} composite features")

        return result
