"""
Normalization transformers for the oWAR pipeline.

All normalizers follow sklearn's transformer pattern (BaseEstimator, TransformerMixin).
"""
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from ..constants import (
    COL_WAR,
    COL_IP,
    COL_PA,
    COL_GS,
    COL_WAR_PER_162,
    COL_WAR_PER_600,
    WAR_NORMALIZATION_IP,
    WAR_NORMALIZATION_PA,
    WAR_NORMALIZATION_IP_STARTER,
    WAR_NORMALIZATION_IP_RELIEVER,
    WAR_NORMALIZATION_IP_SWING,
    PITCHER_STARTER_THRESHOLD,
    PITCHER_RELIEVER_THRESHOLD
)
from ..exceptions import MissingColumnError, InvalidDataTypeError
from ..logging_config import get_logger

logger = get_logger(__name__)


class WARNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize WAR to rate stats.

    For pitchers:
        - Role-specific normalization based on GS/G ratio
        - Starters (GS/G > 0.7): WAR per 162 IP
        - Relievers (GS/G < 0.1): WAR per 48.2 IP (FanGraphs qualification)
        - Swing (0.1 <= GS/G <= 0.7): WAR per 110 IP

    For hitters: WAR/600 (per 600 PA)

    This allows fair comparison while preventing extrapolation issues for relievers.
    """

    def __init__(self, player_type: str = 'pitcher', use_role_specific: bool = True):
        """
        Args:
            player_type: 'pitcher' or 'hitter'
            use_role_specific: If True and player_type='pitcher', use role-specific
                             normalization based on GS/G ratio (default: True)

        Raises:
            ValueError: If player_type is not 'pitcher' or 'hitter'
        """
        match player_type:
            case 'pitcher':
                self.usage_col = COL_IP
                self.rate_denominator = WAR_NORMALIZATION_IP
                self.rate_col_name = COL_WAR_PER_162
            case 'hitter':
                self.usage_col = COL_PA
                self.rate_denominator = WAR_NORMALIZATION_PA
                self.rate_col_name = COL_WAR_PER_600
            case _:
                raise ValueError(f"player_type must be 'pitcher' or 'hitter', got {player_type!r}")

        self.player_type = player_type
        self.use_role_specific = use_role_specific

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'WARNormalizer':
        """Nothing to fit."""
        return self

    def _get_role_specific_denominator(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate role-specific WAR normalization denominators for pitchers.

        Uses GS/G ratio to determine pitcher role:
        - Starter (GS/G > 0.7): 162 IP
        - Reliever (GS/G < 0.1): 48.2 IP
        - Swing (0.1 <= GS/G <= 0.7): 110 IP

        Args:
            X: DataFrame with 'GS' and 'G' columns

        Returns:
            Array of denominators for each pitcher
        """
        # Check if we have GS column (needed for role detection)
        if 'G' not in X.columns:
            # Fall back to default if G column not available
            logger.warning("Column 'G' not found - using default IP normalization (162)")
            return np.full(len(X), WAR_NORMALIZATION_IP)

        # Calculate GS/G ratio (avoid division by zero)
        gs_per_g = np.where(
            X['G'] > 0,
            X.get(COL_GS, 0) / X['G'],
            0.0
        )

        # Determine role-specific denominator for each pitcher
        denominators = np.select(
            [
                gs_per_g > PITCHER_STARTER_THRESHOLD,  # Starters
                gs_per_g < PITCHER_RELIEVER_THRESHOLD,  # Relievers
            ],
            [
                WAR_NORMALIZATION_IP_STARTER,   # 162
                WAR_NORMALIZATION_IP_RELIEVER,  # 48.2
            ],
            default=WAR_NORMALIZATION_IP_SWING  # 110 (swing)
        )

        return denominators

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add WAR rate column.

        Args:
            X: Data with WAR and usage column (IP or PA)

        Returns:
            Data with added rate column

        Raises:
            MissingColumnError: If required columns not found
            InvalidDataTypeError: If X is not a DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        # Validate required columns
        missing_cols = []
        if COL_WAR not in X.columns:
            missing_cols.append(COL_WAR)
        if self.usage_col not in X.columns:
            missing_cols.append(self.usage_col)

        if missing_cols:
            raise MissingColumnError(missing_cols, X.columns.tolist())

        X = X.copy()

        # Determine denominator (role-specific for pitchers if enabled)
        if self.player_type == 'pitcher' and self.use_role_specific:
            # Use role-specific denominators for pitchers
            denominators = self._get_role_specific_denominator(X)

            # Calculate rate stat with role-specific denominators
            X[self.rate_col_name] = np.where(
                X[self.usage_col] > 0,
                (X[COL_WAR] / X[self.usage_col]) * denominators,
                0.0
            )

            # Count roles for logging
            gs_per_g = np.where(X['G'] > 0, X.get(COL_GS, 0) / X['G'], 0.0)
            n_starters = (gs_per_g > PITCHER_STARTER_THRESHOLD).sum()
            n_relievers = (gs_per_g < PITCHER_RELIEVER_THRESHOLD).sum()
            n_swing = ((gs_per_g >= PITCHER_RELIEVER_THRESHOLD) & (gs_per_g <= PITCHER_STARTER_THRESHOLD)).sum()

            logger.info(
                f"WARNormalizer: Added '{self.rate_col_name}' column "
                f"(role-specific: {n_starters} starters/162IP, "
                f"{n_relievers} relievers/48.2IP, {n_swing} swing/110IP)"
            )
        else:
            # Use single denominator (default or hitters)
            X[self.rate_col_name] = np.where(
                X[self.usage_col] > 0,
                (X[COL_WAR] / X[self.usage_col]) * self.rate_denominator,
                0.0
            )

            # Clip outliers for hitters to prevent training on noise
            if self.player_type == 'hitter':
                X[self.rate_col_name] = X[self.rate_col_name].clip(-5, 14)

            logger.info(f"WARNormalizer: Added '{self.rate_col_name}' column")

        return X
