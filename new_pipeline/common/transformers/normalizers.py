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
    COL_WAR_PER_162,
    COL_WAR_PER_600,
    WAR_NORMALIZATION_IP,
    WAR_NORMALIZATION_PA
)
from ..exceptions import MissingColumnError, InvalidDataTypeError
from ..logging_config import get_logger

logger = get_logger(__name__)


class WARNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize WAR to rate stats.

    For pitchers: WAR/162 (per 162 IP)
    For hitters: WAR/600 (per 600 PA)

    This allows fair comparison of players with different usage levels.
    """

    def __init__(self, player_type: str = 'pitcher'):
        """
        Args:
            player_type: 'pitcher' or 'hitter'

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

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'WARNormalizer':
        """Nothing to fit."""
        return self

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

        # Calculate rate stat (avoid division by zero)
        X[self.rate_col_name] = np.where(
            X[self.usage_col] > 0,
            (X[COL_WAR] / X[self.usage_col]) * self.rate_denominator,
            0.0
        )

        logger.info(f"WARNormalizer: Added '{self.rate_col_name}' column")

        return X
