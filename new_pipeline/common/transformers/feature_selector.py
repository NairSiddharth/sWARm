"""
Feature Selector Transformer.

Selects specified modeling features from DataFrame while preserving metadata columns.
"""
from typing import List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from ..exceptions import MissingColumnError, InvalidDataTypeError
from ..logging_config import get_logger

logger = get_logger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select modeling features and metadata from DataFrame.

    This transformer filters pipeline output to include only:
    - Specified modeling features (in order for model input)
    - Metadata columns for tracking and analysis

    The feature order is preserved exactly as specified, which is critical
    for models using monotonic constraints or other order-dependent logic.

    Args:
        feature_columns: List of modeling feature names in exact order
        keep_metadata: Whether to keep metadata columns (default: True)

    Metadata Columns Preserved:
        - MLBAMID: Player identifier
        - Name: Player name
        - Team: Team abbreviation
        - Year: Season year
        - two_way_player: Boolean flag
        - GS, G: Games started, games played
        - IP, PA: Innings pitched, plate appearances
        - WAR: Raw WAR
        - WAR_per_162, WAR_per_600: Normalized WAR targets
        - _multi_team_current: Current team for multi-team players
        - _multi_team_stints: JSON stint data for weighted calculations

    Example:
        >>> selector = FeatureSelector(
        ...     feature_columns=['BB%', 'K%', 'ERA'],
        ...     keep_metadata=True
        ... )
        >>> filtered_df = selector.transform(pipeline_output_df)
        >>> # filtered_df has only modeling features + metadata
    """

    def __init__(self, feature_columns: List[str], keep_metadata: bool = True):
        """
        Initialize feature selector.

        Args:
            feature_columns: List of feature names to select (order preserved)
            keep_metadata: If True, keep metadata columns alongside features
        """
        self.feature_columns = feature_columns
        self.keep_metadata = keep_metadata

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelector':
        """
        Fit method (no-op for feature selection).

        Args:
            X: Input DataFrame
            y: Target values (ignored)

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select specified features and metadata from DataFrame.

        Args:
            X: DataFrame with all pipeline features

        Returns:
            DataFrame with only selected features + metadata

        Raises:
            InvalidDataTypeError: If X is not a DataFrame
            MissingColumnError: If any specified features are missing
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        # Check all features exist
        missing = [f for f in self.feature_columns if f not in X.columns]
        if missing:
            raise MissingColumnError(missing, X.columns.tolist())

        if self.keep_metadata:
            # Metadata columns to preserve (if they exist)
            metadata_cols = [
                'MLBAMID', 'Name', 'Team', 'Year', 'Age', 'Position', 'Primary_Position', 'two_way_player',
                'GS', 'G', 'IP', 'PA', 'WAR', 'WAR_per_162', 'WAR_per_600',
                '_multi_team_current', '_multi_team_stints'  # Multi-team player metadata
            ]

            # Keep only metadata that exists in input
            existing_metadata = [c for c in metadata_cols if c in X.columns]

            # Combine: metadata + features (preserving feature order)
            keep_cols = existing_metadata + self.feature_columns

            logger.info(f"FeatureSelector: Selected {len(self.feature_columns)} features + {len(existing_metadata)} metadata columns")
        else:
            # Features only
            keep_cols = self.feature_columns
            logger.info(f"FeatureSelector: Selected {len(self.feature_columns)} features (no metadata)")

        # Return filtered DataFrame
        return X[keep_cols]
