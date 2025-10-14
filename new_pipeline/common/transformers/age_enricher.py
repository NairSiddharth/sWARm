"""
Age Enricher Transformer.

Merges Age column from Baseball Prospectus data using MLBAMID cross-reference.
Falls back to pybaseball for missing ages.
"""
from typing import List, Dict
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime

from ..constants import BP_HITTER_DIR, BP_PITCHER_DIR
from ..exceptions import InvalidDataTypeError
from ..logging_config import get_logger

logger = get_logger(__name__)


class AgeEnricher(BaseEstimator, TransformerMixin):
    """
    Enrich FanGraphs data with Age from Baseball Prospectus.

    FanGraphs CSVs don't include Age, but BP CSVs do.
    Cross-reference using MLBAMID (FanGraphs) = mlbid (BP).

    Args:
        years: Years to load BP data for
        player_type: 'hitter' or 'pitcher'

    Example:
        >>> enricher = AgeEnricher(years=[2024], player_type='hitter')
        >>> df_with_age = enricher.transform(fangraphs_df)
        >>> 'Age' in df_with_age.columns
        True
    """

    def __init__(self, years: List[int], player_type: str):
        """
        Initialize Age enricher.

        Args:
            years: Years to load BP data for
            player_type: 'hitter' or 'pitcher'
        """
        self.years = years
        self.player_type = player_type
        self.age_lookup = {}  # Will be populated in fit()
        self.pybaseball_cache = {}  # Cache for pybaseball lookups

    def fit(self, X: pd.DataFrame, y=None) -> 'AgeEnricher':
        """
        Load BP data and build MLBAMID -> Age lookup.

        Args:
            X: Input DataFrame (not used, but required for sklearn)
            y: Target values (not used)

        Returns:
            self
        """
        # Determine BP data directory
        if self.player_type == 'hitter':
            bp_dir = BP_HITTER_DIR
            file_pattern = "bp_hitters_{year}.csv"
        elif self.player_type == 'pitcher':
            bp_dir = BP_PITCHER_DIR
            file_pattern = "bp_pitchers_{year}.csv"
        else:
            raise ValueError(f"player_type must be 'hitter' or 'pitcher', got: {self.player_type}")

        # Load BP data for all years
        age_data = []
        for year in self.years:
            file_path = bp_dir / file_pattern.format(year=year)

            if not file_path.exists():
                logger.warning(f"BP data not found for {year}: {file_path}")
                continue

            try:
                df = pd.read_csv(file_path)

                # Check required columns
                if 'mlbid' not in df.columns or 'Age' not in df.columns:
                    logger.warning(f"BP data missing required columns (mlbid, Age): {file_path}")
                    continue

                # Extract mlbid -> Age mapping
                for _, row in df.iterrows():
                    if pd.notna(row['mlbid']) and pd.notna(row['Age']):
                        mlbamid = int(float(row['mlbid']))
                        age = int(float(row['Age']))

                        # Most recent year wins
                        self.age_lookup[mlbamid] = age

            except Exception as e:
                logger.warning(f"Error loading BP data for {year}: {e}")
                continue

        logger.info(f"AgeEnricher: Loaded Age for {len(self.age_lookup)} {self.player_type}s")

        return self

    def _get_age_from_pybaseball(self, mlbamid: int, current_year: int) -> int:
        """
        Get player age from pybaseball using playerid_reverse_lookup.

        Args:
            mlbamid: Player's MLBAM ID
            current_year: Current season year

        Returns:
            Player's age, or None if lookup fails
        """
        # Check cache first
        if mlbamid in self.pybaseball_cache:
            return self.pybaseball_cache[mlbamid]

        try:
            from pybaseball import playerid_reverse_lookup

            # Lookup player info by MLBAM ID
            player_info = playerid_reverse_lookup([mlbamid], key_type='mlbam')

            if player_info is not None and not player_info.empty:
                # Get birth year from player info
                if 'birth_year' in player_info.columns:
                    birth_year = player_info['birth_year'].iloc[0]
                    if pd.notna(birth_year):
                        age = current_year - int(birth_year)
                        # Cache the result
                        self.pybaseball_cache[mlbamid] = age
                        return age

        except Exception as e:
            logger.debug(f"Pybaseball lookup failed for MLBAMID {mlbamid}: {e}")

        return None

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add Age column to DataFrame using MLBAMID lookup.

        Args:
            X: DataFrame with MLBAMID column

        Returns:
            DataFrame with Age column added

        Raises:
            InvalidDataTypeError: If X is not a DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        X = X.copy()

        # Find MLBAMID column (handle variations)
        id_col = None
        for col in ['MLBAMID', 'PlayerId', 'playerid']:
            if col in X.columns:
                id_col = col
                break

        if id_col is None:
            logger.warning("AgeEnricher: No player ID column found, returning without Age")
            X['Age'] = None
            return X

        # Determine current year from data (for pybaseball fallback)
        current_year = X['Year'].iloc[0] if 'Year' in X.columns and not X.empty else max(self.years)

        # Add Age column
        ages = []
        missing_from_bp = 0
        pybaseball_successes = 0

        for _, row in X.iterrows():
            if pd.notna(row[id_col]):
                mlbamid = int(float(row[id_col]))
                age = self.age_lookup.get(mlbamid)

                # If BP lookup failed, try pybaseball
                if age is None:
                    missing_from_bp += 1
                    age = self._get_age_from_pybaseball(mlbamid, current_year)
                    if age is not None:
                        pybaseball_successes += 1

                ages.append(age)
            else:
                ages.append(None)

        X['Age'] = ages

        # Log results
        final_missing = X['Age'].isna().sum()
        if missing_from_bp > 0:
            logger.info(f"AgeEnricher: Missing from BP: {missing_from_bp}/{len(X)}, pybaseball filled: {pybaseball_successes}")

        if final_missing > 0:
            logger.warning(f"AgeEnricher: Still missing Age for {final_missing}/{len(X)} players after pybaseball")

            # Fill remaining missing ages with median (if any ages are available)
            if X['Age'].notna().sum() > 0:
                median_age = X['Age'].median()
                X['Age'].fillna(median_age, inplace=True)
                logger.info(f"AgeEnricher: Filled {final_missing} remaining missing ages with median ({median_age:.0f})")

        logger.info(f"AgeEnricher: Added Age column (range: {X['Age'].min():.0f}-{X['Age'].max():.0f})")

        return X
