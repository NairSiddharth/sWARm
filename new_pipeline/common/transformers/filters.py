"""
Filtering transformers for the oWAR pipeline.

All filters follow sklearn's transformer pattern (BaseEstimator, TransformerMixin).
"""
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from ..constants import (
    COL_MLBAMID,
    COL_IP,
    COL_PA,
    COL_GS,
    COL_TWO_WAY_PLAYER,
    MIN_IP_DEFAULT,
    MIN_PA_DEFAULT,
    TWO_WAY_MIN_IP,
    TWO_WAY_MIN_STARTS,
    TWO_WAY_MIN_PA,
    TWO_WAY_OHTANI_MLBAMID
)
from ..exceptions import MissingColumnError, InvalidDataTypeError
from ..logging_config import get_logger

logger = get_logger(__name__)


class NoIDFilter(BaseEstimator, TransformerMixin):
    """
    Remove rows with missing player IDs.

    This ensures we can track predictions back to specific players.
    """

    def __init__(self, id_column: str = COL_MLBAMID):
        """
        Args:
            id_column: Name of the ID column to check
        """
        self.id_column = id_column

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'NoIDFilter':
        """Nothing to fit."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where ID column is null.

        Args:
            X: Input data

        Returns:
            Filtered data

        Raises:
            MissingColumnError: If ID column not found
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        if self.id_column not in X.columns:
            raise MissingColumnError([self.id_column], X.columns.tolist())

        before_count = len(X)
        X_filtered = X[X[self.id_column].notna()].copy()
        after_count = len(X_filtered)

        removed = before_count - after_count
        if removed > 0:
            logger.info(f"NoIDFilter: Removed {removed} rows with missing {self.id_column}")

        return X_filtered


class IPFilter(BaseEstimator, TransformerMixin):
    """
    Filter pitchers using multi-criteria qualification.

    Excludes position players (spot pitchers in blowouts) while including:
    - All starters (any GS > 0)
    - Meaningful relievers (10+ G and 10+ IP)
    - High-volume relievers (20+ G)
    - Two-way players (20+ IP per MLB designation)

    Auto-adjusts thresholds for partial seasons (firsthalf/quarter).
    """

    def __init__(self, detect_partial_season: bool = True):
        """
        Args:
            detect_partial_season: Auto-adjust thresholds for partial seasons
        """
        self.detect_partial_season = detect_partial_season

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'IPFilter':
        """Nothing to fit."""
        return self

    def _detect_partial_season(self, X: pd.DataFrame) -> bool:
        """
        Detect if data represents a partial season.

        Currently uses simple heuristic: If mean IP < 40, likely partial season.
        Future: Could check Year column or CSV filename patterns.

        Args:
            X: Input data

        Returns:
            bool: True if partial season detected
        """
        if COL_IP in X.columns and len(X) > 0:
            mean_ip = X[COL_IP].mean()
            return mean_ip < 40
        return False

    def _is_qualified_pitcher(self, row: pd.Series, is_partial: bool) -> bool:
        """
        Multi-criteria pitcher qualification.

        Args:
            row: DataFrame row
            is_partial: Whether data is from partial season

        Returns:
            bool: True if pitcher qualifies
        """
        ip = row.get(COL_IP, 0)
        gs = row.get(COL_GS, 0)
        g = row.get('G', 0)
        is_two_way = row.get(COL_TWO_WAY_PLAYER, False)

        # Two-way players: Use MLB threshold (20 IP)
        if is_two_way:
            return ip >= TWO_WAY_MIN_IP

        # Adjust thresholds for partial seasons
        if is_partial:
            min_games_threshold = 5
            min_ip_threshold = 5
        else:
            min_games_threshold = 10
            min_ip_threshold = 10

        # Any starter qualifies (position players don't start games)
        if gs > 0:
            return True

        # Relievers need meaningful volume
        if g >= min_games_threshold and ip >= min_ip_threshold:
            return True

        # High-volume relievers (even if low IP/G)
        if g >= min_games_threshold * 2:
            return True

        return False

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter pitchers using multi-criteria qualification.

        Args:
            X: Input data with IP, GS, G columns

        Returns:
            Filtered data

        Raises:
            MissingColumnError: If required columns not found
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        required_cols = [COL_IP, COL_GS]
        missing = [col for col in required_cols if col not in X.columns]
        if missing:
            raise MissingColumnError(missing, X.columns.tolist())

        # Detect partial season
        is_partial = self._detect_partial_season(X) if self.detect_partial_season else False

        # Apply multi-criteria filter
        before_count = len(X)
        mask = X.apply(lambda row: self._is_qualified_pitcher(row, is_partial), axis=1)
        X_filtered = X[mask].copy()
        after_count = len(X_filtered)

        removed = before_count - after_count
        season_type = "partial season" if is_partial else "full season"
        if removed > 0:
            logger.info(f"IPFilter: Removed {removed} pitchers (position players / insufficient sample, {season_type})")

        return X_filtered


class PAFilter(BaseEstimator, TransformerMixin):
    """
    Filter hitters by minimum plate appearances.

    Automatically adjusts threshold for partial seasons (firsthalf/quarter).
    Two-way players automatically qualify (they have 60+ PA by MLB definition).
    """

    def __init__(self, min_pa: int = MIN_PA_DEFAULT, detect_partial_season: bool = True):
        """
        Args:
            min_pa: Minimum plate appearances threshold (full season)
            detect_partial_season: Auto-adjust threshold for partial seasons
        """
        self.min_pa = min_pa
        self.detect_partial_season = detect_partial_season

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PAFilter':
        """Nothing to fit."""
        return self

    def _detect_partial_season(self, X: pd.DataFrame) -> bool:
        """
        Detect if data represents a partial season.

        Uses heuristic: If mean PA < 200, likely partial season.

        Args:
            X: Input data

        Returns:
            bool: True if partial season detected
        """
        if COL_PA in X.columns and len(X) > 0:
            mean_pa = X[COL_PA].mean()
            return mean_pa < 200
        return False

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove hitters below PA threshold (with partial season adjustment).

        Args:
            X: Input data with PA column

        Returns:
            Filtered data

        Raises:
            MissingColumnError: If PA column not found
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        if COL_PA not in X.columns:
            raise MissingColumnError([COL_PA], X.columns.tolist())

        # Detect partial season
        is_partial = self._detect_partial_season(X) if self.detect_partial_season else False

        # Adjust threshold for partial seasons
        if is_partial:
            adjusted_min_pa = self.min_pa // 2  # Half threshold for partial season
        else:
            adjusted_min_pa = self.min_pa

        before_count = len(X)

        # Apply filter (two-way players automatically pass if two_way_player column exists)
        if COL_TWO_WAY_PLAYER in X.columns:
            # Two-way players qualify regardless (they have 60+ PA by definition)
            mask = (X[COL_PA] >= adjusted_min_pa) | (X[COL_TWO_WAY_PLAYER] == True)
            X_filtered = X[mask].copy()
        else:
            X_filtered = X[X[COL_PA] >= adjusted_min_pa].copy()

        after_count = len(X_filtered)

        removed = before_count - after_count
        season_type = "partial season" if is_partial else "full season"
        if removed > 0:
            logger.info(f"PAFilter: Removed {removed} hitters with < {adjusted_min_pa} PA ({season_type})")

        return X_filtered


class TwoWayPlayerFilter(BaseEstimator, TransformerMixin):
    """
    Identify and mark two-way players (e.g., Ohtani).

    MLB two-way player criteria (approximate):
    - Pitcher: >= 20 IP
    - Hitter: >= 20 starts as position player/DH with >= 3 PA per start (= 60 PA minimum)

    Rule took effect in 2020 season.
    Currently, only Shohei Ohtani (MLBAMID: 660271) should qualify.

    This transformer adds a 'two_way_player' boolean column and performs
    a sanity check - if anyone besides Ohtani qualifies, log a warning.
    """

    def __init__(
        self,
        min_ip: float = TWO_WAY_MIN_IP,
        min_starts: int = TWO_WAY_MIN_STARTS,
        min_pa: int = TWO_WAY_MIN_PA
    ):
        """
        Args:
            min_ip: Minimum IP to qualify as pitcher
            min_starts: Minimum games started as position player/DH
            min_pa: Minimum plate appearances (implies ~3 PA per start)
        """
        self.min_ip = min_ip
        self.min_starts = min_starts
        self.min_pa = min_pa

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TwoWayPlayerFilter':
        """Nothing to fit."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'two_way_player' column with sanity check.

        Args:
            X: Data with IP, GS, and PA columns

        Returns:
            Data with added two_way_player column
        """
        if not isinstance(X, pd.DataFrame):
            raise InvalidDataTypeError('X', 'pd.DataFrame', type(X))

        # Check if we have necessary columns
        has_ip = COL_IP in X.columns
        has_gs = COL_GS in X.columns
        has_pa = COL_PA in X.columns

        X = X.copy()

        if has_ip and has_gs and has_pa:
            # Apply MLB two-way criteria (approximate)
            meets_criteria = (
                (X[COL_IP] >= self.min_ip) &
                (X[COL_GS] >= self.min_starts) &
                (X[COL_PA] >= self.min_pa)
            )

            X[COL_TWO_WAY_PLAYER] = meets_criteria

            # Sanity check: Should only be Ohtani (MLBAMID: 660271)
            qualified_players = X[meets_criteria]

            if len(qualified_players) > 0:
                logger.info(f"TwoWayPlayerFilter: {len(qualified_players)} player(s) qualified")

                # Check if qualified players are only Ohtani
                if COL_MLBAMID in X.columns:
                    expected_only_ohtani = qualified_players[COL_MLBAMID].isin([TWO_WAY_OHTANI_MLBAMID])

                    if not expected_only_ohtani.all():
                        unexpected = qualified_players[~expected_only_ohtani]
                        logger.warning(
                            f"Unexpected two-way players found (should only be Ohtani MLBAMID={TWO_WAY_OHTANI_MLBAMID}):\n"
                            f"{unexpected[[COL_MLBAMID, COL_IP, COL_GS, COL_PA]].to_string()}\n"
                            "This may indicate data quality issues or incorrect criteria"
                        )
        else:
            # Can't determine two-way status without necessary columns
            # This is expected when processing pitchers/hitters separately
            X[COL_TWO_WAY_PLAYER] = False
            logger.debug(
                f"TwoWayPlayerFilter: Missing required columns ({COL_IP}, {COL_GS}, {COL_PA}), "
                "marking all as False (expected when processing single player type)"
            )

        return X
