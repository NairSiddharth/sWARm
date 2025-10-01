"""
Injury Recovery Calculator for Current Season Projections

Adapted from future_season_modules.integration.py for current season use.
Calculates injury recovery adjustments for participation rates and performance.
"""

# Standard library imports
import os
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import numpy as np

# Local application imports
from common_modules.config import DATA_DIR, FANGRAPHS_DATA_DIR
from common_modules.logging import get_logger

# Module logger
logger = get_logger(__name__)


class InjuryRecoveryCalculator:
    """
    Calculate injury recovery adjustments for current season projections.

    Loads 2025 injury data and applies recovery factors based on injury type,
    return date, and recovery timeline.
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize injury recovery calculator.

        Args:
            data_dir: Path to MLB Player Data directory
        """
        if data_dir is None:
            data_dir = str(DATA_DIR)

        self.data_dir = data_dir
        self.fg_data_path = str(FANGRAPHS_DATA_DIR)
        self.injury_data = None

        # Injury recovery coefficients (from validated future_season_modules)
        self.recovery_coefficients = {
            # Major surgeries - longer recovery periods
            'tommy_john': {
                'recovery_factor': 0.85,
                'recovery_duration_days': 365,
                'description': 'Tommy John surgery recovery'
            },
            'shoulder_surgery': {
                'recovery_factor': 0.90,
                'recovery_duration_days': 180,
                'description': 'Shoulder surgery recovery'
            },
            'hip_surgery': {
                'recovery_factor': 0.88,
                'recovery_duration_days': 120,
                'description': 'Hip surgery recovery'
            },
            'elbow_internal_brace': {
                'recovery_factor': 0.92,
                'recovery_duration_days': 90,
                'description': 'Elbow internal brace surgery'
            },
            'other_surgery': {
                'recovery_factor': 0.93,
                'recovery_duration_days': 90,
                'description': 'Other surgical procedure'
            },

            # Non-surgical injuries - shorter recovery periods
            'oblique_strain': {
                'recovery_factor': 0.95,
                'recovery_duration_days': 60,
                'description': 'Oblique strain recovery'
            },
            'hamstring_strain': {
                'recovery_factor': 0.95,
                'recovery_duration_days': 45,
                'description': 'Hamstring strain recovery'
            },
            'shoulder_strain': {
                'recovery_factor': 0.96,
                'recovery_duration_days': 30,
                'description': 'Shoulder strain recovery'
            },
            'back_strain': {
                'recovery_factor': 0.94,
                'recovery_duration_days': 45,
                'description': 'Back strain recovery'
            },
            'groin_strain': {
                'recovery_factor': 0.96,
                'recovery_duration_days': 30,
                'description': 'Groin strain recovery'
            },

            # Default for unknown injuries
            'unknown': {
                'recovery_factor': 0.98,
                'recovery_duration_days': 21,
                'description': 'General injury recovery'
            }
        }

    def load_current_season_injury_data(self, year: int = 2025) -> pd.DataFrame:
        """
        Load current season injury data.

        Args:
            year: Year to load (default 2025)

        Returns:
            Processed injury data
        """
        logger.info(f"Loading {year} injury data...")

        injury_path = os.path.join(self.fg_data_path, "injuries")
        file_path = os.path.join(injury_path, f"fangraphs_injuryreport_{year}.xlsx")

        if not os.path.exists(file_path):
            logger.warning(f"  Injury file not found: {file_path}")
            return pd.DataFrame()

        try:
            df = pd.read_excel(file_path)
            df['data_year'] = year
            logger.info(f"  Loaded {len(df)} injury records from {year}")

            # Process the data
            processed_injuries = self._process_injury_data(df)

            logger.info(f"Total injury records processed: {len(processed_injuries)}")
            logger.info(f"Unique players with injuries: {processed_injuries['MLBAMID'].nunique()}")

            # Store for use
            self.injury_data = processed_injuries
            return processed_injuries

        except Exception as e:
            logger.error(f"  Error loading {year} injury data: {e}", exc_info=True)
            return pd.DataFrame()

    def _process_injury_data(self, raw_injury_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and standardize raw injury data for modeling.

        Args:
            raw_injury_data: Raw injury data from FanGraphs Excel files

        Returns:
            Processed injury data with standardized classifications
        """
        processed = raw_injury_data.copy()

        # Standardize injury classifications
        processed['injury_type'] = processed['Injury / Surgery'].apply(self._classify_injury_type)

        # Convert dates with explicit format specification
        processed['injury_date'] = pd.to_datetime(
            processed['Injury / Surgery Date'], format='mixed', errors='coerce')
        processed['return_date'] = pd.to_datetime(
            processed['Return Date'], format='mixed', errors='coerce')
        processed['eligible_date'] = pd.to_datetime(
            processed['Eligible to Return'], format='mixed', errors='coerce')

        # Calculate recovery time
        recovery_mask = processed['injury_date'].notna() & processed['return_date'].notna()
        processed.loc[recovery_mask, 'recovery_days'] = (
            processed.loc[recovery_mask, 'return_date'] - processed.loc[recovery_mask, 'injury_date']
        ).dt.days

        # Clean up columns
        processed = processed.rename(columns={
            'Name': 'Name',
            'Pos': 'Position',
            'MLBAMID': 'MLBAMID'
        })

        # Add mlbid mapping
        processed['mlbid'] = processed['MLBAMID']

        # Keep only relevant columns
        keep_columns = [
            'mlbid',
            'Name',
            'Position',
            'injury_date',
            'injury_type',
            'return_date',
            'eligible_date',
            'recovery_days',
            'data_year',
            'MLBAMID',
            'Status']

        # Only keep columns that exist in the data
        keep_columns = [col for col in keep_columns if col in processed.columns]
        processed = processed[keep_columns].copy()

        return processed

    def _classify_injury_type(self, injury_description: str) -> str:
        """
        Classify injury descriptions into standardized categories.

        Args:
            injury_description: Raw injury description from FanGraphs

        Returns:
            Standardized injury type classification
        """
        if pd.isna(injury_description):
            return 'unknown'

        injury_lower = str(injury_description).lower()

        # Tommy John surgery (highest priority)
        if 'tommy john' in injury_lower:
            return 'tommy_john'

        # Other major surgeries
        elif 'shoulder surgery' in injury_lower:
            return 'shoulder_surgery'
        elif 'hip surgery' in injury_lower:
            return 'hip_surgery'
        elif 'thoracic outlet' in injury_lower:
            return 'thoracic_outlet'
        elif 'knee surgery' in injury_lower:
            return 'knee_surgery'
        elif 'elbow surgery' in injury_lower and 'internal brace' in injury_lower:
            return 'elbow_internal_brace'
        elif 'surgery' in injury_lower:
            return 'other_surgery'

        # Non-surgical injuries by body part
        elif 'hamstring' in injury_lower:
            return 'hamstring_strain'
        elif 'oblique' in injury_lower:
            return 'oblique_strain'
        elif 'shoulder' in injury_lower and ('strain' in injury_lower or 'inflammation' in injury_lower):
            return 'shoulder_strain'
        elif 'groin' in injury_lower:
            return 'groin_strain'
        elif 'back' in injury_lower:
            return 'back_strain'
        elif 'calf' in injury_lower:
            return 'calf_strain'

        else:
            return 'unknown'

    def apply_injury_curve_to_defense(self,
                                      enhanced_defense: float,
                                      player_id: Union[int, str],
                                      current_date: datetime = None,
                                      lookback_days: int = 60,
                                      exclude_pitching_injuries: bool = False) -> Dict[str, Union[float, int, bool]]:
        """
        Apply ramp-up curve to Enhanced_Defense for players returning from injury.

        Defensive ramp-up curve (shorter than participation):
        - Days 0-14 since return: 70% of Enhanced_Defense
        - Days 15-30: 85%
        - Days 31-60: 95%
        - Days 61+: 100% (full recovery)

        Args:
            enhanced_defense: Raw Enhanced_Defense value from enhanced_features.py
            player_id: Player's MLBAMID
            current_date: Date for projection calculation (default: now)
            lookback_days: How far back to check for injuries (default 60)
            exclude_pitching_injuries: If True, ignore pitching-related injuries
                                      (for two-way players' hitting projections)

        Returns:
            Dictionary containing:
                - adjusted_defense: Defense value with injury curve applied
                - recovery_factor: Factor applied (0.70-1.0)
                - days_since_return: Days since most recent return
                - is_recovering: Whether player is in recovery period
                - injury_type: Type of injury (if applicable)
        """
        if current_date is None:
            current_date = datetime.now()

        if self.injury_data is None:
            logger.debug("No injury data loaded - no defense adjustment")
            return {
                'adjusted_defense': enhanced_defense,
                'recovery_factor': 1.0,
                'days_since_return': None,
                'is_recovering': False,
                'injury_type': None
            }

        # Convert player_id to consistent type
        try:
            player_id = int(player_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid player_id for defense adjustment: {player_id}")
            return {
                'adjusted_defense': enhanced_defense,
                'recovery_factor': 1.0,
                'days_since_return': None,
                'is_recovering': False,
                'injury_type': None
            }

        # Find player's recent injuries
        player_injuries = self.injury_data[
            self.injury_data['MLBAMID'] == player_id
        ].copy()

        # Filter out pitching injuries if requested (for two-way players)
        if exclude_pitching_injuries and len(player_injuries) > 0:
            # Pitching-specific injury categories (aligned with future_season_modules)
            PITCHING_INJURY_CATEGORIES = [
                'tommy_john',
                'shoulder_structural',
                'elbow_injury',
                'shoulder_surgery',
                'elbow_internal_brace'
            ]

            player_injuries = player_injuries[
                ~player_injuries['injury_type'].isin(PITCHING_INJURY_CATEGORIES)
            ].copy()

            logger.debug(
                f"Filtered pitching injuries for player {player_id}, "
                f"{len(player_injuries)} non-pitching injuries remain"
            )

        if len(player_injuries) == 0:
            return {
                'adjusted_defense': enhanced_defense,
                'recovery_factor': 1.0,
                'days_since_return': None,
                'is_recovering': False,
                'injury_type': None
            }

        # Find most recent return within lookback window
        recent_returns = []
        for _, injury in player_injuries.iterrows():
            return_date = injury.get('return_date')
            if pd.isna(return_date):
                continue

            return_date = pd.to_datetime(return_date)
            days_since = (current_date - return_date).days

            if 0 <= days_since <= lookback_days:
                recent_returns.append({
                    'injury_type': injury.get('injury_type', 'unknown'),
                    'return_date': return_date,
                    'days_since': days_since
                })

        # If no recent returns, player is fully recovered
        if not recent_returns:
            return {
                'adjusted_defense': enhanced_defense,
                'recovery_factor': 1.0,
                'days_since_return': None,
                'is_recovering': False,
                'injury_type': None
            }

        # Use most recent return
        most_recent = min(recent_returns, key=lambda x: x['days_since'])
        days_since_return = most_recent['days_since']
        injury_type = most_recent['injury_type']

        # Apply defensive ramp-up curve
        if days_since_return <= 14:
            recovery_factor = 0.70
        elif days_since_return <= 30:
            recovery_factor = 0.85
        elif days_since_return <= 60:
            recovery_factor = 0.95
        else:
            recovery_factor = 1.0

        adjusted_defense = enhanced_defense * recovery_factor

        # Only mark as recovering if within active recovery window (not fully recovered)
        # After 60 days, recovery_factor is 1.0 (fully recovered), so not recovering
        is_recovering = days_since_return < 60

        logger.debug(
            f"Defense injury curve: {days_since_return} days since return, "
            f"factor {recovery_factor:.0%}, defense {enhanced_defense:.1f} → {adjusted_defense:.1f}, "
            f"recovering={is_recovering}"
        )

        return {
            'adjusted_defense': adjusted_defense,
            'recovery_factor': recovery_factor,
            'days_since_return': days_since_return,
            'is_recovering': is_recovering,
            'injury_type': injury_type
        }

    def calculate_injury_recovery_factor(self,
                                         player_id: Union[int, str],
                                         current_date: datetime = None,
                                         enhanced_defense: Optional[float] = None,
                                         exclude_pitching_injuries: bool = False) -> Dict[str, Union[float, str, List, Dict]]:
        """
        Calculate injury recovery adjustment factor for a player.

        Now includes optional defense adjustment integration.

        Args:
            player_id: Player's MLBAMID
            current_date: Current date for calculations (default: now)
            enhanced_defense: If provided, also calculate defense adjustment
            exclude_pitching_injuries: If True, ignore pitching injuries (for hitters)

        Returns:
            Dictionary with recovery factor, details, and optional defense adjustment
        """
        if current_date is None:
            current_date = datetime.now()

        if self.injury_data is None:
            result = {
                'recovery_factor': 1.0,
                'active_injuries': [],
                'reasoning': 'No injury data loaded'
            }
            if enhanced_defense is not None:
                result['defense_adjustment'] = {
                    'adjusted_defense': enhanced_defense,
                    'recovery_factor': 1.0,
                    'days_since_return': None,
                    'is_recovering': False,
                    'injury_type': None
                }
            return result

        # Convert player_id to consistent type
        try:
            player_id = int(player_id)
        except (ValueError, TypeError):
            result = {
                'recovery_factor': 1.0,
                'active_injuries': [],
                'reasoning': 'Invalid player ID'
            }
            if enhanced_defense is not None:
                result['defense_adjustment'] = {
                    'adjusted_defense': enhanced_defense,
                    'recovery_factor': 1.0,
                    'days_since_return': None,
                    'is_recovering': False,
                    'injury_type': None
                }
            return result

        # Find player's recent injuries
        player_injuries = self.injury_data[
            self.injury_data['MLBAMID'] == player_id
        ].copy()

        if len(player_injuries) == 0:
            result = {
                'recovery_factor': 1.0,
                'active_injuries': [],
                'reasoning': 'No injuries found'
            }
            if enhanced_defense is not None:
                result['defense_adjustment'] = {
                    'adjusted_defense': enhanced_defense,
                    'recovery_factor': 1.0,
                    'days_since_return': None,
                    'is_recovering': False,
                    'injury_type': None
                }
            return result

        active_adjustments = []
        total_adjustment = 1.0

        for _, injury in player_injuries.iterrows():
            injury_type = injury.get('injury_type', 'unknown')
            return_date = injury.get('return_date')

            # Skip if no return date or not yet returned
            if pd.isna(return_date):
                continue

            return_date = pd.to_datetime(return_date)
            days_since_return = (current_date - return_date).days

            # Get recovery coefficient
            recovery_config = self.recovery_coefficients.get(
                injury_type, self.recovery_coefficients['unknown'])
            recovery_duration = recovery_config['recovery_duration_days']

            # Apply recovery factor if within recovery period
            if 0 <= days_since_return <= recovery_duration:
                recovery_factor = recovery_config['recovery_factor']

                # Linear recovery: starts at recovery_factor, improves to 1.0
                recovery_progress = days_since_return / recovery_duration
                adjusted_factor = recovery_factor + (1.0 - recovery_factor) * recovery_progress

                total_adjustment *= adjusted_factor

                active_adjustments.append({
                    'injury_type': injury_type,
                    'return_date': return_date.strftime('%Y-%m-%d'),
                    'days_since_return': days_since_return,
                    'recovery_factor': adjusted_factor,
                    'description': recovery_config['description']
                })

        # Apply floor to prevent excessive penalties
        total_adjustment = max(0.70, total_adjustment)

        reasoning = f"Applied {len(active_adjustments)} injury adjustments"
        if active_adjustments:
            injury_types = [adj['injury_type'] for adj in active_adjustments]
            reasoning += f" ({', '.join(injury_types)})"

        result = {
            'recovery_factor': total_adjustment,
            'active_injuries': active_adjustments,
            'reasoning': reasoning
        }

        # Add defense adjustment if requested
        if enhanced_defense is not None:
            defense_adj = self.apply_injury_curve_to_defense(
                enhanced_defense,
                player_id,
                current_date,
                exclude_pitching_injuries=exclude_pitching_injuries
            )
            result['defense_adjustment'] = defense_adj

        return result


def get_injury_recovery_factor(player_id: Union[int, str],
                               injury_calculator: InjuryRecoveryCalculator = None) -> float:
    """
    Convenience function to get injury recovery factor for a player.

    Args:
        player_id: Player's MLBAMID
        injury_calculator: Injury calculator instance (optional)

    Returns:
        Recovery factor (1.0 = no adjustment, <1.0 = recovery penalty)
    """
    if injury_calculator is None:
        injury_calculator = InjuryRecoveryCalculator()
        injury_calculator.load_current_season_injury_data()

    result = injury_calculator.calculate_injury_recovery_factor(player_id)
    return result['recovery_factor']


# Example usage and testing
if __name__ == "__main__":
    # Test injury recovery calculator
    calculator = InjuryRecoveryCalculator()
    injury_data = calculator.load_current_season_injury_data(2025)

    if not injury_data.empty:
        logger.info(f"\nInjury data loaded successfully:")
        logger.info(f"Total records: {len(injury_data)}")
        logger.info(f"Injury types found: {injury_data['injury_type'].value_counts().to_dict()}")

        # Test with a sample player (if any exist)
        sample_players = injury_data['MLBAMID'].unique()[:3]
        for player_id in sample_players:
            result = calculator.calculate_injury_recovery_factor(player_id)
            player_name = injury_data[injury_data['MLBAMID'] == player_id]['Name'].iloc[0]
            logger.info(f"\n{player_name} (ID: {player_id}):")
            logger.info(f"  Recovery factor: {result['recovery_factor']:.3f}")
            logger.info(f"  Reasoning: {result['reasoning']}")
    else:
        logger.warning("No injury data found for testing")
