"""
Injury Recovery Adjustments for Future Projections

Applies position-specific recovery curves for major injuries (Tommy John, ACL).
Adapted from future_season_modules/future_projections.py lines 1345-1768.

See FUTURE_PROJECTIONS_MIGRATION_GUIDE.md Section 5.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class InjuryRecoveryAdjuster:
    """
    Apply recovery adjustments for major injuries to future projections.

    Key injuries covered:
    - Tommy John surgery (pitchers primarily, some position players)
    - ACL surgery (all positions)

    Recovery patterns based on statistical analysis of 2020-2024 injury data.
    """

    def __init__(self):
        """Initialize injury recovery adjuster with evidence-based coefficients."""
        # Tommy John recovery coefficients (from future_projections.py lines 1378-1384)
        # Based on analysis of 244 Tommy John cases (2020-2024)
        self.tommy_john_coefficients = {
            'SP': {'baseline_days': 399.5, 'age_effect': 7.12, 'year1_base': 0.833, 'year1_age': -0.008},
            'RP': {'baseline_days': 383.2, 'age_effect': 9.75, 'year1_base': 0.854, 'year1_age': -0.006},
            'INF': {'baseline_days': 349.4, 'age_effect': 7.78, 'year1_base': 0.891, 'year1_age': -0.009},
            'C': {'baseline_days': 441.0, 'age_effect': 12.07, 'year1_base': 0.785, 'year1_age': -0.012},
            'OF': {'baseline_days': 335.1, 'age_effect': 6.04, 'year1_base': 0.904, 'year1_age': -0.005}
        }

        # ACL surgery recovery coefficients (from future_projections.py lines 1685-1691)
        # Based on analysis: 33 cases, 156 days average recovery
        # Higher impact due to mobility, cutting, and confidence factors
        self.acl_coefficients = {
            'SP': {'year1_base': 0.75, 'year1_age': -0.006, 'year2_base': 0.92, 'year3_base': 0.98},
            'RP': {'year1_base': 0.78, 'year1_age': -0.005, 'year2_base': 0.94, 'year3_base': 0.99},
            'INF': {'year1_base': 0.65, 'year1_age': -0.008, 'year2_base': 0.85, 'year3_base': 0.95},
            'C': {'year1_base': 0.68, 'year1_age': -0.010, 'year2_base': 0.87, 'year3_base': 0.96},
            'OF': {'year1_base': 0.70, 'year1_age': -0.007, 'year2_base': 0.88, 'year3_base': 0.97}
        }

        # Position mapping for coefficient lookup
        self.position_mapping = {
            'SP': 'SP', 'RP': 'RP', 'P': 'SP',
            'C': 'C',
            '1B': 'INF', '2B': 'INF', '3B': 'INF', 'SS': 'INF', 'INF': 'INF',
            'LF': 'OF', 'CF': 'OF', 'RF': 'OF', 'OF': 'OF',
            'DH': 'OF'  # DH uses OF coefficients
        }

    def get_tommy_john_recovery_factors(
        self,
        age: float,
        position: str,
        surgery_year: int,
        projection_year: int
    ) -> float:
        """
        Calculate Tommy John recovery factor for a specific projection year.

        Recovery pattern:
        - Year 1 post-return: 78-90% of baseline (position/age dependent)
        - Year 2 post-return: 85-97% of baseline
        - Year 3+ post-return: 93-99% recovery

        Args:
            age: Player age at time of surgery
            position: Player position
            surgery_year: Year of Tommy John surgery
            projection_year: Year being projected

        Returns:
            Recovery factor (0.0-1.0) to multiply WAR projection
        """
        # Map position to coefficient key
        coeff_key = self.position_mapping.get(position, 'OF')
        coeffs = self.tommy_john_coefficients.get(coeff_key, self.tommy_john_coefficients['OF'])

        # Years post-surgery
        years_post_surgery = projection_year - surgery_year

        if years_post_surgery <= 0:
            # Projection year is before/during surgery year
            return 1.0  # No recovery needed

        # Age adjustment (centered at 28)
        age_adjustment = age - 28

        # Year 1 recovery factor (most impacted)
        year1_factor = coeffs['year1_base'] + (coeffs['year1_age'] * age_adjustment)
        year1_factor = max(0.5, min(1.0, year1_factor))  # Bound between 50-100%

        # Year 2 recovery factor (improved recovery)
        year2_improvement = 0.15  # Average 15% improvement from Year 1 to Year 2
        year2_factor = min(1.0, year1_factor + year2_improvement)

        # Year 3+ recovery factor (near full recovery)
        year3_improvement = 0.08  # Average 8% improvement from Year 2 to Year 3+
        year3_factor = min(1.0, year2_factor + year3_improvement)

        # Return appropriate factor based on years post-surgery
        if years_post_surgery == 1:
            return year1_factor
        elif years_post_surgery == 2:
            return year2_factor
        else:  # 3+ years
            return year3_factor

    def get_acl_recovery_factors(
        self,
        age: float,
        position: str,
        surgery_year: int,
        projection_year: int
    ) -> float:
        """
        Calculate ACL surgery recovery factor for a specific projection year.

        Recovery pattern:
        - Year 1: 65-75% (position/age dependent)
        - Year 2: 85-95%
        - Year 3: 95-100%

        ACL is more severe than general knee injury with longer recovery timeline.
        Higher impact due to mobility, cutting, and confidence factors.

        Args:
            age: Player age at time of surgery
            position: Player position
            surgery_year: Year of ACL surgery
            projection_year: Year being projected

        Returns:
            Recovery factor (0.0-1.0) to multiply WAR projection
        """
        # Map position to coefficient key
        coeff_key = self.position_mapping.get(position, 'OF')
        coeffs = self.acl_coefficients.get(coeff_key, self.acl_coefficients['OF'])

        # Years post-surgery
        years_post_surgery = projection_year - surgery_year

        if years_post_surgery <= 0:
            # Projection year is before/during surgery year
            return 1.0  # No recovery needed

        # Age adjustment (centered at 28)
        age_adjustment = age - 28

        # Year 1 recovery factor (most impacted)
        year1_factor = coeffs['year1_base'] + (coeffs['year1_age'] * age_adjustment)
        year1_factor = max(0.4, min(1.0, year1_factor))  # Bound between 40-100%

        # Year 2 and Year 3 factors (from coefficients)
        year2_factor = coeffs['year2_base']
        year3_factor = coeffs['year3_base']

        # Return appropriate factor based on years post-surgery
        if years_post_surgery == 1:
            return year1_factor
        elif years_post_surgery == 2:
            return year2_factor
        else:  # 3+ years
            return year3_factor

    def apply_injury_adjustments(
        self,
        projections_df: pd.DataFrame,
        injury_records: pd.DataFrame,
        war_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Apply injury recovery adjustments to projections.

        Args:
            projections_df: DataFrame with projections
                Must have: playerid, Age, Position
                And projection columns: war_year_1, war_year_2, war_year_3
            injury_records: DataFrame with injury information
                Must have: playerid, injury_type, surgery_year
                injury_type values: 'tommy_john', 'acl_surgery'
            war_columns: List of WAR columns to adjust (optional)
                Default: ['war_year_1', 'war_year_2', 'war_year_3']

        Returns:
            DataFrame with injury-adjusted projections
        """
        if war_columns is None:
            war_columns = [col for col in ['war_year_1', 'war_year_2', 'war_year_3']
                          if col in projections_df.columns]

        if not war_columns:
            raise ValueError("No WAR projection columns found in projections_df")

        adjusted_df = projections_df.copy()

        # Track adjustments for reporting
        adjustments_made = {'tommy_john': 0, 'acl_surgery': 0}

        # Process each injury record
        for _, injury in injury_records.iterrows():
            playerid = injury.get('playerid')
            injury_type = injury.get('injury_type', '').lower()
            surgery_year = injury.get('surgery_year')

            if pd.isna(playerid) or pd.isna(surgery_year):
                continue

            # Find player in projections
            player_mask = adjusted_df['playerid'] == playerid
            if not player_mask.any():
                continue

            player_row = adjusted_df[player_mask].iloc[0]
            player_age = player_row.get('Age', 28)
            player_position = player_row.get('Position', 'OF')

            # Determine base year for projections (usually current year)
            base_year = injury.get('base_year', surgery_year + 1)

            # Apply appropriate recovery adjustments
            for war_col in war_columns:
                # Extract projection year number
                year_num = int(war_col.split('_')[-1])
                projection_year = base_year + year_num - 1

                # Get recovery factor based on injury type
                if injury_type == 'tommy_john':
                    recovery_factor = self.get_tommy_john_recovery_factors(
                        player_age, player_position, surgery_year, projection_year
                    )
                    adjustments_made['tommy_john'] += 1
                elif injury_type in ['acl_surgery', 'acl']:
                    recovery_factor = self.get_acl_recovery_factors(
                        player_age, player_position, surgery_year, projection_year
                    )
                    adjustments_made['acl_surgery'] += 1
                else:
                    # Unknown injury type, skip
                    continue

                # Apply recovery adjustment
                if war_col in adjusted_df.columns:
                    original_war = adjusted_df.loc[player_mask, war_col].values[0]
                    if not pd.isna(original_war):
                        adjusted_war = original_war * recovery_factor
                        adjusted_df.loc[player_mask, war_col] = adjusted_war

        # Report adjustments
        print("Injury recovery adjustments applied:")
        print(f"  Tommy John: {adjustments_made['tommy_john']} player-years")
        print(f"  ACL Surgery: {adjustments_made['acl_surgery']} player-years")

        return adjusted_df

    def get_expected_recovery_timeline(
        self,
        injury_type: str,
        age: float,
        position: str
    ) -> Dict[str, float]:
        """
        Get expected recovery timeline for an injury.

        Args:
            injury_type: 'tommy_john' or 'acl_surgery'
            age: Player age
            position: Player position

        Returns:
            Dictionary with year-by-year recovery factors
        """
        if injury_type.lower() == 'tommy_john':
            return {
                'year_1': self.get_tommy_john_recovery_factors(age, position, 0, 1),
                'year_2': self.get_tommy_john_recovery_factors(age, position, 0, 2),
                'year_3': self.get_tommy_john_recovery_factors(age, position, 0, 3)
            }
        elif injury_type.lower() in ['acl_surgery', 'acl']:
            return {
                'year_1': self.get_acl_recovery_factors(age, position, 0, 1),
                'year_2': self.get_acl_recovery_factors(age, position, 0, 2),
                'year_3': self.get_acl_recovery_factors(age, position, 0, 3)
            }
        else:
            raise ValueError(f"Unknown injury type: {injury_type}")


def apply_injury_recovery(
    projections_df: pd.DataFrame,
    injury_records: pd.DataFrame = None,
    war_columns: List[str] = None
) -> pd.DataFrame:
    """
    Convenience function to apply injury recovery adjustments.

    Args:
        projections_df: DataFrame with projections
        injury_records: DataFrame with injury information (optional)
        war_columns: List of WAR columns to adjust (optional)

    Returns:
        DataFrame with injury-adjusted projections

    Example:
        >>> projections = joint_model.project_multiple_players(sequences)
        >>> injury_data = pd.read_csv('injury_records.csv')
        >>> adjusted = apply_injury_recovery(projections, injury_data)
    """
    if injury_records is None or len(injury_records) == 0:
        print("No injury records provided. Returning unadjusted projections.")
        return projections_df.copy()

    adjuster = InjuryRecoveryAdjuster()
    return adjuster.apply_injury_adjustments(projections_df, injury_records, war_columns)
