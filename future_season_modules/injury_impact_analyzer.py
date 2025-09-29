"""
Empirical Injury Impact Analyzer
===============================

Analyzes actual pre/post injury performance changes using available
injury data (2020-2024) and performance data (2016-2024) to create
evidence-based injury impact models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
from pathlib import Path

class InjuryImpactAnalyzer:
    """
    Analyzes empirical injury impacts using historical data.

    Creates evidence-based injury impact multipliers rather than
    arbitrary guessed values.
    """

    def __init__(self, injury_data_path: str = "MLB Player Data/FanGraphs_Data/injuries"):
        """
        Initialize with injury data path.

        Args:
            injury_data_path: Path to injury data directory
        """
        self.injury_data_path = Path(injury_data_path)
        self.injury_data = None
        self.performance_data = None
        self.impact_cache = {}

    def load_injury_data(self) -> pd.DataFrame:
        """
        Load and combine all injury data from 2020-2024.

        Returns:
            Combined injury DataFrame
        """
        print("Loading injury data from 2020-2024...")

        injury_files = [
            self.injury_data_path / f"fangraphs_injuryreport_{year}.xlsx"
            for year in range(2020, 2025)
        ]

        all_injuries = []
        for file_path in injury_files:
            if file_path.exists():
                try:
                    df = pd.read_excel(file_path)
                    year = file_path.stem.split('_')[-1]
                    df['injury_year'] = int(year)
                    all_injuries.append(df)
                    print(f"  {year}: {len(df)} injury records")
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")

        if all_injuries:
            combined = pd.concat(all_injuries, ignore_index=True)

            # Clean and standardize data
            combined = self._clean_injury_data(combined)

            self.injury_data = combined
            print(f"Total injury records loaded: {len(combined)}")
            return combined
        else:
            raise ValueError("No injury data files found or loaded successfully")

    def _clean_injury_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize injury data.

        Args:
            df: Raw injury DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Standardize column names
        df = df.rename(columns={
            'Injury / Surgery': 'injury_type',
            'Injury / Surgery Date': 'injury_date',
            'Return Date': 'return_date',
            'Pos': 'position',
            'MLBAMID': 'mlbid'
        })

        # Convert dates
        df['injury_date'] = pd.to_datetime(df['injury_date'], errors='coerce')
        df['return_date'] = pd.to_datetime(df['return_date'], errors='coerce')

        # Calculate recovery days where possible
        df['recovery_days'] = (df['return_date'] - df['injury_date']).dt.days

        # Clean injury types (group similar injuries)
        df['injury_category'] = df['injury_type'].apply(self._categorize_injury)

        # Filter out COVID and undisclosed injuries for impact analysis
        analysis_filter = ~df['injury_type'].isin([
            'COVID-19', 'COVID-19 (protocol)', 'Undisclosed'
        ])

        return df[analysis_filter].copy()

    def _categorize_injury(self, injury_type: str) -> str:
        """
        Categorize injuries into broader groups for analysis.

        Args:
            injury_type: Specific injury description

        Returns:
            Injury category
        """
        if pd.isna(injury_type):
            return 'unknown'

        injury_lower = injury_type.lower()

        # Muscle strains
        if any(muscle in injury_lower for muscle in ['hamstring', 'groin', 'calf', 'quad']):
            return 'leg_muscle_strain'
        elif any(muscle in injury_lower for muscle in ['oblique', 'lat', 'intercostal']):
            return 'torso_muscle_strain'
        elif 'forearm' in injury_lower or 'wrist' in injury_lower:
            return 'arm_muscle_strain'
        elif 'shoulder' in injury_lower and 'strain' in injury_lower:
            return 'shoulder_muscle_strain'
        elif 'back' in injury_lower:
            return 'back_injury'

        # Joint/structural injuries
        elif 'tommy john' in injury_lower:
            return 'tommy_john'
        elif 'shoulder' in injury_lower and ('surgery' in injury_lower or 'impingement' in injury_lower):
            return 'shoulder_structural'
        elif 'elbow' in injury_lower:
            return 'elbow_injury'
        elif 'acl' in injury_lower or 'anterior cruciate' in injury_lower or ('knee' in injury_lower and 'torn' in injury_lower):
            return 'acl_injury'
        elif 'knee' in injury_lower:
            return 'knee_injury'
        elif 'ankle' in injury_lower or 'foot' in injury_lower:
            return 'foot_ankle_injury'

        # Head/other
        elif 'concussion' in injury_lower:
            return 'concussion'
        elif 'hip' in injury_lower:
            return 'hip_injury'
        else:
            return 'other'

    def load_performance_data(self, data_integrator) -> pd.DataFrame:
        """
        Load performance data using existing data integration system.

        Args:
            data_integrator: Instance of DataIntegrator from the pipeline

        Returns:
            Combined performance DataFrame
        """
        print("Loading performance data from 2016-2024...")

        # Use existing system to load data
        years = list(range(2016, 2025))
        performance_data = data_integrator.load_complete_dataset(
            years=years,
            player_types=['hitters', 'pitchers']
        )

        self.performance_data = performance_data
        print(f"Performance data loaded: {len(performance_data)} records")
        return performance_data

    def calculate_injury_impacts(self,
                                injury_category: str,
                                position_filter: str = None,
                                min_sample_size: int = 10) -> Dict:
        """
        Calculate empirical injury impacts for a specific injury category.

        Args:
            injury_category: Injury category to analyze
            position_filter: Optional position filter (P, C, INF, OF)
            min_sample_size: Minimum number of cases required

        Returns:
            Dictionary with impact statistics
        """
        if self.injury_data is None:
            raise ValueError("Injury data not loaded. Call load_injury_data() first.")
        if self.performance_data is None:
            raise ValueError("Performance data not loaded. Call load_performance_data() first.")

        print(f"Analyzing injury impacts for: {injury_category}")
        if position_filter:
            print(f"  Position filter: {position_filter}")

        # Filter injuries
        injury_subset = self.injury_data[
            self.injury_data['injury_category'] == injury_category
        ].copy()

        if position_filter:
            if position_filter == 'P':
                injury_subset = injury_subset[injury_subset['position'].isin(['SP', 'RP'])]
            elif position_filter == 'INF':
                injury_subset = injury_subset[injury_subset['position'].isin(['1B', '2B', '3B', 'SS', 'INF'])]
            elif position_filter == 'OF':
                injury_subset = injury_subset[injury_subset['position'].isin(['OF', 'CF', 'LF', 'RF'])]
            else:
                injury_subset = injury_subset[injury_subset['position'] == position_filter]

        print(f"  Found {len(injury_subset)} {injury_category} cases")

        if len(injury_subset) < min_sample_size:
            print(f"  Insufficient sample size ({len(injury_subset)} < {min_sample_size})")
            return {'insufficient_data': True, 'sample_size': len(injury_subset)}

        # Analyze pre/post performance for each case
        impact_cases = []

        for _, injury_row in injury_subset.iterrows():
            player_id = injury_row['mlbid']
            injury_date = injury_row['injury_date']

            if pd.isna(player_id) or pd.isna(injury_date):
                continue

            # Get player's performance history
            player_perf = self.performance_data[
                self.performance_data['mlbid'] == player_id
            ].copy()

            if len(player_perf) == 0:
                continue

            # Calculate pre-injury performance (1-2 years before)
            pre_injury = player_perf[
                (player_perf['Season'] >= injury_date.year - 2) &
                (player_perf['Season'] < injury_date.year)
            ]

            # Calculate post-injury performance (1-2 years after)
            post_injury = player_perf[
                (player_perf['Season'] > injury_date.year) &
                (player_perf['Season'] <= injury_date.year + 2)
            ]

            if len(pre_injury) > 0 and len(post_injury) > 0:
                case_impact = self._calculate_case_impact(
                    pre_injury, post_injury, injury_row
                )
                if case_impact:
                    impact_cases.append(case_impact)

        print(f"  Analyzed {len(impact_cases)} cases with sufficient data")

        if len(impact_cases) < min_sample_size:
            return {'insufficient_data': True, 'analyzed_cases': len(impact_cases)}

        # Calculate aggregate impact statistics
        return self._aggregate_impact_statistics(impact_cases, injury_category)

    def _calculate_case_impact(self,
                              pre_injury: pd.DataFrame,
                              post_injury: pd.DataFrame,
                              injury_info: pd.Series) -> Optional[Dict]:
        """
        Calculate impact for a single injury case.

        Args:
            pre_injury: Pre-injury performance data
            post_injury: Post-injury performance data
            injury_info: Injury information

        Returns:
            Impact case dictionary or None if insufficient data
        """
        # Calculate mean performance metrics
        pre_metrics = {
            'war': pre_injury['WAR'].mean() if 'WAR' in pre_injury.columns else None,
            'warp': pre_injury['WARP'].mean() if 'WARP' in pre_injury.columns else None,
        }

        post_metrics = {
            'war': post_injury['WAR'].mean() if 'WAR' in post_injury.columns else None,
            'warp': post_injury['WARP'].mean() if 'WARP' in post_injury.columns else None,
        }

        # Calculate ratios (post/pre)
        impact_ratios = {}
        for metric in ['war', 'warp']:
            if (pre_metrics[metric] is not None and
                post_metrics[metric] is not None and
                pre_metrics[metric] != 0):
                impact_ratios[f'{metric}_ratio'] = post_metrics[metric] / pre_metrics[metric]

        if not impact_ratios:
            return None

        return {
            'player_id': injury_info['mlbid'],
            'injury_year': injury_info['injury_year'],
            'position': injury_info['position'],
            'recovery_days': injury_info.get('recovery_days'),
            'pre_metrics': pre_metrics,
            'post_metrics': post_metrics,
            'impact_ratios': impact_ratios
        }

    def _aggregate_impact_statistics(self,
                                   impact_cases: List[Dict],
                                   injury_category: str) -> Dict:
        """
        Aggregate individual case impacts into overall statistics.

        Args:
            impact_cases: List of individual impact cases
            injury_category: Injury category being analyzed

        Returns:
            Aggregated impact statistics
        """
        # Extract impact ratios
        war_ratios = [case['impact_ratios'].get('war_ratio')
                     for case in impact_cases
                     if case['impact_ratios'].get('war_ratio') is not None]

        warp_ratios = [case['impact_ratios'].get('warp_ratio')
                      for case in impact_cases
                      if case['impact_ratios'].get('warp_ratio') is not None]

        # Calculate statistics
        statistics = {
            'injury_category': injury_category,
            'sample_size': len(impact_cases),
            'cases_analyzed': len(impact_cases)
        }

        if war_ratios:
            statistics['war_impact'] = {
                'mean_ratio': np.mean(war_ratios),
                'median_ratio': np.median(war_ratios),
                'std_ratio': np.std(war_ratios),
                'min_ratio': np.min(war_ratios),
                'max_ratio': np.max(war_ratios),
                'sample_size': len(war_ratios)
            }

        if warp_ratios:
            statistics['warp_impact'] = {
                'mean_ratio': np.mean(warp_ratios),
                'median_ratio': np.median(warp_ratios),
                'std_ratio': np.std(warp_ratios),
                'min_ratio': np.min(warp_ratios),
                'max_ratio': np.max(warp_ratios),
                'sample_size': len(warp_ratios)
            }

        # Calculate recovery time statistics
        recovery_times = [case['recovery_days'] for case in impact_cases
                         if case['recovery_days'] is not None and case['recovery_days'] > 0]

        if recovery_times:
            statistics['recovery_time'] = {
                'mean_days': np.mean(recovery_times),
                'median_days': np.median(recovery_times),
                'std_days': np.std(recovery_times),
                'min_days': np.min(recovery_times),
                'max_days': np.max(recovery_times)
            }

        return statistics

    def create_injury_impact_lookup(self,
                                  min_sample_size: int = 10) -> Dict:
        """
        Create comprehensive injury impact lookup table.

        Args:
            min_sample_size: Minimum cases required for reliable statistics

        Returns:
            Dictionary of injury impacts by category and position
        """
        if self.injury_data is None:
            raise ValueError("Must load injury data first")

        print("Creating comprehensive injury impact lookup...")

        # Get all injury categories with sufficient data
        category_counts = self.injury_data['injury_category'].value_counts()
        viable_categories = category_counts[category_counts >= min_sample_size].index.tolist()

        print(f"Analyzing {len(viable_categories)} injury categories with >={min_sample_size} cases")

        impact_lookup = {}

        for category in viable_categories:
            print(f"\\nProcessing {category}...")

            # Overall impact
            overall_impact = self.calculate_injury_impacts(
                category,
                position_filter=None,
                min_sample_size=min_sample_size
            )

            if not overall_impact.get('insufficient_data', False):
                impact_lookup[category] = {
                    'overall': overall_impact
                }

                # Position-specific impacts
                position_impacts = {}
                for pos in ['P', 'C', 'INF', 'OF']:
                    pos_impact = self.calculate_injury_impacts(
                        category,
                        position_filter=pos,
                        min_sample_size=max(3, min_sample_size // 3)  # Lower threshold for position-specific
                    )

                    if not pos_impact.get('insufficient_data', False):
                        position_impacts[pos] = pos_impact

                if position_impacts:
                    impact_lookup[category]['by_position'] = position_impacts

        return impact_lookup

    def create_covid_prorated_war(self,
                                performance_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create prorated WAR values for 2020 COVID-shortened season.

        Args:
            performance_data: Performance data including 2020 season

        Returns:
            DataFrame with additional prorated_war_2020 column
        """
        print("Creating COVID-19 prorated WAR for elite classification...")

        enhanced_data = performance_data.copy()

        # Calculate prorated WAR for 2020 season
        season_2020 = enhanced_data['Season'] == 2020

        if season_2020.any():
            # Assume typical 162 game season vs actual ~60 games played
            prorated_multiplier = 162 / 60  # Approximately 2.7

            enhanced_data.loc[season_2020, 'prorated_WAR_2020'] = (
                enhanced_data.loc[season_2020, 'WAR'] * prorated_multiplier
            )
            enhanced_data.loc[season_2020, 'prorated_WARP_2020'] = (
                enhanced_data.loc[season_2020, 'WARP'] * prorated_multiplier
            )

            print(f"  Prorated {season_2020.sum()} 2020 records")
            print(f"  Multiplier used: {prorated_multiplier:.2f}")

        return enhanced_data