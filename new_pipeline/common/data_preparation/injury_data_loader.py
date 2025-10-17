"""
Injury Data Loader for ROS Projections

Loads and processes FanGraphs injury report Excel files for use in ROS models.
Handles injury classification, date parsing, and season-ending injury detection.
"""

import os
import pandas as pd
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import re


def discover_injury_years(
    data_dir: str = "MLB Player Data/FanGraphs_Data/injuries",
    exclude_current_season: bool = True
) -> List[int]:
    """
    Automatically discover all available injury data years by scanning directory.

    Args:
        data_dir: Path to injury data directory (can be relative or absolute)
        exclude_current_season: If True, exclude current year (for training data)

    Returns:
        Sorted list of available years

    Example:
        >>> years = discover_injury_years()  # Returns [2020, 2021, 2022, 2023, 2024]
        >>> all_years = discover_injury_years(exclude_current_season=False)  # Includes 2025
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Injury data directory not found: {data_path.absolute()}\n"
            f"Expected directory with fangraphs_injuryreport_YYYY.xlsx files"
        )

    # Find all files matching pattern
    pattern = re.compile(r'fangraphs_injuryreport_(\d{4})\.xlsx')
    years = []

    for file in data_path.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                years.append(int(match.group(1)))

    if not years:
        raise FileNotFoundError(
            f"No injury files found in {data_path.absolute()}\n"
            f"Expected files: fangraphs_injuryreport_YYYY.xlsx"
        )

    years.sort()

    # Exclude current season if requested (for training data)
    if exclude_current_season:
        current_year = datetime.now().year
        years = [y for y in years if y < current_year]

    print(f"Discovered injury data years: {years}")

    return years


def load_injury_data(
    year: int,
    data_dir: str = "MLB Player Data/FanGraphs_Data/injuries"
) -> pd.DataFrame:
    """
    Load injury data for a specific year from FanGraphs Excel file.

    Args:
        year: Year to load (e.g., 2024, 2025)
        data_dir: Path to injury data directory

    Returns:
        DataFrame with columns:
        - MLBAMID: Player identifier
        - Name: Player name
        - Position: Player position
        - injury_type: Injury description
        - injury_date: Date of injury/surgery
        - return_date: Date player returned (NaN if not yet returned)
        - eligible_date: Date eligible to return
        - status: IL status (60-Day IL, 15-Day IL, etc.)
        - Year: Year of data

    Example:
        >>> injury_df = load_injury_data(2024)
        >>> injury_df[injury_df['MLBAMID'] == 676917]  # Cade Cavalli
    """
    file_path = Path(data_dir) / f"fangraphs_injuryreport_{year}.xlsx"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Injury file not found: {file_path.absolute()}\n"
            f"Expected file: fangraphs_injuryreport_{year}.xlsx"
        )

    try:
        # Load Excel file
        df = pd.read_excel(file_path)

        # Standardize column names
        df = df.rename(columns={
            'Injury / Surgery': 'injury_type',
            'Injury / Surgery Date': 'injury_date',
            'Return Date': 'return_date',
            'Eligible to Return': 'eligible_date',
            'Pos': 'Position',
            'Status': 'status'
        })

        # Parse dates with mixed format handling
        df['injury_date'] = pd.to_datetime(df['injury_date'], format='mixed', errors='coerce')
        df['return_date'] = pd.to_datetime(df['return_date'], format='mixed', errors='coerce')
        df['eligible_date'] = pd.to_datetime(df['eligible_date'], format='mixed', errors='coerce')

        # Add year column
        df['Year'] = year

        # Keep only relevant columns
        keep_cols = [
            'MLBAMID', 'Name', 'Position', 'injury_type', 'injury_date',
            'return_date', 'eligible_date', 'status', 'Year'
        ]
        df = df[keep_cols].copy()

        print(f"Loaded {len(df)} injury records for {year}")
        print(f"  Status breakdown: {df['status'].value_counts().to_dict()}")

        return df

    except Exception as e:
        raise RuntimeError(
            f"Error loading injury data for {year}: {e}\n"
            f"File: {file_path.absolute()}"
        ) from e


def load_injury_data_multiple_years(
    years: Optional[List[int]] = None,
    data_dir: str = "MLB Player Data/FanGraphs_Data/injuries",
    auto_discover: bool = False,
    exclude_current_season: bool = True
) -> pd.DataFrame:
    """
    Load and combine injury data from multiple years.

    Args:
        years: List of years to load (e.g., [2020, 2021, 2022, 2023, 2024])
               If None and auto_discover=True, will discover available years
        data_dir: Path to injury data directory
        auto_discover: If True and years=None, automatically discover available years
        exclude_current_season: If auto_discover=True, exclude current year (for training)

    Returns:
        Combined DataFrame with all injury records

    Example:
        >>> # Load historical injury data for training (auto-discover)
        >>> historical_injuries = load_injury_data_multiple_years(auto_discover=True)
        >>> # Load specific years
        >>> specific_injuries = load_injury_data_multiple_years([2023, 2024])
    """
    # Auto-discover years if requested and years not provided
    if years is None:
        if auto_discover:
            years = discover_injury_years(data_dir, exclude_current_season)
        else:
            raise ValueError(
                "Must provide either 'years' list or set auto_discover=True"
            )

    if not years:
        raise ValueError("No years to load (list is empty)")

    all_injuries = []
    failed_years = []

    for year in years:
        try:
            year_data = load_injury_data(year, data_dir)
            if not year_data.empty:
                all_injuries.append(year_data)
        except Exception as e:
            failed_years.append((year, str(e)))

    if failed_years:
        error_msg = "Failed to load data for years:\n" + "\n".join(
            f"  {year}: {error}" for year, error in failed_years
        )
        raise RuntimeError(error_msg)

    if not all_injuries:
        raise RuntimeError(
            f"No injury data loaded for any year in {years}\n"
            f"Check that files exist in: {Path(data_dir).absolute()}"
        )

    combined = pd.concat(all_injuries, ignore_index=True)
    print(f"\nCombined injury data: {len(combined)} total records across {len(years)} years")

    return combined


def get_player_injuries(
    injury_df: pd.DataFrame,
    mlbam_id: int,
    year: Optional[int] = None
) -> pd.DataFrame:
    """
    Get injury history for a specific player.

    Args:
        injury_df: Injury DataFrame from load_injury_data()
        mlbam_id: Player's MLBAM ID
        year: Optional year filter

    Returns:
        DataFrame with player's injury records

    Example:
        >>> injuries = load_injury_data(2024)
        >>> player_injuries = get_player_injuries(injuries, 676917)  # Cade Cavalli
    """
    player_data = injury_df[injury_df['MLBAMID'] == mlbam_id].copy()

    if year is not None:
        player_data = player_data[player_data['Year'] == year]

    return player_data.sort_values('injury_date')


def classify_injury_severity(injury_description: str) -> str:
    """
    Classify injury into severity categories for feature engineering.

    Categories align with injury_recovery.py INJURY_RECOVERY_COEFFICIENTS.

    Args:
        injury_description: Injury description from FanGraphs

    Returns:
        Injury category: tommy_john, shoulder_surgery, oblique_strain, etc.
    """
    if pd.isna(injury_description):
        return 'unknown'

    desc_lower = str(injury_description).lower()

    # Major surgeries (severity 3)
    if 'tommy john' in desc_lower or 'ucl' in desc_lower:
        return 'tommy_john'
    elif 'shoulder surgery' in desc_lower or 'labrum' in desc_lower:
        return 'shoulder_surgery'
    elif 'hip surgery' in desc_lower:
        return 'hip_surgery'
    elif 'elbow' in desc_lower and ('brace' in desc_lower or 'internal' in desc_lower):
        return 'elbow_internal_brace'
    elif 'surgery' in desc_lower:
        return 'other_surgery'

    # Strains (severity 1)
    elif 'oblique' in desc_lower:
        return 'oblique_strain'
    elif 'hamstring' in desc_lower:
        return 'hamstring_strain'
    elif 'shoulder' in desc_lower and 'strain' in desc_lower:
        return 'shoulder_strain'
    elif 'back' in desc_lower:
        return 'back_strain'
    elif 'groin' in desc_lower:
        return 'groin_strain'

    return 'unknown'


# Example usage
if __name__ == "__main__":
    # Test loading current season data
    print("Testing injury data loader...")

    injury_2025 = load_injury_data(2025)

    if not injury_2025.empty:
        print(f"\n2025 Injury Data Summary:")
        print(f"Total records: {len(injury_2025)}")
        print(f"Unique players: {injury_2025['MLBAMID'].nunique()}")
        print(f"\nSample record:")
        print(injury_2025.head(1).to_dict('records'))

        # Test player lookup
        if len(injury_2025) > 0:
            sample_id = injury_2025.iloc[0]['MLBAMID']
            player_inj = get_player_injuries(injury_2025, sample_id)
            print(f"\nSample player injuries: {len(player_inj)} record(s)")
