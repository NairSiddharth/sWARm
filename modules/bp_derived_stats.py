#!/usr/bin/env python3
"""
Fix for Baseball Prospectus derived statistics calculation

This module adds the missing K% and BB% calculations for pre-2020 BP data
where these statistics need to be calculated from raw values.

For pre-2020 data:
- K% = SO (or K) / PA
- BB% = BB / PA

For post-2020 data:
- K% and BB% are directly provided in the CSV files
"""

import pandas as pd
import numpy as np
import glob
import os

# Constants
PRE_2020_CUTOFF = 2020  # Year when BP started providing K% and BB% directly
STANDARD_FILE_FILTER = 'standard'  # Filter to exclude standard files

def fix_bp_derived_statistics(df, year):
    """
    Add missing derived statistics for BP data

    Args:
        df: DataFrame with BP data for a given year
        year: The year of the data

    Returns:
        DataFrame with K% and BB% calculated if missing
    """
    df_fixed = df.copy()

    # For pre-2020 data, calculate K% and BB% if missing
    if year < PRE_2020_CUTOFF:
        print(f"   Calculating derived statistics for {year} data...")

        # Calculate K% (strikeouts / plate appearances)
        if 'K%' not in df_fixed.columns and 'SO' in df_fixed.columns and 'PA' in df_fixed.columns:
            # Handle potential division by zero
            df_fixed['K%'] = np.where(
                df_fixed['PA'] > 0,
                (df_fixed['SO'] / df_fixed['PA']) * 100,
                0.0
            )
            print(f"      SUCCESS: Calculated K% from SO/PA")
        elif 'K%' not in df_fixed.columns and 'K' in df_fixed.columns and 'PA' in df_fixed.columns:
            # Some files might use 'K' instead of 'SO'
            df_fixed['K%'] = np.where(
                df_fixed['PA'] > 0,
                (df_fixed['K'] / df_fixed['PA']) * 100,
                0.0
            )
            print(f"      SUCCESS: Calculated K% from K/PA")

        # Calculate BB% (walks / plate appearances)
        if 'BB%' not in df_fixed.columns and 'BB' in df_fixed.columns and 'PA' in df_fixed.columns:
            df_fixed['BB%'] = np.where(
                df_fixed['PA'] > 0,
                (df_fixed['BB'] / df_fixed['PA']) * 100,
                0.0
            )
            print(f"      SUCCESS: Calculated BB% from BB/PA")

        # Report on calculations
        if 'K%' in df_fixed.columns:
            valid_k_pct = df_fixed['K%'].notna().sum()
            print(f"      DATA: K%: {valid_k_pct}/{len(df_fixed)} records have valid values")

        if 'BB%' in df_fixed.columns:
            valid_bb_pct = df_fixed['BB%'].notna().sum()
            print(f"      DATA: BB%: {valid_bb_pct}/{len(df_fixed)} records have valid values")
    else:
        print(f"   OK: {year} data already has K% and BB% - no calculation needed")

    return df_fixed

def load_fixed_bp_data(data_dir=None):
    """
    Load BP data with properly calculated derived statistics

    Args:
        data_dir: Optional path to data directory. If None, uses default location.

    Returns:
        Tuple of (hitter_data, pitcher_data) with fixed K% and BB%
    """
    print("LOADING BP DATA WITH FIXED DERIVED STATISTICS")
    print("=" * 60)

    if data_dir is None:
        data_dir = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

    # Load hitter data
    print("\nProcessing BP Hitter Data:")
    hitter_files = glob.glob(os.path.join(data_dir, "BP_Data", "hitters", "bp_hitters_*.csv"))
    hitter_files = [f for f in hitter_files if STANDARD_FILE_FILTER not in f]  # Exclude standard files

    all_hitter_data = []
    for file in sorted(hitter_files):
        year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        try:
            df = pd.read_csv(file, encoding='utf-8-sig')

            if 'WARP' in df.columns or 'BWARP' in df.columns:
                # Standardize WARP column name
                if 'BWARP' in df.columns and 'WARP' not in df.columns:
                    df = df.rename(columns={'BWARP': 'WARP'})

                # Standardize Name column (pre-2020 uses 'NAME', post-2020 uses 'Name')
                if 'NAME' in df.columns and 'Name' not in df.columns:
                    df = df.rename(columns={'NAME': 'Name'})

                # Add year and season info
                df['Season'] = year
                df['Year'] = year

                # Fix derived statistics
                df_fixed = fix_bp_derived_statistics(df, year)
                all_hitter_data.append(df_fixed)

                print(f"   SUCCESS {year}: {len(df_fixed)} records loaded")
            else:
                print(f"   WARNING {year}: No WARP column found, skipping")

        except Exception as e:
            print(f"   ERROR {year}: Error loading - {e}")

    # Load pitcher data
    print("\nProcessing BP Pitcher Data:")
    pitcher_files = glob.glob(os.path.join(data_dir, "BP_Data", "pitchers", "bp_pitchers_*.csv"))
    pitcher_files = [f for f in pitcher_files if STANDARD_FILE_FILTER not in f]  # Exclude standard files

    all_pitcher_data = []
    for file in sorted(pitcher_files):
        year = int(os.path.basename(file).split('_')[-1].replace('.csv', ''))

        try:
            df = pd.read_csv(file, encoding='utf-8-sig')

            if 'WARP' in df.columns or 'PWARP' in df.columns:
                # Standardize WARP column name
                if 'PWARP' in df.columns and 'WARP' not in df.columns:
                    df = df.rename(columns={'PWARP': 'WARP'})

                # Standardize Name column (pre-2020 uses 'NAME', post-2020 uses 'Name')
                if 'NAME' in df.columns and 'Name' not in df.columns:
                    df = df.rename(columns={'NAME': 'Name'})

                # Add year and season info
                df['Season'] = year
                df['Year'] = year

                # Fix derived statistics
                df_fixed = fix_bp_derived_statistics(df, year)
                all_pitcher_data.append(df_fixed)

                print(f"   SUCCESS {year}: {len(df_fixed)} records loaded")
            else:
                print(f"   WARNING {year}: No WARP column found, skipping")

        except Exception as e:
            print(f"   ERROR {year}: Error loading - {e}")

    # Combine all data
    if all_hitter_data:
        combined_hitters = pd.concat(all_hitter_data, ignore_index=True)
        print(f"\nCombined Hitter Data: {len(combined_hitters)} total records")

        # Check K% and BB% coverage
        k_pct_coverage = combined_hitters['K%'].notna().sum() / len(combined_hitters) * 100
        bb_pct_coverage = combined_hitters['BB%'].notna().sum() / len(combined_hitters) * 100
        print(f"   K% coverage: {k_pct_coverage:.1f}%")
        print(f"   BB% coverage: {bb_pct_coverage:.1f}%")
    else:
        combined_hitters = pd.DataFrame()
        print("\nERROR: No hitter data loaded")

    if all_pitcher_data:
        combined_pitchers = pd.concat(all_pitcher_data, ignore_index=True)
        print(f"\nCombined Pitcher Data: {len(combined_pitchers)} total records")

        # Check K% and BB% coverage
        k_pct_coverage = combined_pitchers['K%'].notna().sum() / len(combined_pitchers) * 100
        bb_pct_coverage = combined_pitchers['BB%'].notna().sum() / len(combined_pitchers) * 100
        print(f"   K% coverage: {k_pct_coverage:.1f}%")
        print(f"   BB% coverage: {bb_pct_coverage:.1f}%")
    else:
        combined_pitchers = pd.DataFrame()
        print("\nERROR: No pitcher data loaded")

    print(f"\nSUCCESS: BP DATA LOADING WITH DERIVED STATISTICS COMPLETE!")
    return combined_hitters, combined_pitchers