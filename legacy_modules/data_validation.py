"""
Data Validation Module

This module provides comprehensive data validation and cleaning utilities
for baseball analytics data processing. Handles missing values, outliers,
and data quality issues.
Extracted from cleanedDataParser.py for better modularity.
"""

import numpy as np
import pandas as pd
import json
import os

# Path configuration
CACHE_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\cache"

def validate_and_clean_data_enhanced(X, y):
    """
    Enhanced data validation and cleaning for neural networks
    Handles missing values, outliers, and data type issues

    Args:
        X: Feature matrix (list of lists or numpy array)
        y: Target values (list or numpy array)

    Returns:
        tuple: (cleaned_X, cleaned_y)
    """
    # Convert to numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    print(f"Input data shape: X={X.shape}, y={y.shape}")

    # Check for and handle missing values
    X_missing = np.isnan(X).any(axis=1)
    y_missing = np.isnan(y)
    missing_mask = X_missing | y_missing

    if missing_mask.any():
        print(f"Removing {missing_mask.sum()} rows with missing values")
        X = X[~missing_mask]
        y = y[~missing_mask]

    # Check for infinite values
    X_inf = np.isinf(X).any(axis=1)
    y_inf = np.isinf(y)
    inf_mask = X_inf | y_inf

    if inf_mask.any():
        print(f"Removing {inf_mask.sum()} rows with infinite values")
        X = X[~inf_mask]
        y = y[~inf_mask]

    # Remove extreme outliers (beyond 5 standard deviations)
    outlier_mask = np.zeros(len(y), dtype=bool)

    # Check y for outliers
    y_mean, y_std = np.mean(y), np.std(y)
    if y_std > 0:
        y_outliers = np.abs(y - y_mean) > 5 * y_std
        outlier_mask |= y_outliers

    # Check each feature for outliers
    for col in range(X.shape[1]):
        col_mean, col_std = np.mean(X[:, col]), np.std(X[:, col])
        if col_std > 0:
            col_outliers = np.abs(X[:, col] - col_mean) > 5 * col_std
            outlier_mask |= col_outliers

    if outlier_mask.any():
        print(f"Removing {outlier_mask.sum()} outlier rows")
        X = X[~outlier_mask]
        y = y[~outlier_mask]

    # Ensure minimum data size
    if len(X) < 10:
        raise ValueError(f"Insufficient data after cleaning: {len(X)} samples")

    print(f"Final cleaned data shape: X={X.shape}, y={y.shape}")

    # Convert back to lists for consistency with original interface
    X_clean = X.tolist()
    y_clean = y.tolist()

    return X_clean, y_clean

def validate_player_data(player_data, required_fields=None):
    """
    Validate player data for completeness and correctness

    Args:
        player_data: Dictionary or DataFrame with player information
        required_fields: List of required field names

    Returns:
        dict: Validation results with errors and warnings
    """
    if required_fields is None:
        required_fields = ['Name']

    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'missing_fields': [],
        'invalid_values': []
    }

    # Check if data is provided
    if player_data is None or len(player_data) == 0:
        validation_results['valid'] = False
        validation_results['errors'].append("No player data provided")
        return validation_results

    # Convert to DataFrame if it's a dictionary
    if isinstance(player_data, dict):
        if 'Name' not in player_data:
            validation_results['valid'] = False
            validation_results['errors'].append("Player data must contain 'Name' field")
            return validation_results
        df = pd.DataFrame([player_data])
    else:
        df = player_data

    # Check required fields
    for field in required_fields:
        if field not in df.columns:
            validation_results['valid'] = False
            validation_results['missing_fields'].append(field)

    # Check for missing names
    if 'Name' in df.columns:
        missing_names = df['Name'].isna().sum()
        if missing_names > 0:
            validation_results['warnings'].append(f"{missing_names} players with missing names")

    # Check for duplicate names
    if 'Name' in df.columns:
        duplicates = df['Name'].duplicated().sum()
        if duplicates > 0:
            validation_results['warnings'].append(f"{duplicates} duplicate player names found")

    # Check numeric fields for reasonable ranges
    numeric_fields = {
        'WAR': (-10, 15),        # WAR typically ranges from -5 to 12
        'WARP': (-10, 15),       # Similar to WAR
        'AVG': (0, 1),           # Batting average 0-1
        'OBP': (0, 1),           # On-base percentage 0-1
        'SLG': (0, 4),           # Slugging percentage typically 0-1, max around 2
        'ERA': (0, 20),          # ERA typically 0-10, extreme cases up to 20
        'IP': (0, 300),          # Innings pitched max around 250-300
        'K': (0, 500),           # Strikeouts reasonable max
        'BB': (0, 300)           # Walks reasonable max
    }

    for field, (min_val, max_val) in numeric_fields.items():
        if field in df.columns:
            out_of_range = ((df[field] < min_val) | (df[field] > max_val)) & df[field].notna()
            if out_of_range.any():
                count = out_of_range.sum()
                validation_results['warnings'].append(f"{count} players with {field} outside normal range ({min_val}-{max_val})")

    return validation_results

def clean_data_types(df, type_mapping=None):
    """
    Clean and standardize data types in a DataFrame

    Args:
        df: Input DataFrame
        type_mapping: Dictionary mapping column names to desired types

    Returns:
        DataFrame: Cleaned DataFrame with proper types
    """
    if type_mapping is None:
        # Default type mapping for common baseball stats
        type_mapping = {
            'WAR': 'float64',
            'WARP': 'float64',
            'AVG': 'float64',
            'OBP': 'float64',
            'SLG': 'float64',
            'ERA': 'float64',
            'IP': 'float64',
            'K': 'int64',
            'BB': 'int64',
            'HR': 'int64',
            'Year': 'int64',
            'Name': 'string',
            'Team': 'string'
        }

    cleaned_df = df.copy()

    for column, target_type in type_mapping.items():
        if column in cleaned_df.columns:
            try:
                if target_type in ['int64', 'int32']:
                    # Handle integer conversion with NaN values
                    cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
                    cleaned_df[column] = cleaned_df[column].round().astype('Int64')  # Nullable integer
                elif target_type in ['float64', 'float32']:
                    cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
                elif target_type == 'string':
                    cleaned_df[column] = cleaned_df[column].astype('string')
                else:
                    cleaned_df[column] = cleaned_df[column].astype(target_type)
            except Exception as e:
                print(f"Warning: Could not convert {column} to {target_type}: {e}")

    return cleaned_df

def detect_outliers(data, method='zscore', threshold=3):
    """
    Detect outliers in numerical data

    Args:
        data: Array-like numerical data
        method: 'zscore' or 'iqr'
        threshold: Threshold for outlier detection

    Returns:
        array: Boolean mask indicating outliers
    """
    data = np.array(data)

    if method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold
    elif method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data < lower_bound) | (data > upper_bound)
    else:
        raise ValueError("Method must be 'zscore' or 'iqr'")

def generate_data_quality_report(df, output_file=None):
    """
    Generate a comprehensive data quality report

    Args:
        df: Input DataFrame
        output_file: Optional file path to save the report

    Returns:
        dict: Data quality report
    """
    report = {
        'summary': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum()
        },
        'missing_data': {},
        'data_types': {},
        'outliers': {},
        'duplicates': {}
    }

    # Missing data analysis
    for column in df.columns:
        missing_count = df[column].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        report['missing_data'][column] = {
            'count': int(missing_count),
            'percentage': round(missing_pct, 2)
        }

    # Data types
    for column in df.columns:
        report['data_types'][column] = str(df[column].dtype)

    # Outliers for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if df[column].notna().sum() > 0:
            outliers = detect_outliers(df[column].dropna())
            report['outliers'][column] = int(outliers.sum())

    # Duplicates
    if 'Name' in df.columns:
        duplicate_names = df['Name'].duplicated().sum()
        report['duplicates']['names'] = int(duplicate_names)

    # Save report if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Data quality report saved to {output_file}")
        except Exception as e:
            print(f"Could not save report: {e}")

    return report

def list_cached_mappings():
    """List all cached mappings with metadata"""
    try:
        import glob
        cache_files = glob.glob(os.path.join(CACHE_DIR, "name_mapping_*.json"))

        print(f"\nFound {len(cache_files)} cached mappings:")
        for filepath in cache_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                metadata = cache_data.get('metadata', {})
                filename = os.path.basename(filepath)
                print(f"   {filename}")
                print(f"     Created: {metadata.get('created_timestamp', 'Unknown')}")
                print(f"     Mappings: {metadata.get('mapping_count', 0)}")
                print(f"     Size: {metadata.get('source_count', 0)} -> {metadata.get('target_count', 0)}")
            except:
                print(f"   {os.path.basename(filepath)} (corrupted)")
        print()
    except Exception as e:
        print(f"Warning: Could not list cache: {e}")