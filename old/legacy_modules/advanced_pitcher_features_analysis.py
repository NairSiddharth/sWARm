"""
Advanced Pitcher Features Analysis - LOB% and GB% Impact Study

Analyzes the potential performance gains from adding LOB% (left on base percentage)
and GB% (ground ball percentage) to the pitcher feature set.
"""

import pandas as pd
import numpy as np
import sys
import os
from scipy.stats import pearsonr
# import matplotlib.pyplot as plt  # Not available in this environment

# Add project path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_path not in sys.path:
    sys.path.append(project_path)

def analyze_feature_availability():
    """Check availability of LOB% and GB% in our datasets."""

    print("=== FEATURE AVAILABILITY ANALYSIS ===")

    data_dir = os.path.join(project_path, "MLB Player Data", "FanGraphs_Data", "pitchers")

    # Check multiple years for feature availability
    years_to_check = [2022, 2023, 2024, 2025]

    for year in years_to_check:
        file_path = os.path.join(data_dir, f"fangraphs_pitchers_{year}.csv")

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)

                has_lob = 'LOB%' in df.columns
                has_gb = 'GB%' in df.columns

                print(f"\n{year} Dataset:")
                print(f"  Records: {len(df)}")
                print(f"  LOB% available: {has_lob}")
                print(f"  GB% available: {has_gb}")

                if has_lob:
                    lob_valid = df['LOB%'].notna().sum()
                    print(f"  LOB% valid values: {lob_valid}/{len(df)} ({lob_valid/len(df)*100:.1f}%)")

                if has_gb:
                    gb_valid = df['GB%'].notna().sum()
                    print(f"  GB% valid values: {gb_valid}/{len(df)} ({gb_valid/len(df)*100:.1f}%)")

                # Sample values
                if has_lob and has_gb:
                    sample = df[['Name', 'IP', 'ERA', 'WAR', 'LOB%', 'GB%']].dropna().head(5)
                    print(f"  Sample data:")
                    for _, row in sample.iterrows():
                        print(f"    {row['Name']}: LOB%={row['LOB%']*100:.1f}%, GB%={row['GB%']*100:.1f}%, WAR={row['WAR']:.2f}")

            except Exception as e:
                print(f"{year}: Error reading file - {e}")
        else:
            print(f"{year}: File not found")

def correlation_analysis():
    """Analyze correlation between LOB%, GB% and WAR."""

    print(f"\n=== CORRELATION ANALYSIS ===")

    data_dir = os.path.join(project_path, "MLB Player Data", "FanGraphs_Data", "pitchers")

    # Use recent complete season (2024)
    file_path = os.path.join(data_dir, "fangraphs_pitchers_2024.csv")

    if not os.path.exists(file_path):
        print("2024 data not available for correlation analysis")
        return None

    df = pd.read_csv(file_path)

    # Check required columns
    required_cols = ['WAR', 'LOB%', 'GB%', 'IP', 'ERA', 'K/9', 'BB/9', 'HR/9']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return None

    # Filter for qualified pitchers (minimum IP)
    min_ip = 50
    qualified = df[df['IP'] >= min_ip].copy()
    print(f"Qualified pitchers (>= {min_ip} IP): {len(qualified)}")

    # Remove missing values
    analysis_data = qualified[required_cols].dropna()
    print(f"Complete data records: {len(analysis_data)}")

    if len(analysis_data) < 20:
        print("Insufficient data for meaningful analysis")
        return None

    # Calculate correlations
    correlations = {}

    features = ['LOB%', 'GB%', 'ERA', 'K/9', 'BB/9', 'HR/9']

    print(f"\nCorrelations with WAR:")
    for feature in features:
        corr, p_value = pearsonr(analysis_data[feature], analysis_data['WAR'])
        correlations[feature] = {'corr': corr, 'p_value': p_value}
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"  {feature:6}: r = {corr:+.3f}, p = {p_value:.4f} {significance}")

    # Compare with current features
    print(f"\nCurrent features ranked by correlation strength:")
    current_features = ['ERA', 'K/9', 'BB/9', 'HR/9']
    sorted_current = sorted([(feat, abs(correlations[feat]['corr'])) for feat in current_features], key=lambda x: x[1], reverse=True)

    for i, (feat, abs_corr) in enumerate(sorted_current, 1):
        print(f"  {i}. {feat}: |r| = {abs_corr:.3f}")

    # Potential new features
    print(f"\nPotential new features:")
    new_features = ['LOB%', 'GB%']
    for feat in new_features:
        abs_corr = abs(correlations[feat]['corr'])
        print(f"  {feat}: |r| = {abs_corr:.3f}")

        # Compare to weakest current feature
        weakest_current = min(sorted_current, key=lambda x: x[1])
        if abs_corr > weakest_current[1]:
            print(f"    -> STRONGER than {weakest_current[0]} ({weakest_current[1]:.3f})")
        else:
            print(f"    -> Weaker than {weakest_current[0]} ({weakest_current[1]:.3f})")

    return analysis_data, correlations

def feature_interaction_analysis(data, correlations):
    """Analyze how LOB% and GB% interact with existing features."""

    print(f"\n=== FEATURE INTERACTION ANALYSIS ===")

    # Correlation matrix for feature relationships
    features = ['WAR', 'LOB%', 'GB%', 'ERA', 'K/9', 'BB/9', 'HR/9']
    corr_matrix = data[features].corr()

    print("Feature correlation matrix:")
    print("        ", "   ".join(f"{feat:>6}" for feat in features))
    for i, feat1 in enumerate(features):
        row_data = []
        for feat2 in features:
            if feat1 == feat2:
                row_data.append("  1.00")
            else:
                corr_val = corr_matrix.loc[feat1, feat2]
                row_data.append(f"{corr_val:+6.3f}")
        print(f"{feat1:6}: " + " ".join(row_data))

    # Check for multicollinearity concerns
    print(f"\nMulticollinearity analysis:")
    current_features = ['ERA', 'K/9', 'BB/9', 'HR/9']
    new_features = ['LOB%', 'GB%']

    for new_feat in new_features:
        print(f"\n{new_feat} correlations with existing features:")
        for curr_feat in current_features:
            corr_val = corr_matrix.loc[new_feat, curr_feat]
            concern_level = ""
            if abs(corr_val) > 0.7:
                concern_level = " (HIGH MULTICOLLINEARITY RISK)"
            elif abs(corr_val) > 0.5:
                concern_level = " (MODERATE CORRELATION)"

            print(f"  vs {curr_feat}: r = {corr_val:+.3f}{concern_level}")

def skill_vs_luck_analysis(data):
    """Analyze whether LOB% and GB% represent skill or luck."""

    print(f"\n=== SKILL vs LUCK ANALYSIS ===")

    # LOB% analysis (convert to percentages)
    lob_pct = data['LOB%'] * 100
    lob_stats = lob_pct.describe()
    print(f"LOB% distribution:")
    print(f"  Mean: {lob_stats['mean']:.1f}%")
    print(f"  Std:  {lob_stats['std']:.1f}%")
    print(f"  Range: {lob_stats['min']:.1f}% - {lob_stats['max']:.1f}%")

    # League average is typically around 72%
    extreme_lob_low = data[data['LOB%'] < 0.68]
    extreme_lob_high = data[data['LOB%'] > 0.76]

    print(f"  Extreme LOB% (skill indicators):")
    print(f"    Low LOB% (<68%): {len(extreme_lob_low)} pitchers")
    print(f"    High LOB% (>76%): {len(extreme_lob_high)} pitchers")

    # GB% analysis (convert to percentages)
    gb_pct = data['GB%'] * 100
    gb_stats = gb_pct.describe()
    print(f"\nGB% distribution:")
    print(f"  Mean: {gb_stats['mean']:.1f}%")
    print(f"  Std:  {gb_stats['std']:.1f}%")
    print(f"  Range: {gb_stats['min']:.1f}% - {gb_stats['max']:.1f}%")

    # Ground ball specialists
    gb_specialists = data[data['GB%'] > 0.50]
    fly_ball_pitchers = data[data['GB%'] < 0.35]

    print(f"  Pitching style indicators:")
    print(f"    Ground ball specialists (>50%): {len(gb_specialists)} pitchers")
    print(f"    Fly ball pitchers (<35%): {len(fly_ball_pitchers)} pitchers")

def implementation_recommendation():
    """Provide implementation recommendations."""

    print(f"\n=== IMPLEMENTATION RECOMMENDATIONS ===")

    print("Based on the analysis:")
    print()
    print("LOB% (Left on Base Percentage):")
    print("  - Measures pitcher's ability to strand baserunners")
    print("  - Mix of skill and luck, but persistent over seasons")
    print("  - League average ~72%, elite pitchers consistently >75%")
    print("  - Could help identify clutch performance")
    print()
    print("GB% (Ground Ball Percentage):")
    print("  - Measures pitcher's batted ball profile")
    print("  - Strong skill component, relatively stable")
    print("  - Affects defensive context and home run prevention")
    print("  - Could improve HR/FB rate predictions")
    print()
    print("Implementation strategy:")
    print("1. Add both features if correlations are meaningful (|r| > 0.20)")
    print("2. Monitor for multicollinearity with existing features")
    print("3. Test ensemble performance improvement")
    print("4. Consider feature engineering combinations")
    print("5. Validate on historical data before production use")

if __name__ == "__main__":
    print("ADVANCED PITCHER FEATURES ANALYSIS")
    print("=" * 60)

    # Run analysis pipeline
    analyze_feature_availability()

    analysis_data, correlations = correlation_analysis()

    if analysis_data is not None:
        feature_interaction_analysis(analysis_data, correlations)
        skill_vs_luck_analysis(analysis_data)

    implementation_recommendation()