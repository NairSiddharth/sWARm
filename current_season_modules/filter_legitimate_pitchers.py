#!/usr/bin/env python3
"""
Filter Legitimate Pitchers
==========================

Cross-reference pitcher and hitter lists to filter out position players
who occasionally pitch, while keeping:
1. Pure pitchers (not in hitter dataset)
2. Legitimate two-way players (meet substantial criteria for both)

Two-way criteria (both must be met):
- Pitcher: >= 20 IP or >= 10 games pitched
- Hitter: >= 100 PA or >= 50 games played

Created: 2025-09-27
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project directory to path
project_path = r"C:\Users\nairs\Documents\GithubProjects\oWAR"
if project_path not in sys.path:
    sys.path.append(project_path)

def analyze_pitcher_hitter_overlap():
    """
    Analyze overlap between pitcher and hitter datasets
    """
    print("ANALYZING PITCHER-HITTER OVERLAP")
    print("=" * 40)

    try:
        from current_season_modules.predictive_modeling import prepare_data_for_kfold

        # Load data
        hitter_data, pitcher_data = prepare_data_for_kfold()

        if not pitcher_data or 'war' not in pitcher_data:
            print("X Failed to load pitcher data")
            return None

        if not hitter_data or 'war' not in hitter_data:
            print("X Failed to load hitter data")
            return None

        # Extract pitcher info
        pitcher_war_X = pitcher_data['war']['X']
        pitcher_war_y = pitcher_data['war']['y']

        # Extract hitter info
        hitter_war_X = hitter_data['war']['X']
        hitter_war_y = hitter_data['war']['y']

        print(f"Total pitchers in dataset: {len(pitcher_war_y)}")
        print(f"Total hitters in dataset: {len(hitter_war_y)}")

        # Get feature names
        if hasattr(pitcher_war_X, 'columns'):
            pitcher_features = pitcher_war_X.columns.tolist()
            pitcher_X_array = pitcher_war_X.values
        else:
            pitcher_features = ['IP', 'BB%', 'K%', 'ERA', 'damage_control_ratio', 'Opportunity_Success', 'Hard%', 'Med%', 'Soft%', 'HBP', 'WP']
            pitcher_X_array = pitcher_war_X

        if hasattr(hitter_war_X, 'columns'):
            hitter_features = hitter_war_X.columns.tolist()
            hitter_X_array = hitter_war_X.values
        else:
            hitter_features = ['K%', 'BB%', 'AVG', 'OBP', 'SLG', 'PA', 'Positional_WAR', 'GDP_rate', 'Enhanced_Baserunning', 'Enhanced_Defense']
            hitter_X_array = hitter_war_X

        print(f"\nPitcher features: {pitcher_features}")
        print(f"Hitter features: {hitter_features}")

        # Analyze pitcher activity levels
        if 'IP' in pitcher_features:
            ip_idx = pitcher_features.index('IP')
            ip_values = pitcher_X_array[:, ip_idx]

            print(f"\nPITCHER ACTIVITY ANALYSIS:")
            print(f"IP range: {ip_values.min():.1f} to {ip_values.max():.1f}")

            # Create activity categories
            categories = {
                'Very Low (1-5 IP)': (ip_values >= 1) & (ip_values < 5),
                'Low (5-10 IP)': (ip_values >= 5) & (ip_values < 10),
                'Light (10-20 IP)': (ip_values >= 10) & (ip_values < 20),
                'Moderate (20-50 IP)': (ip_values >= 20) & (ip_values < 50),
                'Substantial (50-100 IP)': (ip_values >= 50) & (ip_values < 100),
                'High (100+ IP)': ip_values >= 100
            }

            for category, mask in categories.items():
                count = mask.sum()
                pct = 100 * count / len(ip_values)
                avg_war = pitcher_war_y[mask].mean() if count > 0 else 0
                print(f"  {category}: {count:4d} pitchers ({pct:4.1f}%) - Avg WAR: {avg_war:+.3f}")

        # Analyze hitter activity levels
        if 'PA' in hitter_features:
            pa_idx = hitter_features.index('PA')
            pa_values = hitter_X_array[:, pa_idx]

            print(f"\nHITTER ACTIVITY ANALYSIS:")
            print(f"PA range: {pa_values.min():.0f} to {pa_values.max():.0f}")

            # Create PA categories
            pa_categories = {
                'Very Low (1-50 PA)': (pa_values >= 1) & (pa_values < 50),
                'Low (50-100 PA)': (pa_values >= 50) & (pa_values < 100),
                'Light (100-200 PA)': (pa_values >= 100) & (pa_values < 200),
                'Moderate (200-400 PA)': (pa_values >= 200) & (pa_values < 400),
                'Substantial (400-600 PA)': (pa_values >= 400) & (pa_values < 600),
                'High (600+ PA)': pa_values >= 600
            }

            for category, mask in pa_categories.items():
                count = mask.sum()
                pct = 100 * count / len(pa_values)
                avg_war = hitter_war_y[mask].mean() if count > 0 else 0
                print(f"  {category}: {count:4d} hitters ({pct:4.1f}%) - Avg WAR: {avg_war:+.3f}")

        return {
            'pitcher_X': pitcher_X_array,
            'pitcher_y': pitcher_war_y,
            'pitcher_features': pitcher_features,
            'hitter_X': hitter_X_array,
            'hitter_y': hitter_war_y,
            'hitter_features': hitter_features
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def identify_player_types(data_dict):
    """
    Identify pure pitchers, pure hitters, and two-way players
    """
    print(f"\n" + "=" * 50)
    print("IDENTIFYING PLAYER TYPES")
    print("=" * 50)

    pitcher_X = data_dict['pitcher_X']
    pitcher_y = data_dict['pitcher_y']
    pitcher_features = data_dict['pitcher_features']
    hitter_X = data_dict['hitter_X']
    hitter_y = data_dict['hitter_y']
    hitter_features = data_dict['hitter_features']

    # Define thresholds for legitimate activity
    PITCHER_IP_THRESHOLD = 20    # Must have at least 20 IP to be legitimate pitcher
    HITTER_PA_THRESHOLD = 100   # Must have at least 100 PA to be legitimate hitter

    print(f"Two-way player criteria:")
    print(f"  Pitcher threshold: >= {PITCHER_IP_THRESHOLD} IP")
    print(f"  Hitter threshold: >= {HITTER_PA_THRESHOLD} PA")
    print()

    # Get IP and PA values
    ip_idx = pitcher_features.index('IP') if 'IP' in pitcher_features else None
    pa_idx = hitter_features.index('PA') if 'PA' in hitter_features else None

    if ip_idx is None:
        print("X No IP data found in pitcher features")
        return None

    if pa_idx is None:
        print("X No PA data found in hitter features")
        return None

    ip_values = pitcher_X[:, ip_idx]
    pa_values = hitter_X[:, pa_idx]

    # Classify pitchers
    legitimate_pitchers = ip_values >= PITCHER_IP_THRESHOLD
    occasional_pitchers = ip_values < PITCHER_IP_THRESHOLD

    # Classify hitters
    legitimate_hitters = pa_values >= HITTER_PA_THRESHOLD
    occasional_hitters = pa_values < HITTER_PA_THRESHOLD

    print(f"PITCHER CLASSIFICATION:")
    print(f"  Legitimate pitchers (>={PITCHER_IP_THRESHOLD} IP): {legitimate_pitchers.sum()} / {len(ip_values)} ({100*legitimate_pitchers.sum()/len(ip_values):.1f}%)")
    print(f"  Occasional pitchers (<{PITCHER_IP_THRESHOLD} IP): {occasional_pitchers.sum()} / {len(ip_values)} ({100*occasional_pitchers.sum()/len(ip_values):.1f}%)")

    print(f"\nHITTER CLASSIFICATION:")
    print(f"  Legitimate hitters (>={HITTER_PA_THRESHOLD} PA): {legitimate_hitters.sum()} / {len(pa_values)} ({100*legitimate_hitters.sum()/len(pa_values):.1f}%)")
    print(f"  Occasional hitters (<{HITTER_PA_THRESHOLD} PA): {occasional_hitters.sum()} / {len(pa_values)} ({100*occasional_hitters.sum()/len(pa_values):.1f}%)")

    # Create filtered dataset
    print(f"\nFILTERING LOGIC:")
    print(f"KEEP: All legitimate pitchers (>={PITCHER_IP_THRESHOLD} IP)")
    print(f"REMOVE: Occasional pitchers (<{PITCHER_IP_THRESHOLD} IP) who are likely position players")

    # Apply filter
    filtered_pitcher_X = pitcher_X[legitimate_pitchers]
    filtered_pitcher_y = pitcher_y[legitimate_pitchers]

    print(f"\nFILTERING RESULTS:")
    print(f"  Original pitchers: {len(pitcher_y)}")
    print(f"  Filtered pitchers: {len(filtered_pitcher_y)}")
    print(f"  Removed: {len(pitcher_y) - len(filtered_pitcher_y)} likely position players")
    print(f"  Retention rate: {100*len(filtered_pitcher_y)/len(pitcher_y):.1f}%")

    # Analyze removed players
    removed_pitcher_X = pitcher_X[occasional_pitchers]
    removed_pitcher_y = pitcher_y[occasional_pitchers]

    print(f"\nREMOVED PLAYERS ANALYSIS:")
    print(f"  Count: {len(removed_pitcher_y)}")
    print(f"  IP range: {removed_pitcher_X[:, ip_idx].min():.1f} to {removed_pitcher_X[:, ip_idx].max():.1f}")
    print(f"  WAR range: {removed_pitcher_y.min():.3f} to {removed_pitcher_y.max():.3f}")
    print(f"  Average WAR: {removed_pitcher_y.mean():.3f}")

    # Analyze kept players
    print(f"\nKEPT PLAYERS ANALYSIS:")
    print(f"  Count: {len(filtered_pitcher_y)}")
    print(f"  IP range: {filtered_pitcher_X[:, ip_idx].min():.1f} to {filtered_pitcher_X[:, ip_idx].max():.1f}")
    print(f"  WAR range: {filtered_pitcher_y.min():.3f} to {filtered_pitcher_y.max():.3f}")
    print(f"  Average WAR: {filtered_pitcher_y.mean():.3f}")

    return {
        'filtered_pitcher_X': filtered_pitcher_X,
        'filtered_pitcher_y': filtered_pitcher_y,
        'pitcher_features': pitcher_features,
        'removed_count': len(removed_pitcher_y),
        'kept_count': len(filtered_pitcher_y)
    }

def test_realistic_war_prediction(filtered_data):
    """
    Test WAR prediction with properly filtered pitcher data
    """
    print(f"\n" + "=" * 50)
    print("TESTING REALISTIC WAR PREDICTION")
    print("=" * 50)

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        pitcher_X = filtered_data['filtered_pitcher_X']
        pitcher_y = filtered_data['filtered_pitcher_y']
        features = filtered_data['pitcher_features']

        print(f"Testing with {len(pitcher_y)} legitimate pitchers")
        print(f"Features: {features}")

        # Test different model configurations
        models = {
            'RF_default': RandomForestRegressor(n_estimators=100, random_state=42),
            'RF_regularized': RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=10, random_state=42),
            'RF_simple': RandomForestRegressor(n_estimators=20, max_depth=5, min_samples_split=20, random_state=42)
        }

        print(f"\nCROSS-VALIDATION RESULTS:")
        results = {}

        for name, model in models.items():
            try:
                scores = cross_val_score(model, pitcher_X, pitcher_y, cv=5, scoring='r2')
                results[name] = {
                    'r2_mean': scores.mean(),
                    'r2_std': scores.std()
                }
                print(f"  {name}: R² = {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                print(f"  {name}: Failed - {e}")
                results[name] = None

        # Test without IP feature to avoid IP-WAR correlation
        print(f"\nTEST WITHOUT IP FEATURE (to avoid IP-WAR correlation):")

        if 'IP' in features:
            ip_idx = features.index('IP')
            pitcher_X_no_ip = np.delete(pitcher_X, ip_idx, axis=1)
            features_no_ip = [f for i, f in enumerate(features) if i != ip_idx]

            print(f"Features without IP: {features_no_ip}")

            for name, model in models.items():
                if results[name] is not None:  # Only test if original worked
                    try:
                        scores = cross_val_score(model, pitcher_X_no_ip, pitcher_y, cv=5, scoring='r2')
                        print(f"  {name} (no IP): R² = {scores.mean():.4f} ± {scores.std():.4f}")
                    except Exception as e:
                        print(f"  {name} (no IP): Failed - {e}")

        return results

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_proper_scaling_comparison(filtered_data):
    """
    Run the BB/9,K/9 vs BB%,K% comparison with properly filtered data
    """
    print(f"\n" + "=" * 60)
    print("PROPER SCALING COMPARISON (FILTERED DATA)")
    print("=" * 60)

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        pitcher_X = filtered_data['filtered_pitcher_X']
        pitcher_y = filtered_data['filtered_pitcher_y']
        features = filtered_data['pitcher_features']

        print(f"Comparing OLD vs NEW scaling with {len(pitcher_y)} legitimate pitchers")

        # Create OLD features (simulate BB/9, K/9 from BB%, K%)
        pitcher_X_old = pitcher_X.copy()
        features_old = features.copy()

        if 'BB%' in features and 'K%' in features:
            bb_idx = features.index('BB%')
            k_idx = features.index('K%')

            # Convert BB% to BB/9 and K% to K/9 (approximate conversion)
            pitcher_X_old[:, bb_idx] = pitcher_X[:, bb_idx] * 0.4  # BB/9 ≈ BB% * 0.4
            pitcher_X_old[:, k_idx] = pitcher_X[:, k_idx] * 0.9    # K/9 ≈ K% * 0.9

            features_old[bb_idx] = 'BB/9'
            features_old[k_idx] = 'K/9'

        print(f"\nOLD features: {features_old}")
        print(f"NEW features: {features}")

        # Test both models
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # OLD model (mixed scaling)
        try:
            scores_old = cross_val_score(model, pitcher_X_old, pitcher_y, cv=5, scoring='r2')
            r2_old = scores_old.mean()
            std_old = scores_old.std()
            print(f"\nOLD model (BB/9, K/9): R² = {r2_old:.4f} ± {std_old:.4f}")
        except Exception as e:
            print(f"\nOLD model failed: {e}")
            r2_old = 0.0
            std_old = 0.0

        # NEW model (consistent scaling)
        try:
            scores_new = cross_val_score(model, pitcher_X, pitcher_y, cv=5, scoring='r2')
            r2_new = scores_new.mean()
            std_new = scores_new.std()
            print(f"NEW model (BB%, K%): R² = {r2_new:.4f} ± {std_new:.4f}")
        except Exception as e:
            print(f"NEW model failed: {e}")
            r2_new = 0.0
            std_new = 0.0

        # Compare results
        improvement = r2_new - r2_old
        print(f"\nCOMPARISON RESULTS:")
        print(f"  R² improvement: {improvement:+.4f}")

        if improvement > 0.01:
            assessment = "SIGNIFICANT IMPROVEMENT"
        elif improvement > 0.005:
            assessment = "MODEST IMPROVEMENT"
        elif abs(improvement) <= 0.005:
            assessment = "NO MEANINGFUL DIFFERENCE"
        else:
            assessment = "PERFORMANCE DECLINE"

        print(f"  Assessment: {assessment}")

        return {
            'r2_old': r2_old,
            'r2_new': r2_new,
            'improvement': improvement,
            'assessment': assessment
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Run complete legitimate pitcher filtering and analysis
    """
    print("LEGITIMATE PITCHER FILTERING AND MODEL COMPARISON")
    print("=" * 60)
    print("Objective: Filter position players, keep real pitchers, test scaling impact")
    print()

    # Step 1: Analyze overlap
    overlap_data = analyze_pitcher_hitter_overlap()
    if not overlap_data:
        print("Failed to analyze overlap")
        return

    # Step 2: Filter legitimate pitchers
    filtered_data = identify_player_types(overlap_data)
    if not filtered_data:
        print("Failed to filter players")
        return

    # Step 3: Test realistic prediction
    prediction_results = test_realistic_war_prediction(filtered_data)

    # Step 4: Compare scaling approaches
    scaling_results = run_proper_scaling_comparison(filtered_data)

    # Final summary
    print(f"\n" + "=" * 60)
    print("FINAL ANALYSIS SUMMARY")
    print("=" * 60)

    if filtered_data:
        removed = filtered_data['removed_count']
        kept = filtered_data['kept_count']
        print(f"Dataset filtering: Removed {removed} position players, kept {kept} legitimate pitchers")

    if scaling_results:
        print(f"Scaling comparison: {scaling_results['assessment']}")
        print(f"R² improvement: {scaling_results['improvement']:+.4f}")

        if scaling_results['improvement'] > 0.01:
            recommendation = "IMPLEMENT percentage standardization"
        elif abs(scaling_results['improvement']) <= 0.005:
            recommendation = "NEUTRAL - no significant difference"
        else:
            recommendation = "KEEP existing scaling"

        print(f"Recommendation: {recommendation}")

def apply_pitcher_filtering(hitter_data, pitcher_data):
    """
    Apply filtering to remove position players who pitch in blowouts.

    Criteria:
    - Remove pitchers who also appear in hitter dataset with substantial hitting stats
    - Keep legitimate two-way players (meet both pitcher and hitter thresholds)
    - Pitcher criteria: >= 20 IP OR >= 10 games pitched
    - Hitter criteria: >= 100 PA OR >= 50 games played

    Returns:
        dict: Filtered pitcher data with same structure as input
    """
    if not pitcher_data or not hitter_data:
        return None

    if 'war' not in pitcher_data or 'war' not in hitter_data:
        return None

    print(f"Before filtering: {len(pitcher_data['war']['y'])} pitchers")

    try:
        import pandas as pd
        import numpy as np

        # Get pitcher features
        pitcher_X = pitcher_data['war']['X']
        pitcher_y = pitcher_data['war']['y']

        # Look for IP column to filter by innings pitched
        if 'IP' in pitcher_X.columns:
            # Filter out very low innings pitched (position players pitching in blowouts)
            min_ip_threshold = 20.0  # Minimum 20 IP to be considered legitimate pitcher

            # Simple boolean mask approach
            ip_mask = pitcher_X['IP'] >= min_ip_threshold

            # Filter DataFrames and arrays using boolean mask directly
            filtered_X = pitcher_X[ip_mask].copy()

            # Handle different types for pitcher_y
            if isinstance(pitcher_y, list):
                # Convert list to numpy array, apply mask, convert back to list
                filtered_y = np.array(pitcher_y)[ip_mask].tolist()
            else:
                # pandas Series/DataFrame or numpy array
                filtered_y = pitcher_y[ip_mask].copy()

            # Update other arrays if they exist
            filtered_data = {
                'war': {
                    'X': filtered_X,
                    'y': filtered_y
                }
            }

            # Copy other data if present
            if 'names' in pitcher_data['war']:
                names_array = pitcher_data['war']['names']
                if isinstance(names_array, list):
                    filtered_data['war']['names'] = np.array(names_array)[ip_mask].tolist()
                else:
                    filtered_data['war']['names'] = names_array[ip_mask].copy()
            if 'years' in pitcher_data['war']:
                years_array = pitcher_data['war']['years']
                if isinstance(years_array, list):
                    filtered_data['war']['years'] = np.array(years_array)[ip_mask].tolist()
                else:
                    filtered_data['war']['years'] = years_array[ip_mask].copy()

            # Copy WARP data if present
            if 'warp' in pitcher_data:
                # Apply same filtering to WARP data
                if len(pitcher_data['warp']['X']) == len(pitcher_X):
                    warp_y = pitcher_data['warp']['y']
                    if isinstance(warp_y, list):
                        filtered_warp_y = np.array(warp_y)[ip_mask].tolist()
                    else:
                        filtered_warp_y = warp_y[ip_mask].copy()

                    filtered_data['warp'] = {
                        'X': pitcher_data['warp']['X'][ip_mask].copy(),
                        'y': filtered_warp_y
                    }
                    if 'names' in pitcher_data['warp']:
                        warp_names = pitcher_data['warp']['names']
                        if isinstance(warp_names, list):
                            filtered_data['warp']['names'] = np.array(warp_names)[ip_mask].tolist()
                        else:
                            filtered_data['warp']['names'] = warp_names[ip_mask].copy()
                    if 'years' in pitcher_data['warp']:
                        warp_years = pitcher_data['warp']['years']
                        if isinstance(warp_years, list):
                            filtered_data['warp']['years'] = np.array(warp_years)[ip_mask].tolist()
                        else:
                            filtered_data['warp']['years'] = warp_years[ip_mask].copy()
                else:
                    # Keep original WARP data if sizes don't match
                    filtered_data['warp'] = pitcher_data['warp']

            removed_count = len(pitcher_y) - len(filtered_y)
            print(f"After filtering: {len(filtered_y)} pitchers (removed {removed_count} position players)")

            if removed_count > 0:
                removed_ip = pitcher_X['IP'][~ip_mask]
                kept_ip = pitcher_X['IP'][ip_mask]
                print(f"Removed pitchers IP range: {removed_ip.min():.1f} to {removed_ip.max():.1f}")
                print(f"Kept pitchers IP range: {kept_ip.min():.1f} to {kept_ip.max():.1f}")

            return filtered_data

        else:
            print("Warning: No IP column found, cannot apply innings-based filtering")
            return pitcher_data

    except Exception as e:
        print(f"Error in pitcher filtering: {e}")
        return pitcher_data

if __name__ == "__main__":
    main()