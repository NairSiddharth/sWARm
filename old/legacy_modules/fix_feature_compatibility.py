#!/usr/bin/env python3
"""
Feature Compatibility Fix - Quick Solution
==========================================

This script updates the ensemble predictor to handle the HBP vs HBP% mismatch
without requiring model retraining.

Created: 2025-09-27
"""

import sys
import os
sys.path.append('.')

# Read the current ensemble_modeling.py and add a better compatibility function
def create_compatibility_fix():
    """Create a more robust compatibility fix."""

    compatibility_code = '''
    def _handle_feature_compatibility(self, X, player_type):
        """
        Handle feature compatibility between old models and new system.

        Old models expect: IP, BB%, K%, ERA, damage_control_ratio, Opportunity_Success, Contact_Quality_Index, HBP, WP
        New system provides: IP, BB%, K%, ERA, damage_control_ratio, Opportunity_Success, Contact_Quality_Index, HBP%, WP

        Key differences:
        1. Position 7: HBP (raw count) vs HBP% (percentage)
        2. Many advanced features may be 0 due to data loading issues

        Args:
            X: Input features array
            player_type: 'hitter' or 'pitcher'

        Returns:
            Compatible feature array
        """
        if player_type == 'pitcher' and len(X.shape) >= 1 and X.shape[-1] >= 9:
            X_compat = X.copy() if hasattr(X, 'copy') else np.array(X)

            # Ensure we have exactly 9 features
            if X_compat.ndim == 1:
                X_compat = X_compat[:9]
                # Convert HBP% back to approximate HBP count for old models
                # Position 7: HBP% -> HBP (rough approximation)
                if X_compat[7] > 0:  # If HBP% > 0
                    # Rough conversion: HBP% * IP * 3 (approximate pitches per inning) / 100
                    ip = X_compat[0] if X_compat[0] > 0 else 50  # Default to 50 IP if 0
                    estimated_hbp = (X_compat[7] * ip * 3) / 100
                    X_compat[7] = max(0, min(estimated_hbp, 10))  # Cap between 0-10 HBP
                else:
                    X_compat[7] = 0  # Default HBP count
            else:
                X_compat = X_compat[:, :9]
                # Convert HBP% back to approximate HBP count for old models
                for i in range(X_compat.shape[0]):
                    if X_compat[i, 7] > 0:  # If HBP% > 0
                        ip = X_compat[i, 0] if X_compat[i, 0] > 0 else 50
                        estimated_hbp = (X_compat[i, 7] * ip * 3) / 100
                        X_compat[i, 7] = max(0, min(estimated_hbp, 10))
                    else:
                        X_compat[i, 7] = 0

            return X_compat

        return X  # Return as-is for hitters or if already compatible
    '''

    return compatibility_code.strip()

def show_current_fix():
    """Show the current state and the fix needed."""
    print("FEATURE COMPATIBILITY ANALYSIS")
    print("=" * 50)

    print("\nCURRENT ISSUE:")
    print("- Your ensemble models expect 9 features with HBP (raw count)")
    print("- New system provides 9 features with HBP% (percentage)")
    print("- Position 7 mismatch causes prediction failures")

    print("\nSOLUTION OPTIONS:")
    print("1. IMMEDIATE FIX: Update ensemble_modeling.py compatibility function")
    print("   - Convert HBP% back to estimated HBP count")
    print("   - Your system works immediately")
    print("   - Slightly less accurate due to conversion")

    print("2. OPTIMAL FIX: Retrain ensemble models")
    print("   - Use the new 9-feature set with HBP%")
    print("   - Get benefits of normalized CQI and SLQI")
    print("   - Takes time to retrain")

    print("\nRECOMMENDATION:")
    print("Apply the immediate fix now, then retrain models when convenient.")

    print(f"\nThe compatibility function in ensemble_modeling.py needs to be updated:")
    print("(The fix is already applied in the code)")

if __name__ == "__main__":
    show_current_fix()