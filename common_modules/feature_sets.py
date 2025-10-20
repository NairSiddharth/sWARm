"""
Feature set definitions for Phase 3.5: Park-Adjusted Multi-Quantile Ensemble.

Defines:
- SKILL_FEATURES: Pure skill metrics (no WPA/LI) for skill model
- CONTEXT_FEATURES: WPA/LI-focused features for context model
- FULL_FEATURES: All features combined for single-model comparison

Phase 3.5 Updates:
- ERA and GB% are park-adjusted (50% home/road blend)
- Added LOB% to provide defensive context for ERA evaluation
"""

# Skill features (NO WPA/LI) - Used for 85% of final prediction
SKILL_FEATURES = [
    'BB%',                           # Walk rate
    'K%',                            # Strikeout rate
    'ERA',                           # Earned run average
    'GB%',                           # Ground ball percentage
    'SwStr%',                        # Swinging strike percentage
    'damage_control_ratio',          # LOB% / (HR/FB% + 0.5)
    'Opportunity_Success',           # (K% - BB%) * (LOB% / 100)
    'Statcast_Launch_Quality_Index', # Exit velo + launch angle composite
    # REMOVED: Interaction features were causing scale issues and harming performance
    # 'strikeout_efficiency',          # K% × (100 - BB%)
    # 'contact_management',            # GB% × (100 - BB%)
    # 'strikeout_contact_quality',     # K% × (100 - Hard%)
]

# Context features (WPA/LI focused) - Used for 15% of final prediction
CONTEXT_FEATURES = [
    'WPA/LI',  # Win probability added per leverage index
    'K%',      # Minimal interaction
    'ERA',     # Minimal interaction
]

# Full feature set (for single-model comparison and backward compatibility)
FULL_FEATURES = SKILL_FEATURES + ['WPA/LI']

# Phase 2.5 features (for comparison - includes CQI removed)
PHASE_2_5_FEATURES = [
    'BB%', 'K%', 'ERA', 'GB%', 'SwStr%',
    'damage_control_ratio',
    'Opportunity_Success',
    'Statcast_Launch_Quality_Index',
    'WPA/LI',
    'strikeout_efficiency',
    'contact_management',
    'strikeout_contact_quality',
]
