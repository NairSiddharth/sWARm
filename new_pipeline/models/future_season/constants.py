"""
Constants for Future Season Projections

Feature sets optimized for year-to-year prediction with high-correlation features.
"""

# Import injury features from common (these stay the same)
from new_pipeline.common.constants import (
    COL_MLBAMID,
    COL_NAME,
    COL_TEAM,
    COL_YEAR,
    COL_POSITION,
    COL_IP,
    COL_PA,
    COL_WAR
)

# ============================================================================
# Future Season Feature Lists
# ============================================================================

# Pitcher base features (NO ERA, LOB%, HR/FB%)
FUTURE_PITCHER_BASE_FEATURES = [
    'BB%',          # Keep (reused from common)
    'K%',           # Keep (reused from common)
    'GB%',          # Keep (reused from common)
    'SwStr%',       # Keep (reused from common)
    'WPA/LI',       # Keep (reused from common)
    'Running_Control',  # Keep (reused from common)
    # NEW FEATURES
    'Contact%',     # New (from stuff file)
    'O-Swing%',     # New (from stuff file)
    'Zone%',        # New (from stuff file)
    'O-Contact%',   # New (from stuff file)
    'F-Strike%'     # New (from stuff file)
]

# Pitcher composite features (EXCLUDE damage_control_ratio, Opportunity_Success)
FUTURE_PITCHER_COMPOSITE_FEATURES = [
    'strikeout_efficiency',           # Keep (uses K%, BB%)
    'contact_management',             # Keep (uses GB%, BB%)
    'strikeout_contact_quality',      # Keep (uses K%, Hard%)
    'Statcast_Launch_Quality_Index',  # Keep (uses Statcast)
    'SD_MD_Net'                       # Keep (uses SD, MD)
]

# Injury features (same as common)
INJURY_FEATURES = [
    'has_injury_data',
    'had_tommy_john_ever',
    'years_since_tommy_john',
    'total_il_days_past_year',
    'had_major_injury_past_year'
]

# Complete pitcher model features
FUTURE_PITCHER_MODEL_FEATURES = (
    FUTURE_PITCHER_BASE_FEATURES +
    FUTURE_PITCHER_COMPOSITE_FEATURES +
    INJURY_FEATURES
)

# Hitter base features (adds ISO, GB%, HR/FB, Hard%, Pull%)
FUTURE_HITTER_BASE_FEATURES = [
    'K%',                    # Keep (reused from common)
    'BB%',                   # Keep (reused from common)
    'AVG',                   # Keep (reused from common)
    'OBP',                   # Keep (reused from common)
    'SLG',                   # Keep (reused from common)
    'GDP',                   # Keep (reused from common)
    'Positional_WAR',        # Keep (reused from common)
    'Enhanced_Baserunning',  # Keep (reused from common)
    'Enhanced_Defense',      # Keep (reused from common)
    # NEW FEATURES
    'ISO',                   # New (from advanced file)
    'GB%',                   # New (from battedball file)
    'HR/FB',                 # New (from battedball file)
    'Hard%',                 # New (from battedball file)
    'Pull%'                  # New (from battedball file)
]

# Complete hitter model features
FUTURE_HITTER_MODEL_FEATURES = (
    FUTURE_HITTER_BASE_FEATURES +
    INJURY_FEATURES
)

# ============================================================================
# Validation Ranges for New Features
# ============================================================================

# Pitcher feature ranges
VALID_RANGE_CONTACT_PCT = (60, 90)      # Contact%
VALID_RANGE_O_SWING_PCT = (20, 40)      # O-Swing%
VALID_RANGE_ZONE_PCT = (35, 55)         # Zone%
VALID_RANGE_O_CONTACT_PCT = (45, 75)    # O-Contact%
VALID_RANGE_F_STRIKE_PCT = (50, 70)     # F-Strike%

# Hitter feature ranges
VALID_RANGE_ISO = (0.050, 0.400)        # ISO (decimal)
VALID_RANGE_HITTER_GB_PCT = (25, 65)    # GB%
VALID_RANGE_HR_FB = (0, 40)             # HR/FB
VALID_RANGE_HITTER_HARD_PCT = (20, 55)  # Hard%
VALID_RANGE_PULL_PCT = (25, 55)         # Pull%
