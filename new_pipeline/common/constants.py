"""
Constants for the oWAR pipeline.

All magic strings, numbers, and configuration values centralized here
for maintainability and consistency.
"""

# ============================================================================
# Column Names
# ============================================================================

# Player Identification
COL_MLBAMID = 'MLBAMID'
COL_NAME = 'Name'
COL_TEAM = 'Team'
COL_YEAR = 'Year'
COL_POSITION = 'Pos'

# Usage Metrics
COL_IP = 'IP'  # Innings Pitched
COL_PA = 'PA'  # Plate Appearances
COL_GS = 'GS'  # Games Started
COL_WAR = 'WAR'  # Wins Above Replacement
COL_GDP = 'GDP'  # Ground into Double Play

# Rate Stats (Pitchers)
COL_BB_PCT = 'BB%'  # Walk percentage
COL_K_PCT = 'K%'  # Strikeout percentage
COL_SWSTR_PCT = 'SwStr%'  # Swinging strike percentage
COL_WPA_LI = 'WPA/LI'  # Win probability added / Leverage index
COL_SD = 'SD'  # Shutdowns (high-leverage success)
COL_MD = 'MD'  # Meltdowns (high-leverage failure)
COL_LOB_PCT = 'LOB%'  # Left on base percentage
COL_HARD_PCT = 'Hard%'  # Hard contact percentage
COL_ERA = 'ERA'  # Earned run average
COL_GB_PCT = 'GB%'  # Ground ball percentage
COL_HR_FB_PCT = 'HR/FB%'  # Home run to fly ball percentage

# Rate Stats (Hitters)
COL_AVG = 'AVG'  # Batting average
COL_OBP = 'OBP'  # On-base percentage
COL_SLG = 'SLG'  # Slugging percentage

# Composite Features (Pitchers)
COL_AVG_HIT_ANGLE = 'avg_hit_angle'  # Statcast
COL_ANGLE_SWEET_SPOT_PCT = 'anglesweetspotpercent'  # Statcast
COL_RUNNING_CONTROL = 'Running_Control'

# Composite Features (Hitters)
COL_POSITIONAL_WAR = 'Positional_WAR'
COL_ENHANCED_BASERUNNING = 'Enhanced_Baserunning'
COL_ENHANCED_DEFENSE = 'Enhanced_Defense'

# Derived Columns
COL_WAR_PER_162 = 'WAR_per_162'  # Pitcher WAR rate
COL_WAR_PER_600 = 'WAR_per_600'  # Hitter WAR rate
COL_TWO_WAY_PLAYER = 'two_way_player'  # Boolean flag


# ============================================================================
# Filtering Thresholds
# ============================================================================

# Minimum usage thresholds for data quality
MIN_IP_DEFAULT = 20  # Minimum innings pitched for pitchers
MIN_PA_DEFAULT = 75  # Minimum plate appearances for hitters (~12% season, reduces bias)

# Two-way player criteria (MLB designation, rule effective 2020)
TWO_WAY_MIN_IP = 20  # Minimum IP to qualify as pitcher
TWO_WAY_MIN_STARTS = 20  # Minimum starts as position player/DH
TWO_WAY_MIN_PA = 60  # Minimum PA (implies ~3 PA per start)
TWO_WAY_OHTANI_MLBAMID = 660271  # Expected only qualifier


# ============================================================================
# Player Classification Thresholds
# ============================================================================

# Pitcher role classification (based on GS/G ratio)
PITCHER_STARTER_THRESHOLD = 0.7   # GS/G > 0.7 = Starter
PITCHER_RELIEVER_THRESHOLD = 0.1  # GS/G < 0.1 = Reliever
# Between 0.1 and 0.7 = Swing pitcher

# Hitter position classification (based on % time at position)
HITTER_PRIMARY_POSITION_THRESHOLD = 0.90  # >90% time at one position = Primary
HITTER_DUAL_POSITION_THRESHOLD = 0.08     # >8% time = significant playing time
# Exactly 2 positions >8% = Dual position (e.g., "SS/2B")
# 3+ positions >8% OR <2 positions >8% = Utility player


# ============================================================================
# WAR Normalization
# ============================================================================

WAR_NORMALIZATION_IP = 162  # Normalize pitcher WAR per 162 IP (default/starters)
WAR_NORMALIZATION_PA = 600  # Normalize hitter WAR per 600 PA

# Role-specific WAR normalization for pitchers
# Based on FanGraphs qualification thresholds and typical workloads
WAR_NORMALIZATION_IP_STARTER = 162    # Starters: Full season workload
WAR_NORMALIZATION_IP_RELIEVER = 48.2  # Relievers: FanGraphs qualification threshold (2025)
WAR_NORMALIZATION_IP_SWING = 110      # Swing pitchers: Middle ground between roles

# Full season usage for projection calculations (ROS projections)
FULL_SEASON_GAMES = 162  # MLB regular season games
FULL_SEASON_PA = 600     # Standard full-time hitter PA


# ============================================================================
# Rookie/Call-up Regression Thresholds
# ============================================================================

# Qualification rates for determining when to apply regression to ROS projections
# Based on MLB participation standards (IP/game or PA/game)
QUALIFICATION_RATES = {
    'starter': WAR_NORMALIZATION_IP_STARTER / 162,     # 1.0 IP/game
    'reliever': WAR_NORMALIZATION_IP_RELIEVER / 162,   # 0.298 IP/game (48.2/162)
    'swing': WAR_NORMALIZATION_IP_SWING / 162,         # 0.679 IP/game (110/162)
    'hitter': 3.1  # MLB standard (not 600/162 = 3.7)
}

# Minimum usage thresholds to prevent 1-game flukes
# Values chosen via empirical testing (see ROOKIE_ROS_REGRESSION_TESTING_PLAN.md)
MINIMUM_USAGE_THRESHOLDS = {
    'starter': 20,      # IP
    'reliever': 15,     # IP
    'swing': 15,        # IP
    'hitter': 40        # PA
}


# ============================================================================
# Validation Ranges (for data quality checks)
# ============================================================================

# Pitcher feature ranges (expected values)
VALID_RANGE_BB_PCT = (0, 25)  # Walk percentage
VALID_RANGE_K_PCT = (0, 50)  # Strikeout percentage
VALID_RANGE_SWSTR_PCT = (0, 25)  # Swinging strike percentage
VALID_RANGE_ERA = (0, 15)  # Earned run average
VALID_RANGE_GB_PCT = (20, 80)  # Ground ball percentage
VALID_RANGE_LOB_PCT = (50, 100)  # Left on base percentage
VALID_RANGE_HR_FB_PCT = (0, 50)  # Home run to fly ball percentage
VALID_RANGE_HARD_PCT = (0, 60)  # Hard contact percentage
VALID_RANGE_WAR = (-3, 12)  # Wins above replacement

# Hitter feature ranges (expected values)
VALID_RANGE_HITTER_BB_PCT = (0, 30)  # Walk percentage (hitters)
VALID_RANGE_HITTER_K_PCT = (0, 50)  # Strikeout percentage (hitters)
VALID_RANGE_AVG = (0.100, 0.400)  # Batting average
VALID_RANGE_OBP = (0.200, 0.500)  # On-base percentage
VALID_RANGE_SLG = (0.200, 0.800)  # Slugging percentage


# ============================================================================
# Imputation
# ============================================================================

# Percentile to use for replacement level imputation
REPLACEMENT_LEVEL_PERCENTILE = 0.25  # 25th percentile = replacement level


# ============================================================================
# Data Directories
# ============================================================================

# Project root resolution (works from any subdirectory)
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).parent.parent.parent

# FanGraphs data directories
FANGRAPHS_PITCHER_DIR = PROJECT_ROOT / "MLB Player Data/FanGraphs_Data/pitchers"
FANGRAPHS_HITTER_DIR = PROJECT_ROOT / "MLB Player Data/FanGraphs_Data/hitters"
DEFENSIVE_DIR = PROJECT_ROOT / "MLB Player Data/FanGraphs_Data/defensive"

# Statcast data directories
STATCAST_DIR = PROJECT_ROOT / "MLB Player Data/Statcast_Data"
STATCAST_RUNNING_SPLITS_DIR = PROJECT_ROOT / "MLB Player Data/Statcast_Data/running_splits"

# Baseball Prospectus data directories
BP_HITTER_DIR = PROJECT_ROOT / "MLB Player Data/BP_Data/hitters"
BP_PITCHER_DIR = PROJECT_ROOT / "MLB Player Data/BP_Data/pitchers"
BP_BASERUNNING_DIR = PROJECT_ROOT / "MLB Player Data/BP_Data/baserunning"

# Cache directory
CACHE_DIR = PROJECT_ROOT / "cache"


# ============================================================================
# Feature Lists (for validation and iteration)
# ============================================================================

# All pitcher features (11 total)
PITCHER_FEATURES = [
    COL_BB_PCT,
    COL_K_PCT,
    COL_SWSTR_PCT,
    COL_WPA_LI,
    COL_LOB_PCT,
    COL_HARD_PCT,
    COL_ERA,
    COL_GB_PCT,
    COL_HR_FB_PCT,
    COL_AVG_HIT_ANGLE,
    COL_ANGLE_SWEET_SPOT_PCT,
    COL_RUNNING_CONTROL
]

# All hitter features (10 total)
HITTER_FEATURES = [
    COL_K_PCT,
    COL_BB_PCT,
    COL_PA,
    COL_GDP,
    COL_AVG,
    COL_OBP,
    COL_SLG,
    COL_POSITIONAL_WAR,
    COL_ENHANCED_BASERUNNING,
    COL_ENHANCED_DEFENSE
]

# Critical features that must not have NaN (pitchers)
PITCHER_CRITICAL_FEATURES = [COL_BB_PCT, COL_K_PCT]

# Critical features that must not have NaN (hitters)
HITTER_CRITICAL_FEATURES = [COL_BB_PCT, COL_K_PCT, COL_PA]


# ============================================================================
# Modeling Feature Lists (for model input)
# ============================================================================

# Pitcher modeling features (19 total: 14 base + 5 injury features)
# ORDER MATCHES PITCHER_MONOTONIC_CONSTRAINTS in pitcher_ensemble.py
# These are the final features used for model training after pipeline processing
PITCHER_MODEL_FEATURES = [
    # Base and composite features
    'BB%',                              # Base feature
    'K%',                               # Base feature
    'ERA',                              # Base feature
    'GB%',                              # Base feature
    'SwStr%',                           # Base feature
    'WPA/LI',                           # Base feature
    'damage_control_ratio',             # Composite feature
    'Opportunity_Success',              # Composite feature
    'strikeout_efficiency',             # Composite feature
    'contact_management',               # Composite feature
    'strikeout_contact_quality',        # Composite feature
    'Statcast_Launch_Quality_Index',    # Composite feature
    'Running_Control',                  # Base feature
    'SD_MD_Net',                        # Composite feature (reliever-specific signal)
    # Injury features (added Phase 1: Injury Feature Engineering)
    'has_injury_data',
    'had_tommy_john_ever',
    'years_since_tommy_john',
    'total_il_days_past_year',
    'had_major_injury_past_year'
]

# Hitter modeling features (14 total: 9 base + 5 injury features)
# ORDER MATCHES HITTER_MONOTONIC_CONSTRAINTS in hitter_ensemble.py
# These are the final features used for model training after pipeline processing
HITTER_MODEL_FEATURES = [
    # Base features
    'K%',                    # Base feature
    'BB%',                   # Base feature
    'AVG',                   # Base feature
    'OBP',                   # Base feature
    'SLG',                   # Base feature
    'GDP',                   # Base feature (derived from GDP count)
    'Positional_WAR',        # Derived feature
    'Enhanced_Baserunning',  # Derived feature
    'Enhanced_Defense',      # Derived feature
    # Injury features (added Phase 1: Injury Feature Engineering)
    'has_injury_data',
    'had_tommy_john_ever',
    'years_since_tommy_john',
    'total_il_days_past_year',
    'had_major_injury_past_year'
]


# ============================================================================
# Logging
# ============================================================================

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
