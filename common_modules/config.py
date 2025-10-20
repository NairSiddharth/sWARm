"""
Central configuration for oWAR project.

This module defines project-wide constants, paths, and configuration values
following the Configuration Management principles from CODING_PRINCIPLES.md.
"""

from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "MLB Player Data"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Specific data subdirectories
BP_DATA_DIR = DATA_DIR / "BP_Data"
FANGRAPHS_DATA_DIR = DATA_DIR / "FanGraphs_Data"
STATCAST_DATA_DIR = DATA_DIR / "Statcast_Data"

# Ensure directories exist
for dir_path in [CACHE_DIR, RESULTS_DIR, LOGS_DIR]:
	dir_path.mkdir(exist_ok=True, parents=True)

# Baseball constants
QUALIFIED_BATTER_MIN_PA = 502  # MLB qualification threshold
QUALIFIED_PITCHER_MIN_IP = 162.0  # 1 IP per team game
SEASON_GAMES = 162
MIN_MLB_SEASON = 1871
CURRENT_SEASON = 2025

# Positional adjustments (runs per 162 games, FanGraphs standard)
POSITIONAL_ADJUSTMENTS = {
    'C': 12.5,
    '1B': -12.5,
    '2B': 2.5,
    '3B': 2.5,
    'SS': 7.5,
    'LF': -7.5,
    'CF': 2.5,
    'RF': -7.5,
    'DH': -17.5
}

# Data processing constants
PRE_2020_CUTOFF = 2020  # Year when BP started providing K% and BB% directly
STANDARD_FILE_FILTER = 'standard'  # Filter to exclude standard files

# Statistical thresholds and defaults
DEFAULT_PARK_FACTOR = 1.0
MAX_INNINGS_PER_GAME = 9

# Pitcher feature defaults (league averages)
DEFAULT_BB_PCT = 9.0  # League average BB%
DEFAULT_K_PCT = 20.0  # League average K%
DEFAULT_ERA = 4.50  # League average ERA
DEFAULT_LOB_PCT = 72.0  # League average LOB%
DEFAULT_HR_FB_PCT = 10.0  # League average HR/FB%
DEFAULT_DAMAGE_CONTROL_RATIO = 2.4
DEFAULT_OPPORTUNITY_SUCCESS = 0.0

# Contact quality defaults (percentages)
DEFAULT_HARD_PCT = 35.0
DEFAULT_MED_PCT = 40.0
DEFAULT_SOFT_PCT = 25.0

# Normalized index defaults (centered at 50)
DEFAULT_CONTACT_QUALITY_INDEX = 50.0
DEFAULT_STATCAST_LAUNCH_QUALITY_INDEX = 50.0

# Feature calculation constants
OPTIMAL_LAUNCH_ANGLE = 14.2  # Empirically-derived optimal angle from analysis
DAMAGE_CONTROL_RATIO_MAX = 3.5  # Maximum reasonable damage control ratio (LOB% / (HR/9 + 0.5))
DAMAGE_CONTROL_RATIO_MIN = 0.5  # Minimum reasonable damage control ratio
SV_EFFICIENCY_CAP = 10.0  # Cap for extreme closers

# Normalization parameters for indices
Z_SCORE_SCALE = 15  # Scale factor for z-score normalization
Z_SCORE_CENTER = 50  # Center point for normalized indices

# Model configuration
DEFAULT_TEST_SPLIT = 0.2
RANDOM_STATE = 42  # For reproducibility
DEFAULT_HOLDOUT_YEAR = 2024
MODEL_CACHE_DIR = PROJECT_ROOT / "models"
MODEL_CACHE_PATH = MODEL_CACHE_DIR / "ensemble_models.pkl"
MODEL_HISTORY_DIR = MODEL_CACHE_DIR / "history"

# Ensure model directories exist
for dir_path in [MODEL_CACHE_DIR, MODEL_HISTORY_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# File naming patterns
FANGRAPHS_PITCHER_PATTERN = "fangraphs_pitchers_{year}_{type}.csv"
FANGRAPHS_FIRSTHALF_PATTERN = "fangraphs_pitchers_{year}_firsthalf_{type}.csv"
BP_PITCHER_PATTERN = "bp_pitchers_{year}.csv"
BP_HITTER_PATTERN = "bp_hitters_{year}.csv"
STATCAST_EXIT_VELOCITY_PATTERN = "exit_velocity_pitchers_{year}.csv"
FANGRAPHS_DEFENSIVE_ADVANCED_PATTERN = "fangraphs_defensive_advanced_{year}.csv"
FANGRAPHS_DEFENSIVE_STANDARD_PATTERN = "fangraphs_defensive_standard_{year}.csv"
FANGRAPHS_DEFENSIVE_STATCAST_PATTERN = "fangraphs_defensive_statcast_{year}.csv"

# Years for data loading
DEFAULT_DATA_YEARS = list(range(2016, 2026))  # 2016-2025
STATCAST_AVAILABLE_YEARS = list(range(2016, 2025))  # 2016-2024

# Current season specific configuration
from datetime import date

CURRENT_SEASON_CONFIG = {
	'season_year': CURRENT_SEASON,
	'season_start': date(CURRENT_SEASON, 3, 30),  # Late March/Early April
	'season_end': date(CURRENT_SEASON, 10, 1),  # Early October
	'all_star_break': date(CURRENT_SEASON, 7, 15),  # Mid-July
	'season_games': SEASON_GAMES,
	'qualification_thresholds': {
		'batter_min_pa': QUALIFIED_BATTER_MIN_PA,
		'pitcher_min_ip': QUALIFIED_PITCHER_MIN_IP
	}
}

# Game progress milestones
SEASON_MILESTONES = {
	'season_start': 0,
	'early_sample': 20,
	'month_sample': 30,
	'quarter_season': SEASON_GAMES // 4,
	'all_star_break': SEASON_GAMES // 2,
	'three_quarters': 3 * SEASON_GAMES // 4,
	'playoff_race': 7 * SEASON_GAMES // 8,
	'season_end': SEASON_GAMES
}

# Projection confidence thresholds (games played)
PROJECTION_CONFIDENCE = {
	'very_high': {'games': 100, 'confidence': 0.9},
	'high': {'games': 60, 'confidence': 0.8},
	'medium': {'games': 30, 'confidence': 0.7},
	'low': {'games': 15, 'confidence': 0.6},
	'very_low': {'games': 10, 'confidence': 0.5},
	'minimal': {'games': 0, 'confidence': 0.3}
}