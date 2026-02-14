"""
Constants for the Team Wins Projection System.

Centralizes all magic numbers, budgets, thresholds, and mappings
used across the team wins modules.
"""

# ============================================================================
# League Structure
# ============================================================================

TOTAL_LEAGUE_WINS = 2430        # 30 teams * 162 games / 2
GAMES_PER_SEASON = 162
NUM_TEAMS = 30

# Replacement level: a team of all replacement-level players (0 WAR)
# Total league WAR above replacement = 1000 (from existing constraint optimizer)
# Replacement wins = (2430 - 1000) / 30 = 47.67
LEAGUE_TOTAL_WAR = 1000.0
REPLACEMENT_WINS_PER_TEAM = (TOTAL_LEAGUE_WINS - LEAGUE_TOTAL_WAR) / NUM_TEAMS  # 47.67


# ============================================================================
# Team PA/IP Budgets
# ============================================================================

TEAM_PA_BUDGET = 5700           # Typical team total PA per 162-game season
TEAM_IP_BUDGET = 1450           # Typical team total IP per 162-game season
ROTATION_IP_BUDGET = 900        # Starting pitchers share ~900 IP (quality-weighted)
BULLPEN_IP_BUDGET = 550         # Relievers + closer share ~550 IP


# ============================================================================
# Hitter PA Allocation
# ============================================================================

# Lineup-order PA range for starters (top of order -> bottom)
STARTER_PA_HIGH = 625           # Leadoff / top-of-order PA
STARTER_PA_LOW = 550            # Bottom-of-order PA
NUM_LINEUP_SPOTS = 9            # 8 fielding positions + DH

# Bench hitter caps
MAX_HITTER_PA = 700             # Absolute cap per hitter
MAX_BENCH_PA = 350              # Bench player cap


# ============================================================================
# Pitcher IP Allocation
# ============================================================================

MIN_STARTER_IP = 40             # Floor per starter (spot start contribution)
MAX_STARTER_IP = 220            # Absolute cap per starting pitcher
CLOSER_BASE_IP = 65             # Standard closer workload
MAX_RELIEVER_IP = 80            # Reliever cap (non-closer)
MIN_RELIEVER_IP = 20            # Floor per reliever


# ============================================================================
# Age Adjustments
# ============================================================================

AGE_PA_PENALTY_START = 35       # Begin reducing PA at age 35+
AGE_PA_PENALTY_PER_YEAR = 25    # Lose ~25 PA per year over threshold

AGE_IP_PENALTY_START = 36       # Begin reducing IP at age 36+
AGE_IP_PENALTY_PER_YEAR = 15    # Lose ~15 IP per year over threshold


# ============================================================================
# Valid Roles
# ============================================================================

HITTER_ROLES = {'starter_hitter', 'bench_hitter', 'replacement_hitter'}
PITCHER_ROLES = {'starter_pitcher', 'reliever', 'closer', 'swing_pitcher', 'replacement_pitcher'}
ALL_ROLES = HITTER_ROLES | PITCHER_ROLES
REPLACEMENT_ROLES = {'replacement_hitter', 'replacement_pitcher'}

# Map roles to player type
ROLE_TO_PLAYER_TYPE = {
    'starter_hitter': 'hitter',
    'bench_hitter': 'hitter',
    'replacement_hitter': 'hitter',
    'starter_pitcher': 'pitcher',
    'reliever': 'pitcher',
    'closer': 'pitcher',
    'swing_pitcher': 'pitcher',
    'replacement_pitcher': 'pitcher',
}

# Valid positions
HITTER_POSITIONS = {'C', '1B', '2B', 'SS', '3B', 'LF', 'CF', 'RF', 'DH'}
PITCHER_POSITIONS = {'SP', 'RP'}
ALL_POSITIONS = HITTER_POSITIONS | PITCHER_POSITIONS


# ============================================================================
# Team Abbreviations
# ============================================================================

MLB_TEAMS = {
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE',
    'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL',
    'MIN', 'NYM', 'NYY', 'ATH', 'PHI', 'PIT', 'SDP', 'SFG',
    'SEA', 'STL', 'TBR', 'TEX', 'TOR', 'WSN'
}


# ============================================================================
# Division and League Mappings
# ============================================================================

DIVISION_MAP = {
    'BAL': ('AL', 'AL East'), 'BOS': ('AL', 'AL East'),
    'NYY': ('AL', 'AL East'), 'TBR': ('AL', 'AL East'),
    'TOR': ('AL', 'AL East'),

    'CHW': ('AL', 'AL Central'), 'CLE': ('AL', 'AL Central'),
    'DET': ('AL', 'AL Central'), 'KCR': ('AL', 'AL Central'),
    'MIN': ('AL', 'AL Central'),

    'HOU': ('AL', 'AL West'), 'LAA': ('AL', 'AL West'),
    'ATH': ('AL', 'AL West'), 'SEA': ('AL', 'AL West'),
    'TEX': ('AL', 'AL West'),

    'ATL': ('NL', 'NL East'), 'MIA': ('NL', 'NL East'),
    'NYM': ('NL', 'NL East'), 'PHI': ('NL', 'NL East'),
    'WSN': ('NL', 'NL East'),

    'CHC': ('NL', 'NL Central'), 'CIN': ('NL', 'NL Central'),
    'MIL': ('NL', 'NL Central'), 'PIT': ('NL', 'NL Central'),
    'STL': ('NL', 'NL Central'),

    'ARI': ('NL', 'NL West'), 'COL': ('NL', 'NL West'),
    'LAD': ('NL', 'NL West'), 'SDP': ('NL', 'NL West'),
    'SFG': ('NL', 'NL West'),
}

# Division ordering for standings display
DIVISION_ORDER = [
    'AL East', 'AL Central', 'AL West',
    'NL East', 'NL Central', 'NL West'
]


# ============================================================================
# FV Prospect Integration
# ============================================================================

from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[4]

# FV grade -> expected first-year WAR: (hitter, pitcher)
# Based on FanGraphs tier definitions with first-year discount.
# Pitchers discounted ~20% vs hitters due to higher bust rates.
FV_TO_FIRST_YEAR_WAR = {
    '80':  (3.5, 3.0),
    '70':  (2.8, 2.3),
    '65':  (2.2, 1.8),
    '60':  (1.8, 1.5),
    '55':  (1.4, 1.1),
    '50':  (1.0, 0.8),
    '45+': (0.7, 0.55),
    '45':  (0.5, 0.4),
    '40+': (0.3, 0.2),
    '40':  (0.15, 0.1),
    '35+': (0.05, 0.0),
    '35':  (0.0, 0.0),
}

# Risk modifier on blending confidence
FV_RISK_CONFIDENCE = {'Low': 1.0, 'Med': 0.85, 'High': 0.70}

# Rookie thresholds (career PA/IP below these = eligible for FV blending)
ROOKIE_PA_THRESHOLD = 200
ROOKIE_IP_THRESHOLD = 80

# Blending alpha: weight on statistical projection
# At 0 career PA/IP -> alpha = 0.3 (70% FV weight)
# At threshold -> alpha = 0.7 (30% FV weight)
# Above threshold -> alpha = 1.0 (pure projection, no FV)
FV_BLEND_ALPHA_MIN = 0.30
FV_BLEND_ALPHA_MAX = 0.70

# Data paths for prospect files
FANGRAPHS_PROSPECT_DIR = PROJECT_ROOT / "MLB Player Data/FanGraphs_Data/prospects"
FANGRAPHS_INTL_PROSPECT_DIR = PROJECT_ROOT / "MLB Player Data/FanGraphs_Data/international_prospects"


# ============================================================================
# MLE (Minor League Equivalency) Translation
# ============================================================================

MILB_DATA_DIR = PROJECT_ROOT / "MLB Player Data/FanGraphs_Data/minor_leaguers"
MLE_MIN_PA = 100       # Minimum PA in both AAA and MLB for translation model
MLE_MIN_IP = 30        # Minimum IP in both AAA and MLB for translation model
MLE_WAR_FLOOR = -0.5   # Floor for MLE WAR estimates
MLE_WAR_CAP = 2.5      # Cap for MLE WAR estimates
MLE_AGE_BONUS_THRESHOLD = 25   # Under this age -> +adjustment (young upside)
MLE_AGE_PENALTY_THRESHOLD = 29  # At or above this age -> -adjustment (AAAA ceiling)
MLE_AGE_ADJUSTMENT = 0.2       # WAR adjustment magnitude for age
