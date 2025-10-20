# Current Season Modules

## Overview

The `current_season_modules` directory contains all functionality for analyzing and projecting current season (2025) baseball performance. These modules handle real-time data loading, participation rate calculations, injury adjustments, and season projections.

## Key Components

### Data Loading (`current_season_data_loading.py`)

Handles loading and processing of current season data from multiple sources.

**Key Functions:**

- `load_primary_datasets()`: Loads all primary CSV datasets for analysis
- `calculate_team_games_from_hitters(year, data_source)`: **NEW** - Calculates actual games played by each team using hitter data for accurate participation rates

### Participation Rate Calculator (`participation_rate_calculator.py`)

Calculates realistic remaining games projections based on actual usage patterns.

**Key Features:**

- **Team-Aware Projections**: Uses actual team games played (from hitter data) instead of estimates
- **Role Detection**: Identifies player roles (everyday, platoon, backup, etc.)
- **Performance Adjustments**: Boosts projections for elite performers
- **Multi-Position Support**: Handles modern versatile players

**Usage Example:**

```python
from current_season_modules.current_season_data_loading import calculate_team_games_from_hitters
from current_season_modules.participation_rate_calculator import calculate_participation_adjusted_games_with_team_data

# Get actual team games
team_games = calculate_team_games_from_hitters(2025, 'fangraphs')

# Calculate participation-adjusted projections
projection = calculate_participation_adjusted_games_with_team_data(
    player_data=player_row,
    team_games_dict=team_games,
    current_war=2.5,
    injury_adjustment=0.9
)
```

### Game Progress Calculator (`game_progress_calculator.py`)

Estimates season progress and remaining games based on calendar date.

### Injury Recovery Calculator (`injury_recovery_calculator.py`)

Calculates recovery timelines and performance impact from injuries.

### Real-Time Data Loader (`real_time_data_loader.py`)

Loads and processes 2025 first-half data for current season projections.

### Modeling Module (`modeling/`)

Contains data preparation and model training components:

- `data_preparation.py`: Prepares data for K-fold cross-validation
- `data_loading.py`: Loads expanded FanGraphs and BP data

### Visualization Module (`visualization/`)

Creates interactive visualizations for projections and analysis.

### Warp Calculator (`warp_calculator.py`)

Calculates Baseball Prospectus WARP metric adjustments.

### Retrain Ensemble Models (`retrain_ensemble_models.py`)

Retrains ensemble models with updated data.

## Recent Updates (v2.1.0)

### Team Games Integration

- **Problem Solved**: Pitcher projections were using blanket assumptions (32 games for all starters)
- **Solution**: Calculate actual team games from hitter data
- **Impact**: More accurate projections for injured players, two-way players, and late-season callups

### Participation Rate Improvements

- Now uses actual team games instead of estimates
- Properly handles players with limited usage (like Ohtani's pitching)
- Prevents over-projection of remaining games

### Example Impact

**Before (Ohtani pitching projection):**

- 5 games pitched → projected 27 more (32 total)
- Unrealistic for a two-way player

**After:**

- 5 games in 81 team games = 6.2% participation
- Projects ~5 more games (realistic!)

## Usage Guidelines

### For Current Season Projections

1. **Load team games first:**

```python
team_games = calculate_team_games_from_hitters(2025, 'fangraphs')
```

2. **Pass to projection functions:**

```python
from common_modules.pitcher_workload_calculator import calculate_pitcher_projections

projection = calculate_pitcher_projections(
    player_data=pitcher_row,
    ensemble_predictor=model,
    player_feature_vector=features,
    team_games_dict=team_games  # Pass actual team games
)
```

### For Hitter Projections

Use the participation rate calculator with team games for accurate remaining games projections.

## Dependencies

- `common_modules/`: Core calculation modules
- `pandas`, `numpy`: Data manipulation
- `sklearn`: Machine learning models
- FanGraphs/BP data files in `MLB Player Data/`

## Known Limitations

- Assumes 5-man rotation for all teams (future improvement: detect rotation size)
- Requires manual data updates for 2025 second half
- Two-way player handling could be more sophisticated

## Contributing

When adding new features to current season modules:

1. Update this README with new functionality
2. Follow CODING_PRINCIPLES.md guidelines
3. Add proper type hints and docstrings
4. Include usage examples
