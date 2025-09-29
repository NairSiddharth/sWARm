# MLB Player Data Repository

This directory contains comprehensive baseball datasets from multiple sources used by the sWARm (simplified WAR model) analytics system.

## Purpose
Central repository for all raw and processed baseball statistics, including traditional metrics, advanced analytics, and modern tracking data spanning multiple seasons (2016-2024).

## Data Sources

### Primary Sources
- **FanGraphs** - Advanced baseball statistics and analytics
- **Baseball Prospectus (BP)** - Alternative advanced metrics and WARP calculations
- **MLB Statcast** - Player tracking and batted ball data
- **MLB Official** - Traditional statistics and awards data

## Directory Structure

### `/FanGraphs_Data/` - FanGraphs Advanced Analytics
- **`hitters/`** - Hitting statistics (wRC+, wOBA, ISO, etc.)
- **`pitchers/`** - Pitching statistics (FIP, xFIP, SIERA, etc.)
- **`defensive/`** - Defensive metrics (UZR, OAA, Statcast defensive)
- **`park_factors/`** - Ballpark adjustment factors by season
- **`injuries/`** - Player injury tracking and recovery data

### `/BP_Data/` - Baseball Prospectus Analytics
- **`hitters/`** - BP hitting metrics and WARP calculations
- **`pitchers/`** - BP pitching metrics and WARP calculations
- **`fielding/`** - BP defensive metrics (FRAA)
- **`baserunning/`** - Baserunning value metrics

### `/Statcast_Data/` - MLB Advanced Tracking Data
- **`batted_ball/`** - Exit velocity, launch angle, hit probability
- **`catch_probability/`** - Defensive catch probability data
- **`exit_velocity/`** - Exit velocity distributions and percentiles
- **`expected_stats/`** - xBA, xSLG, xwOBA calculations
- **`running_splits/`** - Sprint speed and baserunning analytics

### `/Original_Data/` - Raw Source Files
- Unprocessed data files as downloaded from sources
- Backup copies of original datasets
- Historical data archives

### `/awards/` - Player Recognition Data
- Award winners and voting results
- All-Star selections and honors
- Hall of Fame information

## Archive Files
- **`batting.zip`** - Compressed historical batting data
- **`fielding.zip`** - Compressed historical fielding data
- **`pitching.zip`** - Compressed historical pitching data

## Data Coverage

### Temporal Coverage
- **Primary Analysis Period**: 2016-2024 (9 seasons)
- **Historical Context**: Some datasets extend to earlier seasons
- **Current Season**: 2024 data updated through regular season

### Player Coverage
- **Hitters**: ~2,000+ unique players across timespan
- **Pitchers**: ~2,500+ unique players across timespan
- **Minimum Thresholds**: Varies by analysis (typically 37+ PA for hitters, 7+ IP for pitchers)

## Data Integration Notes

### Player Identification
- **Primary Key**: MLBAID (when available)
- **Fallback**: Name matching with manual verification
- **Cross-Reference**: FanGraphs ID, BP ID linkage maintained

### Data Quality Considerations
- **Missing Data**: Handled through multiple imputation strategies
- **Inconsistencies**: Cross-validated between sources when possible
- **Updates**: FanGraphs data most current, BP data may lag
- **Statcast Limitations**: Available from 2015+, full coverage from 2016+

### Known Issues
- **Name Variations**: Player name spelling inconsistencies across sources
- **Position Changes**: Players switching positions mid-season
- **Two-Way Players**: Special handling for pitcher/hitter combinations
- **Injury Adjustments**: Incomplete injury data may affect projections

## Usage in sWARm System

### Data Loading
- **Module**: `future_season_modules/data_integration.py`
- **Cache**: Processed data cached in `/cache/` directory
- **Updates**: Manual refresh required for new data

### Feature Engineering
- **Enhanced Features**: Contact quality, launch angle adjustments
- **Park Factors**: Applied via `common_modules/park_factors.py`
- **Positional Adjustments**: Via `common_modules/positional_adjustments.py`

### Quality Validation
- **Data Checks**: Automated validation in data loading pipeline
- **Issue Logging**: Problems logged to `results/issues/dropped_players_log.txt`
- **Cross-Validation**: Multiple source comparison for accuracy

## Maintenance Notes

### Update Frequency
- **Season End**: Complete season data refresh
- **Mid-Season**: Monthly updates for current season
- **Injury Data**: Weekly updates during active season

### File Management
- **Archive**: Compress historical data annually
- **Backup**: Original files preserved in Original_Data/
- **Cleanup**: Remove temporary/intermediate files regularly

### Dependencies
- **sWARm Analysis**: All modules depend on this data
- **External Sources**: Requires active subscriptions/access to data providers
- **Processing Power**: Large datasets require adequate memory for processing

## Future Enhancements
- **Real-Time Integration**: Automated daily updates during season
- **Additional Sources**: Potential integration with other data providers
- **Data Validation**: Enhanced automated quality checking
- **Compression**: More efficient storage for historical data