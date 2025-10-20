# sWARm Research Methodology

Research methodology, data sources, and methodological contributions for the sWARm project.

---

## Table of Contents

- [Overview](#overview)
- [Data Sources & Statistical Foundations](#data-sources--statistical-foundations)
- [Methodological Contributions](#methodological-contributions)
- [Copyright & Intellectual Property](#copyright--intellectual-property)

---

## Overview

sWARm (Sid Wins Above Replacement Metric) combines data from multiple authoritative baseball analytics sources with novel methodological approaches to create accurate, interpretable player value projections. This document outlines the research foundations, data sources, and original contributions that underpin the system.

---

## Data Sources & Statistical Foundations

### Primary Data Sources

- **FanGraphs**: Sullivan, Jeff, et al. "FanGraphs Sabermetrics Library." *FanGraphs*, 2002-2025. [https://www.fangraphs.com/](https://www.fangraphs.com/)
  - Comprehensive baseball statistics and advanced metrics
  - Industry-standard WAR calculations and component metrics
  - 2016-2024 historical data coverage

- **Baseball Prospectus**: Silver, Nate, et al. "WARP (Wins Above Replacement Player) Methodology." *Baseball Prospectus*, 2003-2025. [https://www.baseballprospectus.com/](https://www.baseballprospectus.com/)
  - Alternative WAR implementation (WARP)
  - DRC+ and other proprietary metrics
  - Complementary perspective on player value

- **MLB Statcast**: "Statcast Search." *Baseball Savant*, Major League Baseball, 2015-2025. [https://baseballsavant.mlb.com/](https://baseballsavant.mlb.com/)
  - High-resolution tracking data
  - Exit velocity, launch angle, sprint speed
  - Advanced defensive metrics (Outs Above Average)

### Data Integration Approach

- **Coverage**: 2016-2024 (9 years)
- **Volume**: 195MB preprocessed cache
- **Players**: Thousands of MLB hitters and pitchers
- **Features**: 50+ raw metrics harmonized across sources

---

## Methodological Contributions

### Original Research & Analysis

#### Manual K%/BB% Calculations
Novel methodology for calculating pre-2020 Baseball Prospectus derived statistics, ensuring 100% feature coverage across all years (2016-2024). This addressed a critical gap where certain statistics were not available in early years of the dataset.

**Key innovations:**
- Backward-compatible calculation methods
- Validation against available data years
- Consistent feature availability across temporal boundaries

#### Strategic Feature Selection
7-feature focused approach prioritizing interpretability and statistical significance over quantity. Rather than using hundreds of features, sWARm carefully selects the most predictive and interpretable metrics.

**Rationale:**
- Reduces overfitting risk
- Improves model interpretability
- Focuses on statistically robust relationships
- Easier to validate and debug

#### Enhanced Baserunning Analytics
Run expectancy matrix-based calculations for situational baserunning value assessment. This goes beyond simple stolen base metrics to capture the full value of baserunning decisions.

**Components:**
- Context-aware baserunning value
- Run expectancy integration
- Situational decision quality metrics

### Data Integration Innovations

#### Multi-Source Harmonization
Comprehensive integration of FanGraphs, Baseball Prospectus, and MLB Statcast data with advanced name matching algorithms. Player identification across data sources presents significant challenges that required custom solutions.

**Technical approach:**
- Fuzzy name matching algorithms
- MLBAMID-based cross-referencing
- Manual validation for edge cases
- Duplicate name disambiguation

#### Temporal Consistency
Standardized feature engineering across 9-year dataset spanning significant rule and measurement changes in baseball (e.g., 2019 ball changes, 2020 shortened season, evolving defensive metrics).

**Challenges addressed:**
- Evolving measurement systems
- Rule changes affecting statistics
- Data availability variations across years
- Metric definition changes

### Model Architecture Innovations

#### Multi-Quantile Uncertainty Quantification
Simultaneous prediction of 10th, 50th, and 90th percentiles using HistGradientBoosting ensemble. This provides uncertainty bounds rather than single-point estimates.

**Advantages:**
- Risk assessment for player evaluation
- Confidence intervals for projections
- Downside/upside scenario planning

#### Future Projection Framework
Novel integration of:
- **Longitudinal modeling**: Year-to-year WAR progression patterns
- **Cox Proportional Hazards**: Retirement probability estimation
- **Position-specific age curves**: Differential aging by position (catchers: 3.5%/year decline, DH: 1.5%/year)
- **Elite player protection**: Separate modeling for MVP/Superstar/All-Star tiers

#### Temporal Validation
Rigorous train-test split preventing data leakage:
- **Training**: 2016-2022 (7 years)
- **Testing**: 2023 (held-out year)
- **Performance**: R² = 0.362 (hitters), 0.386 (pitchers)

---

## Copyright & Intellectual Property

### Original Work

**Code & Implementation:**
- All code, analysis, and documentation: © 2025 Siddharth Nair
- Original algorithms and methodological improvements: © 2025 Siddharth Nair
- Licensed under Mozilla Public License 2.0 (MPL-2.0)

**Novel Contributions:**
- Multi-quantile ensemble architecture
- Future projection framework combining survival analysis with age curves
- Elite player adjustment system
- Temporal validation methodology
- Data harmonization algorithms

### Data Acknowledgments

**Usage Rights:**
- Baseball statistics used under fair use provisions for research and analysis
- All commercial data sources properly licensed and attributed
- No proprietary data redistributed

**Attribution:**
- FanGraphs metrics and calculations remain property of FanGraphs
- Baseball Prospectus WARP methodology remains property of Baseball Prospectus
- MLB Statcast data remains property of Major League Baseball

---

## See Also

- [README.md](../README.md) - Project overview and quick start
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture details
