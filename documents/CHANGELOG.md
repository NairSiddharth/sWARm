# sWARm Change Log

All notable changes to the sWARm project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [4.0.0] - October 2025 - New Pipeline Architecture & Complete Module Migration

### Breaking Changes
- Complete migration from old modules to new_pipeline/ architecture
- Old modules (common_modules/, current_season_modules/, future_season_modules/) deprecated and replaced

### Added
- **Architecture**: New modular system: new_pipeline/{common, models, notebooks, tests}/
- **Models**: Multi-quantile HistGradientBoosting ensemble models (10th, 50th, 90th percentiles)
- **Current Season**: Enhanced current season modeling with uncertainty quantification
- **ROS Projections**: Rest-of-season projection models for hitters and pitchers
- **Future Projections**: Complete 1-3 year projection system with:
  - Longitudinal modeling (year-to-year WAR progression)
  - Cox Proportional Hazards survival modeling (retirement probability)
  - Position-specific age curves (catcher: 3.5%/year, DH: 1.5%/year decline)
  - Joint projection model combining WAR + survival + aging
  - Elite player adjustment system (MVP/Superstar/All-Star protection tiers)
  - Temporal validation framework (train 2016-2022, test 2023)
- **Features**: Comprehensive feature engineering:
  - Confidence scoring system (0-8 scale based on performance, age, position, volume)
  - Injury feature engineering with recovery impact modeling
  - Elite player detection and rookie classification
  - Enhanced baserunning and defensive metrics integration
- **Testing**: Comprehensive pytest suite with 18 test modules covering integration, models, features, and temporal validation
- **Notebooks**: Reorganized into sWARm_overview.ipynb (main), sWARm_deep_dive.ipynb, hitter/pitcher-specific pipelines, and future projection notebooks

### Changed
- **Dependencies**: Split into requirements-core.txt (15 packages) and requirements-dev.txt (6 packages) - 89% reduction from 154 packages
- **Data Management**: Transitive dependency handling - pip auto-installs 130+ sub-dependencies

### Performance
- Temporal validation R²: 0.362 (hitters), 0.386 (pitchers) on 2023 holdout data

---

## [3.0.0] - September 2025 - Major Architecture Overhaul & Advanced Analytics

### Breaking Changes
- Complete repository restructure and modular architecture redesign

### Added
- **Architecture**: New modular system with common_modules/, current_season_modules/, future_season_modules/
- **Features**: Massive feature expansion - contact quality metrics, Statcast integration, percentage standardization, LOB%, damage control ratio, elite adjustments
- **Notebooks**: Split into specialized sWARm_CS.ipynb and sWARm_FutureProjections.ipynb
- **Testing**: Comprehensive testing framework with organized validation, integration, and performance testing
- **Models**: Complete model version control system with production and historical storage
- **Documentation**: Full documentation overhaul including Claude Code framework
- **Data Pipeline**: Enhanced multi-source integration with advanced feature engineering

---

## [2.1.0] - September 2025 - Feature Complete v1

### Added
- LICENSE file
- Comprehensive PLAN.md
- Feature-complete analysis notebooks

### Changed
- **Major**: Comprehensive planning and documentation overhaul
- **Status**: All planned features implemented and working

---

## [2.0.3] - September 2025 - Data Quality Fixes

### Fixed
- Resolved pre-2020 BP data statistical mismatches
- Fixed animated visualizations

### Changed
- Standardized FanGraphs vs Baseball Prospectus feature alignment

---

## [2.0.2] - September 2025 - Code Stability

### Fixed
- General code fixes and bug resolution

### Changed
- Improved system reliability

---

## [2.0.1] - September 2025 - Organization & Documentation

### Added
- Created TODO.md
- Renamed files for consistency

### Changed
- Deprecated old files
- Improved naming conventions
- Enhanced project structure

---

## [2.0.0] - September 2025 - Modular Architecture

### Breaking Changes
- Complete modularization from monolithic structure

### Added
- **Architecture**: 24 specialized modules for better maintainability
- **Data**: Expanded coverage to 2016-2024 (vs single year)
- **Features**: Enhanced duplicate name handling, improved park factors

### Removed
- Poorly performing algorithms (AdaBoost, Gaussian Process)
- Spring league data contamination

---

## [1.3.0] - September 2025 - Enhanced Data & Visualization

### Added
- WARP data for 2016-2020
- Catcher framing metrics
- TODO tracking

### Changed
- Architecture: Began modularization process from 2000+ line files
- Documentation: Improved README

### Removed
- Deprecated code for clarity

---

## [1.2.0] - September 2025 - Expanded Dataset

### Added
- Baseball Prospectus data (2016-2020, 2022-2024)
- More parameters for model selection

### Changed
- **Accuracy**: Massively improved correlation calculations
- **Tuning**: Park factor adjustments (1.5 → 1.2)

---

## [1.1.0] - September 2025 - Performance Crisis & Recovery

### Changed
- **Challenge**: Major performance issues identified
- **Strategy**: Increased training data to address model weaknesses
- Improved data quality
- Feature re-evaluation and data source review

### Removed
- Spring training contamination

---

## [1.0.0] - September 2025 - Machine Learning Foundation

### Added
- Keras/TensorFlow neural networks
- XGBoost and traditional ML methods
- Enhanced graphs, fWAR/WARP comparisons

### Changed
- **Optimization**: Cached data mapping (many-to-one relationships)
- **Performance**: Significant speed improvements

---

## [0.2.0] - September 2025 - Advanced ML Integration

### Added
- Keras/TensorFlow integration
- XGBoost implementation

### Changed
- Improved player mapping and data cleaning
- **Performance**: Major optimizations for computational efficiency

---

## [0.1.1] - September 2025 - Initial Cleanup

### Added
- Uploaded cleaned code and datasets
- Established baseline functionality

---

## [0.1.0] - September 2025 - Project Genesis

### Added
- Initial project structure and concept
- Basic WAR calculation framework
