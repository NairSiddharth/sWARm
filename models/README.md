# sWARm Model Repository

This directory contains the versioned ensemble models for the sWARm (simplified WAR model) system.

## Directory Structure

### `/models/ensemble_models.pkl`
**Current Production Model**: The latest trained ensemble model currently in use by the sWARm system.

### `/models/history/`
**Model Version History**: Archived versions of ensemble models as features and metrics evolve.

## Model Naming Convention

Historical models follow the pattern:
`ensemble_models_{feature_description}_{YYYYMMDD}_{HHMMSS}.pkl`

### Current Model Versions

- **ensemble_models_10feat_20250928_001119.pkl** - 10-feature model (latest)
- **ensemble_models_normalized_20250927_235106.pkl** - Normalized features model (v2)
- **ensemble_models_normalized_20250927_234949.pkl** - Normalized features model (v1)

## Model Evolution Timeline

1. **Normalized Features Era** (Sept 27, 2025)
   - Added normalized Contact Quality Index (CQI)
   - Added normalized Statcast Launch Quality Index (SLQI)
   - Fixed HBP% vs raw HBP count issue

2. **10-Feature Model** (Sept 28, 2025)
   - Current production model
   - Enhanced feature set with latest improvements

## Usage

Models are loaded by the ensemble prediction system in:
- `common_modules/ensemble_modeling.py`
- Future projection notebooks
- Current season prediction workflows

## Model Persistence Strategy

- **Production**: Keep latest model as `ensemble_models.pkl` in `/models/`
- **Versioning**: Archive significant model iterations in `/models/history/`
- **Naming**: Include feature description and timestamp for historical tracking
- **Testing**: Validate model performance before promoting to production

## Integration Points

These models are used by:
- sWARm_FutureProjections.ipynb
- sWARm_CS.ipynb
- Real-time prediction workflows
- Validation and testing systems in `/testing/`