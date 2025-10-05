"""
Elite Player Adjustment Module
=============================

DEPRECATED: This module has been refactored into three separate modules for better maintainability.

This module is maintained for backward compatibility only.
It will be removed in version 3.0.0.

Please update your imports:
	from legacy_modules.elite_adjustment import ElitePlayerAdjuster  # OLD (deprecated)
	from common_modules.elite_adjustment_base import ElitePlayerAdjuster  # NEW

	from legacy_modules.elite_adjustment import TwoWayEliteProtection  # OLD (deprecated)
	from common_modules.two_way_elite_protection import TwoWayEliteProtection  # NEW

	from legacy_modules.elite_adjustment import RookieEliteProtection  # OLD (deprecated)
	from common_modules.rookie_elite_protection import RookieEliteProtection  # NEW

The module has been split into:
- elite_adjustment_base.py: Base ElitePlayerAdjuster class
- two_way_elite_protection.py: TwoWayEliteProtection class for two-way players
- rookie_elite_protection.py: RookieEliteProtection class for rookies

Implements the two-stage pipeline approach for Option C:
Base Projections → Elite Adjustment → Constraint Optimization

This module applies confidence-based regression reduction to protect elite players
from over-regression before mathematical constraints are applied.
"""

from common_modules.rookie_elite_protection import (
    PITCHER_IP_THRESHOLD,
    HITTER_AB_THRESHOLD,
    MIN_CURRENT_IP,
    MIN_CURRENT_AB,
    MIN_WAR_THRESHOLD,
    ROOKIE_CEILING
)
from common_modules.rookie_elite_protection import RookieEliteProtection
from common_modules.two_way_elite_protection import TwoWayEliteProtection
from common_modules.elite_adjustment_base import ElitePlayerAdjuster
__version__ = '2.0.0'
__author__ = 'oWAR Development Team'

import warnings

# Issue deprecation warning
warnings.warn(
    "The elite_adjustment module has been refactored. "
    "Please update your imports to use the new module structure. "
    "See module docstring for details. "
    "This compatibility module will be removed in version 3.0.0.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all classes from new modules for backward compatibility

# Re-export constants for backward compatibility

# Make all exported items available when using "from elite_adjustment import *"
__all__ = [
    'ElitePlayerAdjuster',
    'TwoWayEliteProtection',
    'RookieEliteProtection',
    'PITCHER_IP_THRESHOLD',
    'HITTER_AB_THRESHOLD',
    'MIN_CURRENT_IP',
    'MIN_CURRENT_AB',
    'MIN_WAR_THRESHOLD',
    'ROOKIE_CEILING'
]
