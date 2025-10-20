"""
Future season feature loaders.

Optimized for year-to-year prediction with high-correlation features.
"""

# Import all pitcher loaders (new + reused)
from .pitcher_loaders import (
    # New loaders
    load_contact_pct_all_years,
    load_o_swing_pct_all_years,
    load_zone_pct_all_years,
    load_o_contact_pct_all_years,
    load_f_strike_pct_all_years,
    # Reused loaders
    load_bb_pct_all_years as load_pitcher_bb_pct_all_years,
    load_k_pct_all_years as load_pitcher_k_pct_all_years,
    load_swstr_all_years,
    load_gb_pct_park_adjusted,
    load_wpa_li_all_years,
    load_running_control_all_years,
    load_sd_all_years,
    load_md_all_years,
    load_hard_pct_all_years as load_pitcher_hard_pct_all_years,
    load_statcast_data
)

# Import all hitter loaders (new + reused)
from .hitter_loaders import (
    # New loaders
    load_iso_all_years,
    load_gb_pct_all_years as load_hitter_gb_pct_all_years,
    load_hr_fb_pct_all_years,
    load_hard_pct_all_years as load_hitter_hard_pct_all_years,
    load_pull_pct_all_years,
    # Reused loaders
    load_k_pct_all_years as load_hitter_k_pct_all_years,
    load_bb_pct_all_years as load_hitter_bb_pct_all_years,
    load_pa_all_years,
    load_gdp_all_years,
    load_avg_park_adjusted,
    load_obp_park_adjusted,
    load_slg_park_adjusted,
    load_positional_war,
    load_positions_all_years,
    load_enhanced_baserunning,
    load_enhanced_defense
)

__all__ = [
    # Pitcher loaders
    'load_contact_pct_all_years',
    'load_o_swing_pct_all_years',
    'load_zone_pct_all_years',
    'load_o_contact_pct_all_years',
    'load_f_strike_pct_all_years',
    'load_pitcher_bb_pct_all_years',
    'load_pitcher_k_pct_all_years',
    'load_swstr_all_years',
    'load_gb_pct_park_adjusted',
    'load_wpa_li_all_years',
    'load_running_control_all_years',
    'load_sd_all_years',
    'load_md_all_years',
    'load_pitcher_hard_pct_all_years',
    'load_statcast_data',
    # Hitter loaders
    'load_iso_all_years',
    'load_hitter_gb_pct_all_years',
    'load_hr_fb_pct_all_years',
    'load_hitter_hard_pct_all_years',
    'load_pull_pct_all_years',
    'load_hitter_k_pct_all_years',
    'load_hitter_bb_pct_all_years',
    'load_pa_all_years',
    'load_gdp_all_years',
    'load_avg_park_adjusted',
    'load_obp_park_adjusted',
    'load_slg_park_adjusted',
    'load_positional_war',
    'load_positions_all_years',
    'load_enhanced_baserunning',
    'load_enhanced_defense'
]
