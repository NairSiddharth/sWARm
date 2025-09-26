"""
Pitcher Workload Calculator
Handles realistic pitcher workload projections based on role (starter vs reliever)
"""

import pandas as pd
import numpy as np


def classify_pitcher_role(games_pitched, innings_pitched, games_started=None):
    """
    Classify pitcher as starter or reliever based on usage patterns

    Args:
        games_pitched: Number of games pitched
        innings_pitched: Total innings pitched
        games_started: Number of games started (if available)

    Returns:
        dict: {'role': 'starter'|'reliever', 'confidence': float}
    """
    if games_pitched <= 0:
        return {'role': 'unknown', 'confidence': 0.0}

    # Calculate average innings per appearance
    ip_per_game = innings_pitched / games_pitched

    # Use games started if available
    if games_started is not None and games_started > 0:
        start_percentage = games_started / games_pitched
        if start_percentage >= 0.7:  # 70%+ starts = starter
            return {'role': 'starter', 'confidence': min(0.95, 0.5 + start_percentage)}
        elif start_percentage <= 0.1:  # 10% or less starts = reliever
            return {'role': 'reliever', 'confidence': min(0.95, 0.9 - start_percentage)}

    # Fallback to innings per game analysis
    if ip_per_game >= 4.0:  # 4+ IP per game = likely starter
        confidence = min(0.9, ip_per_game / 6.0)  # Max confidence at 6 IP/G
        return {'role': 'starter', 'confidence': confidence}
    elif ip_per_game <= 1.5:  # 1.5 IP or less = likely reliever
        confidence = min(0.9, (1.5 - ip_per_game) / 1.5 + 0.5)
        return {'role': 'reliever', 'confidence': confidence}
    else:
        # Ambiguous - could be swingman/opener
        return {'role': 'swing', 'confidence': 0.3}


def calculate_pitcher_remaining_workload(current_games, current_ip, role_classification):
    """
    Calculate realistic remaining workload for pitcher based on role

    Args:
        current_games: Games pitched so far
        current_ip: Innings pitched so far
        role_classification: Result from classify_pitcher_role()

    Returns:
        dict: {
            'remaining_games': int,
            'remaining_ip': float,
            'total_season_games': int,
            'total_season_ip': float,
            'projection_basis': str
        }
    """
    role = role_classification['role']
    confidence = role_classification['confidence']

    if role == 'starter':
        # Starters: ~30-34 games, 180-220 IP per season
        # Account for 5-6 man rotation (162 games / 5 = ~32 starts)
        projected_total_games = 32
        projected_total_ip = 200  # Conservative starter target

        # Adjust based on current pace if significantly different
        if current_games > 0:
            current_pace_games = (current_games / current_games) * 32  # Linear projection
            current_pace_ip = (current_ip / current_games) * projected_total_games

            # Blend current pace with typical starter expectations
            blend_factor = min(0.7, confidence)  # Higher confidence = more typical
            projected_total_games = int(
                blend_factor * projected_total_games +
                (1 - blend_factor) * min(35, current_pace_games)
            )
            projected_total_ip = (
                blend_factor * projected_total_ip +
                (1 - blend_factor) * min(230, current_pace_ip)
            )

        remaining_games = max(0, projected_total_games - current_games)
        remaining_ip = max(0, projected_total_ip - current_ip)

        return {
            'remaining_games': remaining_games,
            'remaining_ip': remaining_ip,
            'total_season_games': projected_total_games,
            'total_season_ip': projected_total_ip,
            'projection_basis': f'Starter rotation (conf: {confidence:.2f})'
        }

    elif role == 'reliever':
        # Relievers: ~50-70 games, 60-80 IP per season
        projected_total_games = 60
        projected_total_ip = 70

        # Adjust based on current usage pattern
        if current_games > 0:
            ip_per_game = current_ip / current_games

            # High-leverage closer: fewer games, more IP per game
            if ip_per_game >= 1.0:
                projected_total_games = 55
                projected_total_ip = 65
            # Setup/middle relief: more games, standard IP
            else:
                projected_total_games = 65
                projected_total_ip = 70

        remaining_games = max(0, projected_total_games - current_games)
        remaining_ip = max(0, projected_total_ip - current_ip)

        return {
            'remaining_games': remaining_games,
            'remaining_ip': remaining_ip,
            'total_season_games': projected_total_games,
            'total_season_ip': projected_total_ip,
            'projection_basis': f'Reliever usage (conf: {confidence:.2f})'
        }

    else:  # swing/unknown role
        # Conservative projection - assume flexible role
        projected_total_games = 40
        projected_total_ip = 120

        remaining_games = max(0, projected_total_games - current_games)
        remaining_ip = max(0, projected_total_ip - current_ip)

        return {
            'remaining_games': remaining_games,
            'remaining_ip': remaining_ip,
            'total_season_games': projected_total_games,
            'total_season_ip': projected_total_ip,
            'projection_basis': f'Swing role/Unknown (conf: {confidence:.2f})'
        }


def calculate_pitcher_projections(player_data, ensemble_predictor, player_feature_vector, total_remaining_games=None):
    """
    Calculate pitcher projections using realistic workload expectations

    Args:
        player_data: DataFrame row with pitcher data
        ensemble_predictor: Trained ensemble model
        player_feature_vector: Feature vector for predictions
        total_remaining_games: Optional constraint for two-way players or late-season callups

    Returns:
        dict: Complete projection data with realistic workload
    """
    # Extract current stats
    current_games = player_data.get('G', 0)
    current_ip = player_data.get('IP', 0.0)
    current_gs = player_data.get('GS', None)  # Games started if available

    # Classify pitcher role
    role_classification = classify_pitcher_role(current_games, current_ip, current_gs)

    # Calculate realistic remaining workload
    workload_projection = calculate_pitcher_remaining_workload(
        current_games, current_ip, role_classification
    )

    # Apply game constraint for two-way players or late-season callups
    if total_remaining_games is not None:
        original_remaining = workload_projection['remaining_games']
        constrained_remaining = min(workload_projection['remaining_games'], total_remaining_games)

        if constrained_remaining != original_remaining:
            # Adjust IP proportionally when games are constrained
            ip_ratio = constrained_remaining / original_remaining if original_remaining > 0 else 0
            workload_projection['remaining_games'] = constrained_remaining
            workload_projection['remaining_ip'] = workload_projection['remaining_ip'] * ip_ratio
            workload_projection['total_season_games'] = current_games + constrained_remaining
            workload_projection['total_season_ip'] = current_ip + workload_projection['remaining_ip']
            workload_projection['projection_basis'] += f' (constrained by {total_remaining_games} remaining team games)'

    # Calculate current performance using ensemble
    current_war = ensemble_predictor.predict_ensemble(player_feature_vector, 'war', 'pitcher')['ensemble']
    current_warp = ensemble_predictor.predict_ensemble(player_feature_vector, 'warp', 'pitcher')['ensemble']

    # Calculate per-game and per-inning rates
    war_per_game = current_war / current_games if current_games > 0 else 0
    warp_per_game = current_warp / current_games if current_games > 0 else 0
    war_per_ip = current_war / current_ip if current_ip > 0 else 0
    warp_per_ip = current_warp / current_ip if current_ip > 0 else 0

    # Define projection scenarios (enhanced with upside scenarios)
    scenarios = {
        '150% (Hot Streak)': 1.5,
        '125% (Above Pace)': 1.25,
        '100% (Maintain Pace)': 1.0,
        '75% (Slight Regression)': 0.75,
        '50% (Major Regression)': 0.50,
        '25% (Horrible Regression)': 0.25,
        'Career Average': 0.60
    }

    projection_results = {}

    for scenario_name, multiplier in scenarios.items():
        # Use games-based projection (more stable for pitchers)
        remaining_war = war_per_game * multiplier * workload_projection['remaining_games']
        remaining_warp = warp_per_game * multiplier * workload_projection['remaining_games']

        full_season_war = current_war + remaining_war
        full_season_warp = current_warp + remaining_warp

        projection_results[scenario_name] = {
            'remaining_war': remaining_war,
            'remaining_warp': remaining_warp,
            'full_season_war': full_season_war,
            'full_season_warp': full_season_warp
        }

    return {
        'current_war': current_war,
        'current_warp': current_warp,
        'current_games': current_games,
        'current_ip': current_ip,
        'role_classification': role_classification,
        'workload_projection': workload_projection,
        'projections': projection_results,
        'rates': {
            'war_per_game': war_per_game,
            'warp_per_game': warp_per_game,
            'war_per_ip': war_per_ip,
            'warp_per_ip': warp_per_ip
        }
    }