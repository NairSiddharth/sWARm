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


def calculate_pitcher_remaining_workload(current_games, current_ip, role_classification,
                                        team_games_played=None, team_games_total=162):
    """
    Calculate realistic remaining workload for pitcher based on role.

    Now supports actual team games for accurate participation-based projections.

    Args:
        current_games: Games pitched so far
        current_ip: Innings pitched so far
        role_classification: Result from classify_pitcher_role()
        team_games_played: Actual games team has played (for participation rate)
        team_games_total: Total games in season (default 162)

    Returns:
        dict: {
            'remaining_games': int,
            'remaining_ip': float,
            'total_season_games': int,
            'total_season_ip': float,
            'projection_basis': str,
            'participation_rate': float (if team games provided)
        }
    """
    role = role_classification['role']
    confidence = role_classification['confidence']

    if role == 'starter':
        # Starters: ~30-34 games, 180-220 IP per season
        # Account for 5-6 man rotation (162 games / 5 = ~32 starts)
        projected_total_games = 32
        projected_total_ip = 200  # Conservative starter target

        # Use participation rate if team games available
        if team_games_played and current_games > 0:
            # Calculate actual participation rate
            participation_rate = current_games / team_games_played
            remaining_team_games = team_games_total - team_games_played

            # Project based on participation rate (accounting for 5-man rotation)
            # Max ~32 starts for starters even with perfect attendance
            max_starts_remaining = remaining_team_games / 5  # 5-man rotation
            projected_remaining = min(participation_rate * remaining_team_games, max_starts_remaining)

            projected_total_games = current_games + int(projected_remaining)

            # Calculate IP based on current average
            ip_per_game = current_ip / current_games
            projected_total_ip = projected_total_games * ip_per_game
        elif current_games > 0:
            # Fallback to simple pace calculation without team games
            ip_per_game = current_ip / current_games

            # For starters, typical is 32 games, but adjust based on current usage
            # No need to blend - just use typical starter values
            projected_total_games = 32
            projected_total_ip = ip_per_game * projected_total_games
        else:
            # No games played yet - use defaults
            projected_total_games = 32
            projected_total_ip = 200

        remaining_games = max(0, projected_total_games - current_games)
        remaining_ip = max(0, projected_total_ip - current_ip)

        result = {
            'remaining_games': remaining_games,
            'remaining_ip': remaining_ip,
            'total_season_games': projected_total_games,
            'total_season_ip': projected_total_ip,
            'projection_basis': f'Starter rotation (conf: {confidence:.2f})'
        }

        # Add participation rate if calculated
        if team_games_played and current_games > 0:
            result['participation_rate'] = current_games / team_games_played

        return result

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


def calculate_pitcher_projections(
        player_data,
        ensemble_predictor,
        player_feature_vector,
        total_remaining_games=None,
        team_games_dict=None,
        hitter_df=None,
        pitcher_df=None):
    """
    Calculate pitcher projections using realistic workload expectations.

    Now supports:
    - Actual team games for participation-based projections
    - Automatic two-way player detection and constraints

    Args:
        player_data: DataFrame row with pitcher data
        ensemble_predictor: Trained ensemble model
        player_feature_vector: Feature vector for predictions
        total_remaining_games: Optional manual constraint
        team_games_dict: Dictionary mapping team to actual games played
        hitter_df: Full hitter DataFrame for two-way player detection
        pitcher_df: Full pitcher DataFrame for two-way player detection

    Returns:
        dict: Complete projection data with realistic workload
    """
    # Extract current stats
    current_games = player_data.get('G', 0)
    current_ip = player_data.get('IP', 0.0)
    current_gs = player_data.get('GS', None)  # Games started if available

    # Get team games - load automatically if not provided
    team_games_played = None
    if team_games_dict is None:
        # Try to load team games for current season
        try:
            from current_season_modules.current_season_data_loading import calculate_team_games_from_hitters
            team_games_dict = calculate_team_games_from_hitters(2025, 'fangraphs')
        except Exception:
            # If loading fails, team_games_dict remains None
            pass

    if team_games_dict:
        team = player_data.get('Team', player_data.get('team', None))
        if team and team in team_games_dict:
            team_games_played = team_games_dict[team]

    # Classify pitcher role
    role_classification = classify_pitcher_role(current_games, current_ip, current_gs)

    # Calculate realistic remaining workload with team games if available
    workload_projection = calculate_pitcher_remaining_workload(
        current_games, current_ip, role_classification,
        team_games_played=team_games_played
    )

    # Check for two-way player constraints
    if hitter_df is not None and pitcher_df is not None:
        from common_modules.two_way_player_handler import apply_two_way_constraints_to_projections

        original_remaining = workload_projection['remaining_games']
        constrained_remaining = apply_two_way_constraints_to_projections(
            player_data,
            'pitcher',
            original_remaining,
            hitter_df,
            pitcher_df
        )

        if constrained_remaining != original_remaining:
            # Adjust for two-way player
            ip_ratio = constrained_remaining / original_remaining if original_remaining > 0 else 0
            workload_projection['remaining_games'] = constrained_remaining
            workload_projection['remaining_ip'] = workload_projection['remaining_ip'] * ip_ratio
            workload_projection['total_season_games'] = current_games + constrained_remaining
            workload_projection['total_season_ip'] = current_ip + workload_projection['remaining_ip']
            workload_projection['projection_basis'] += ' (two-way player adjusted)'

    # Apply manual constraint if provided (for late-season callups, etc.)
    elif total_remaining_games is not None:
        original_remaining = workload_projection['remaining_games']
        constrained_remaining = min(workload_projection['remaining_games'], total_remaining_games)

        if constrained_remaining != original_remaining:
            # Adjust IP proportionally when games are constrained
            ip_ratio = constrained_remaining / original_remaining if original_remaining > 0 else 0
            workload_projection['remaining_games'] = constrained_remaining
            workload_projection['remaining_ip'] = workload_projection['remaining_ip'] * ip_ratio
            workload_projection['total_season_games'] = current_games + constrained_remaining
            workload_projection['total_season_ip'] = current_ip + workload_projection['remaining_ip']
            workload_projection['projection_basis'] += f' (manually constrained to {total_remaining_games} games)'

    # Calculate current performance using ensemble
    print(
        f"DEBUG: Feature vector for {
            player_data.get(
                'Name',
                'Unknown')}: {player_feature_vector}")
    print(f"DEBUG: Feature vector shape: {np.array(player_feature_vector).shape}")
    print(f"DEBUG: Feature vector type: {type(player_feature_vector)}")

    # Check if this is a Phase 1 pitcher ensemble (rate-based three-path)
    if hasattr(ensemble_predictor, 'pitcher_ensemble') and ensemble_predictor.pitcher_ensemble:
        # Use Phase 1 interface
        print("DEBUG: Using Phase 1 rate-based prediction interface")

        # Phase 1 expects features WITHOUT IP (IP is first feature, so skip it)
        features_no_ip = player_feature_vector[1:] if len(player_feature_vector) > 1 else player_feature_vector

        # Get G and GS from player_data
        G = player_data.get('G', current_games)
        GS = player_data.get('GS', 0)

        # Predict current performance using Phase 1
        war_pred = ensemble_predictor.pitcher_ensemble.predict(
            features=features_no_ip,
            GS=int(GS),
            G=int(G),
            IP=float(current_ip),
            metric_type='war'
        )
        warp_pred = ensemble_predictor.pitcher_ensemble.predict(
            features=features_no_ip,
            GS=int(GS),
            G=int(G),
            IP=float(current_ip),
            metric_type='warp'
        )

        current_war = war_pred['current_war']
        current_warp = warp_pred['current_war']  # Phase 1 calls it 'current_war' for both metrics

        # Also get projected WAR if available (for future use)
        phase1_projected_war = war_pred.get('projected_war', None)
        phase1_projected_warp = warp_pred.get('projected_war', None)
        phase1_role = war_pred['role']

        print(f"DEBUG: Phase 1 predicted WAR: {current_war:.3f}, WARP: {current_warp:.3f}")
        print(f"DEBUG: Phase 1 classified role: {phase1_role}")
    else:
        # Use original interface (backward compatibility)
        current_war = ensemble_predictor.predict_ensemble(
            player_feature_vector, 'war', 'pitcher')['ensemble']
        current_warp = ensemble_predictor.predict_ensemble(
            player_feature_vector, 'warp', 'pitcher')['ensemble']

        print(f"DEBUG: Predicted WAR: {current_war:.3f}, WARP: {current_warp:.3f}")

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
