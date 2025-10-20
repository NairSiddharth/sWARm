"""
Table formatting utilities for oWAR notebooks.

Provides PrettyTable formatting for:
- Featured player tables
- Model metrics comparison tables
- Player type/position classification
- Ranking within groups
- Two-way player WAR handling
"""

from typing import Dict, Optional, List
import pandas as pd
from prettytable import PrettyTable, MARKDOWN

from new_pipeline.common.constants import (
    COL_MLBAMID,
    COL_NAME,
    PITCHER_STARTER_THRESHOLD,
    PITCHER_RELIEVER_THRESHOLD,
    HITTER_PRIMARY_POSITION_THRESHOLD,
    HITTER_DUAL_POSITION_THRESHOLD
)


def create_featured_table(
    df: pd.DataFrame,
    player_names: List[str],
    player_type: str,
    two_way_data: Optional[Dict] = None
) -> PrettyTable:
    """
    Create PrettyTable for featured players.

    Args:
        df: Full predictions data
        player_names: Players to feature
        player_type: 'pitcher' or 'hitter'
        two_way_data: Two-way player combined WAR (optional)
                     Format: {player_name: {'current_WAR': float, 'ROS_WAR': float, 'total_proj': float}}

    Returns:
        prettytable.PrettyTable: Formatted table with markdown style

    Features:
        - Dividers between types/positions
        - Rank within type/position
        - Two-way player support

    Example:
        >>> table = create_featured_table(
        ...     df=pitcher_predictions,
        ...     player_names=['Tarik Skubal', 'Zack Wheeler'],
        ...     player_type='pitcher',
        ...     two_way_data={'Shohei Ohtani': {'current_WAR': 11.7, ...}}
        ... )
        >>> print(table)
    """
    if player_type not in ['pitcher', 'hitter']:
        raise ValueError(f"player_type must be 'pitcher' or 'hitter', got: {player_type}")

    # Create table with appropriate columns
    if player_type == 'pitcher':
        table = PrettyTable()
        table.field_names = ['Type Rank', 'Overall', 'Name', 'Type', 'Team', 'Current WAR', 'ROS WAR', 'Total Proj']
    else:
        table = PrettyTable()
        table.field_names = ['Pos Rank', 'Overall', 'Name', 'Pos', 'Team', 'Current WAR', 'ROS WAR', 'Total Proj']

    table.set_style(MARKDOWN)
    table.align = 'l'
    table.align['Current WAR'] = 'r'
    table.align['ROS WAR'] = 'r'
    table.align['Total Proj'] = 'r'

    # Right-align both rank columns
    if player_type == 'pitcher':
        table.align['Type Rank'] = 'r'
    else:
        table.align['Pos Rank'] = 'r'
    table.align['Overall'] = 'r'

    # Add rows for each featured player
    for player_name in player_names:
        # Check if two-way player
        if two_way_data and player_name in two_way_data:
            # Use combined two-way data
            two_way_info = two_way_data[player_name]
            player_row = df[df[COL_NAME] == player_name].iloc[0] if player_name in df[COL_NAME].values else None

            if player_row is not None:
                # Determine position for ranking
                # For hitters: rank as DH; for pitchers: use actual pitcher type
                if player_type == 'hitter':
                    position_for_ranking = 'DH'
                else:
                    # For pitchers: get their actual type (Starter/Reliever/Swing)
                    gs_per_g = player_row.get('GS_per_G', player_row.get('GS', 0) / max(player_row.get('G', 1), 1))
                    position_for_ranking = get_pitcher_type(gs_per_g)

                type_rank = get_rank_within_type(
                    player_name,
                    position_for_ranking,
                    df,
                    player_type
                )
                overall_rank = get_overall_rank(player_name, df)

                table.add_row([
                    type_rank,
                    overall_rank,
                    player_name,
                    'Two-Way',
                    player_row.get('Team', 'N/A'),
                    f"{two_way_info.get('current_WAR', 0.0):.1f}",
                    f"{two_way_info.get('ROS_WAR', 0.0):.1f}",
                    f"{two_way_info.get('total_proj', 0.0):.1f}"
                ])
        else:
            # Regular player
            player_row = df[df[COL_NAME] == player_name]

            if len(player_row) == 0:
                continue  # Skip if player not found

            player_row = player_row.iloc[0]

            # Get player type/position
            if player_type == 'pitcher':
                gs_per_g = player_row.get('GS_per_G', player_row.get('GS', 0) / max(player_row.get('G', 1), 1))
                player_type_label = get_pitcher_type(gs_per_g)
            else:
                player_type_label = player_row.get('Primary_Position', 'Unknown')

            # Get rank within type and overall rank
            type_rank = get_rank_within_type(
                player_name,
                player_type_label,
                df,
                player_type
            )
            overall_rank = get_overall_rank(player_name, df)

            # Get WAR values
            current_war = player_row.get('Current_WAR', player_row.get('WAR', 0.0))
            ros_war = player_row.get('ROS_WAR', 0.0)
            total_proj = player_row.get('Total_Projected_WAR', current_war + ros_war)

            table.add_row([
                type_rank,
                overall_rank,
                player_name,
                player_type_label,
                player_row.get('Team', 'N/A'),
                f"{current_war:.1f}",
                f"{ros_war:.1f}",
                f"{total_proj:.1f}"
            ])

    return table


def create_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> PrettyTable:
    """
    Create comparison table for model metrics.

    Args:
        metrics_dict: {group: {metric: value}}

    Returns:
        prettytable.PrettyTable: Metrics comparison table

    Example:
        >>> metrics = {
        ...     'Starter': {'MAE': 0.52, 'RMSE': 0.71, 'R²': 0.83},
        ...     'Reliever': {'MAE': 0.48, 'RMSE': 0.65, 'R²': 0.79}
        ... }
        >>> table = create_metrics_table(metrics)
    """
    # Create table
    table = PrettyTable()

    # Get all unique metrics
    all_metrics = set()
    for group_metrics in metrics_dict.values():
        all_metrics.update(group_metrics.keys())

    all_metrics = sorted(list(all_metrics))

    # Set field names
    table.field_names = ['Group'] + all_metrics
    table.set_style(MARKDOWN)
    table.align = 'l'

    # Right-align numeric columns
    for metric in all_metrics:
        table.align[metric] = 'r'

    # Add rows
    for group, metrics in metrics_dict.items():
        row = [group]
        for metric in all_metrics:
            value = metrics.get(metric, 0.0)
            # Format based on metric type
            if metric == 'R²':
                row.append(f"{value:.3f}")
            else:
                row.append(f"{value:.2f}")
        table.add_row(row)

    return table


def get_pitcher_type(gs_per_g: float) -> str:
    """
    Classify pitcher by role.

    Args:
        gs_per_g: Games started per game played

    Returns:
        str: 'Starter', 'Reliever', or 'Swing'

    Example:
        >>> get_pitcher_type(0.95)
        'Starter'
        >>> get_pitcher_type(0.05)
        'Reliever'
        >>> get_pitcher_type(0.50)
        'Swing'
    """
    if gs_per_g > PITCHER_STARTER_THRESHOLD:
        return 'Starter'
    elif gs_per_g < PITCHER_RELIEVER_THRESHOLD:
        return 'Reliever'
    else:
        return 'Swing'


def get_hitter_position(position_dict: Dict[str, float]) -> str:
    """
    Classify hitter position.

    Args:
        position_dict: {position: percentage} mapping

    Returns:
        str: Primary position, dual position (e.g., 'SS/2B'), or 'Utility'

    Logic:
        - >90% at one position -> Primary
        - Two positions >8% each -> Dual (e.g., 'OF/1B')
        - Otherwise -> 'Utility'

    Example:
        >>> get_hitter_position({'SS': 0.95})
        'SS'
        >>> get_hitter_position({'OF': 0.55, '1B': 0.45})
        'OF/1B'
        >>> get_hitter_position({'2B': 0.3, 'SS': 0.3, '3B': 0.4})
        'Utility'
    """
    if not position_dict:
        return 'Utility'

    # Sort by percentage descending
    sorted_positions = sorted(position_dict.items(), key=lambda x: x[1], reverse=True)

    # Primary position
    if sorted_positions[0][1] > HITTER_PRIMARY_POSITION_THRESHOLD:
        return sorted_positions[0][0]

    # Count positions with significant playing time
    significant_positions = [pos for pos, pct in sorted_positions if pct > HITTER_DUAL_POSITION_THRESHOLD]

    # Dual position (exactly 2 positions >8%)
    if len(significant_positions) == 2:
        return f"{sorted_positions[0][0]}/{sorted_positions[1][0]}"

    # Otherwise utility (0, 1, or 3+ positions >8%)
    return 'Utility'


def get_rank_within_type(
    player_name: str,
    player_type: str,
    full_df: pd.DataFrame,
    category: str
) -> str:
    """
    Calculate player's rank within their type/position.

    Args:
        player_name: Player to rank
        player_type: Type/position group
        full_df: All players data
        category: 'pitcher' or 'hitter'

    Returns:
        str: Rank format "3/342" (rank/total)

    Example:
        >>> get_rank_within_type('Tarik Skubal', 'Starter', df_all, 'pitcher')
        '1/342'
    """
    if category == 'pitcher':
        # Classify all pitchers by type
        def classify_pitcher(row):
            gs_per_g = row.get('GS_per_G', row.get('GS', 0) / max(row.get('G', 1), 1))
            return get_pitcher_type(gs_per_g)

        full_df = full_df.copy()
        full_df['PlayerType'] = full_df.apply(classify_pitcher, axis=1)
    else:
        # Use Primary_Position for hitters
        full_df = full_df.copy()
        if 'Primary_Position' in full_df.columns:
            full_df['PlayerType'] = full_df['Primary_Position'].fillna('Unknown')
        else:
            full_df['PlayerType'] = 'Unknown'

    # Filter to same type
    same_type_df = full_df[full_df['PlayerType'] == player_type]

    # Get WAR column
    war_col = 'Total_Projected_WAR' if 'Total_Projected_WAR' in same_type_df.columns else 'WAR'

    # Sort by WAR descending
    same_type_df = same_type_df.sort_values(war_col, ascending=False)

    # Find player rank
    player_rows = same_type_df[same_type_df[COL_NAME] == player_name]

    if len(player_rows) == 0:
        return "N/A"

    rank = list(same_type_df[COL_NAME]).index(player_name) + 1
    total = len(same_type_df)

    return f"{rank}/{total}"


def get_overall_rank(
    player_name: str,
    full_df: pd.DataFrame
) -> str:
    """
    Calculate player's rank across ALL players of the same category.

    Args:
        player_name: Player to rank
        full_df: All players data

    Returns:
        str: Rank format "3/510" (rank/total)

    Example:
        >>> get_overall_rank('Bobby Witt Jr.', df_all_hitters)
        '3/510'
    """
    # Get WAR column
    war_col = 'Total_Projected_WAR' if 'Total_Projected_WAR' in full_df.columns else 'WAR'

    # Sort by WAR descending
    sorted_df = full_df.sort_values(war_col, ascending=False)

    # Find player rank
    player_rows = sorted_df[sorted_df[COL_NAME] == player_name]

    if len(player_rows) == 0:
        return "N/A"

    rank = list(sorted_df[COL_NAME]).index(player_name) + 1
    total = len(sorted_df)

    return f"{rank}/{total}"


def handle_two_way_player(
    pitcher_war: Dict[str, float],
    hitter_war: Dict[str, float]
) -> Dict[str, float]:
    """
    Combine WAR for two-way players.

    Args:
        pitcher_war: {'current': float, 'ROS': float, 'total': float}
        hitter_war: {'current': float, 'ROS': float, 'total': float}

    Returns:
        dict: Combined WAR values

    Example:
        >>> combined = handle_two_way_player(
        ...     pitcher_war={'current': 0.8, 'ROS': 0.4, 'total': 1.2},
        ...     hitter_war={'current': 7.7, 'ROS': 2.8, 'total': 10.5}
        ... )
        >>> # Returns: {'current_WAR': 8.5, 'ROS_WAR': 3.2, 'total_proj': 11.7}
    """
    return {
        'current_WAR': pitcher_war.get('current', 0.0) + hitter_war.get('current', 0.0),
        'ROS_WAR': pitcher_war.get('ROS', 0.0) + hitter_war.get('ROS', 0.0),
        'total_proj': pitcher_war.get('total', 0.0) + hitter_war.get('total', 0.0)
    }


def create_ros_diagnostic_table(
    names: List[str],
    teams: List[str],
    tiers: List[str],
    usage_current: List[float],
    usage_projected: List[float],
    ros_display: Dict[str, List[float]],
    player_type: str,
    role: str = None,
    additional_cols: Dict[str, List[float]] = None
) -> PrettyTable:
    """
    Create formatted diagnostic table for ROS predictions.

    Args:
        names: Player names
        teams: Teams
        tiers: Tier classifications ('average', 'good', 'elite')
        usage_current: Current IP (pitchers) or PA (hitters)
        usage_projected: Projected remaining IP or PA
        ros_display: Dict from format_pitcher_ros_display/format_hitter_ros_display
                    Expected keys: 'ros_war', 'ros_rate', 'ros_q50', 'ros_q90'
        player_type: 'pitcher' or 'hitter'
        role: 'starter', 'reliever', 'swing' (pitchers only)
        additional_cols: Optional dict of {column_name: values} for role-specific columns

    Returns:
        PrettyTable with proper formatting and markdown style

    Example:
        >>> # Starter example
        >>> table = create_ros_diagnostic_table(
        ...     names=['Tarik Skubal', 'Zack Wheeler'],
        ...     teams=['DET', 'PHI'],
        ...     tiers=['elite', 'good'],
        ...     usage_current=[130.0, 125.0],
        ...     usage_projected=[64.0, 68.0],
        ...     ros_display={'ros_war': [2.9, 2.7], 'ros_rate': [4.52, 3.97], ...},
        ...     player_type='pitcher',
        ...     role='starter',
        ...     additional_cols={'Starts': [21, 20], 'IP/Start': [6.2, 6.3], 'TeamG': [95, 96]}
        ... )
        >>> print(table)
    """
    if player_type not in ['pitcher', 'hitter']:
        raise ValueError(f"player_type must be 'pitcher' or 'hitter', got: {player_type}")

    # Create table
    table = PrettyTable()
    table.set_style(MARKDOWN)
    table.align = 'l'

    # Build field names based on player type and role
    base_fields = ['Name', 'Team', 'Tier']

    if player_type == 'pitcher':
        usage_label = 'IP'
        usage_proj_label = 'Proj_IP'

        if role == 'starter':
            # Starter-specific columns
            field_names = base_fields + [usage_label]
            if additional_cols and 'Starts' in additional_cols:
                field_names.append('Starts')
            if additional_cols and 'IP/Start' in additional_cols:
                field_names.append('IP/Start')
            if additional_cols and 'TeamG' in additional_cols:
                field_names.append('TeamG')
            field_names += [usage_proj_label, 'ROS_Rate', 'ROS_WAR', 'ROS_q50', 'ROS_q90']
        else:
            # Reliever/Swing columns
            field_names = base_fields + [usage_label]
            if additional_cols and 'G' in additional_cols:
                field_names.append('G')
            if additional_cols and 'IP/G' in additional_cols:
                field_names.append('IP/G')
            if additional_cols and 'App_Rate' in additional_cols:
                field_names.append('App_Rate')
            if additional_cols and 'TeamG' in additional_cols:
                field_names.append('TeamG')
            field_names += [usage_proj_label, 'ROS_Rate', 'ROS_WAR', 'ROS_q50', 'ROS_q90']
    else:
        # Hitter columns
        usage_label = 'PA'
        usage_proj_label = 'Remaining_PA'
        field_names = base_fields + [usage_label]
        if additional_cols and 'G' in additional_cols:
            field_names.append('G')
        if additional_cols and 'TeamG' in additional_cols:
            field_names.append('TeamG')
        field_names += [usage_proj_label, 'Total_PA_Proj', 'ROS_Rate', 'ROS_WAR', 'ROS_q50', 'ROS_q90']

    table.field_names = field_names

    # Right-align numeric columns
    numeric_cols = [usage_label, usage_proj_label, 'ROS_Rate', 'ROS_WAR', 'ROS_q50', 'ROS_q90']
    if 'Total_PA_Proj' in field_names:
        numeric_cols.append('Total_PA_Proj')
    if 'Starts' in field_names:
        numeric_cols.append('Starts')
    if 'G' in field_names:
        numeric_cols.append('G')
    if 'TeamG' in field_names:
        numeric_cols.append('TeamG')
    if 'IP/Start' in field_names:
        numeric_cols.append('IP/Start')
    if 'IP/G' in field_names:
        numeric_cols.append('IP/G')
    if 'App_Rate' in field_names:
        numeric_cols.append('App_Rate')

    for col in numeric_cols:
        if col in field_names:
            table.align[col] = 'r'

    # Add rows
    for i in range(len(names)):
        row = [names[i], teams[i], tiers[i]]

        # Add usage current
        row.append(f"{usage_current[i]:.0f}")

        # Add role-specific columns
        if additional_cols:
            if 'Starts' in field_names and 'Starts' in additional_cols:
                row.append(f"{additional_cols['Starts'][i]:.0f}")
            if 'IP/Start' in field_names and 'IP/Start' in additional_cols:
                row.append(f"{additional_cols['IP/Start'][i]:.1f}")
            if 'G' in field_names and 'G' in additional_cols:
                row.append(f"{additional_cols['G'][i]:.0f}")
            if 'IP/G' in field_names and 'IP/G' in additional_cols:
                row.append(f"{additional_cols['IP/G'][i]:.2f}")
            if 'App_Rate' in field_names and 'App_Rate' in additional_cols:
                row.append(f"{additional_cols['App_Rate'][i]:.2f}")
            if 'TeamG' in field_names and 'TeamG' in additional_cols:
                row.append(f"{additional_cols['TeamG'][i]:.0f}")

        # Add projected usage
        row.append(f"{usage_projected[i]:.0f}")

        # Add total projected PA for hitters
        if 'Total_PA_Proj' in field_names:
            total_proj = usage_current[i] + usage_projected[i]
            row.append(f"{total_proj:.0f}")

        # Add ROS predictions
        row.append(f"{ros_display['ros_rate'][i]:.2f}")
        row.append(f"{ros_display['ros_war'][i]:.1f}")
        row.append(f"{ros_display['ros_q50'][i]:.1f}")
        row.append(f"{ros_display['ros_q90'][i]:.1f}")

        table.add_row(row)

    return table
