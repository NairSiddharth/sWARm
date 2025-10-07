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
        table.field_names = ['Rank', 'Name', 'Type', 'Team', 'Current WAR', 'ROS WAR', 'Total Proj']
    else:
        table = PrettyTable()
        table.field_names = ['Rank', 'Name', 'Pos', 'Team', 'Current WAR', 'ROS WAR', 'Total Proj']

    table.set_style(MARKDOWN)
    table.align = 'l'
    table.align['Current WAR'] = 'r'
    table.align['ROS WAR'] = 'r'
    table.align['Total Proj'] = 'r'
    table.align['Rank'] = 'r'

    # Add rows for each featured player
    for player_name in player_names:
        # Check if two-way player
        if two_way_data and player_name in two_way_data:
            # Use combined two-way data
            two_way_info = two_way_data[player_name]
            player_row = df[df[COL_NAME] == player_name].iloc[0] if player_name in df[COL_NAME].values else None

            if player_row is not None:
                rank = get_rank_within_type(
                    player_name,
                    'Two-Way',
                    df,
                    player_type
                )

                table.add_row([
                    rank,
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

            # Get rank within type
            rank = get_rank_within_type(
                player_name,
                player_type_label,
                df,
                player_type
            )

            # Get WAR values
            current_war = player_row.get('Current_WAR', player_row.get('WAR', 0.0))
            ros_war = player_row.get('ROS_WAR', 0.0)
            total_proj = player_row.get('Total_Projected_WAR', current_war + ros_war)

            table.add_row([
                rank,
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
        full_df['PlayerType'] = full_df.get('Primary_Position', 'Unknown')

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
