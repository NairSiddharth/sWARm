# Player Game Data Parser
import os
import pandas as pd
import re

# ====== PATH CONFIG ======
DATA_DIR = r"C:\Users\nairs\Documents\GithubProjects\oWAR\MLB Player Data"

# ====== REGEX ======
capitalized_words = r"((?:[A-Z][a-z']+ ?)+)"  # regex to get capitalized words in sentence

# ====== LOAD DATA ======
hitter_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "hittersByGame(player_offense_data).csv"))
pitcher_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "pitchersByGame(pitcher_data).csv"))
baserunning_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "baserunningNotes(player_offense_data).csv"))
fielding_by_game_df = pd.read_csv(os.path.join(DATA_DIR, "fieldingNotes(player_defensive_data).csv"))

warp_hitter_df = pd.read_csv(os.path.join(DATA_DIR, "bp_hitters_2021.csv"))
warp_pitcher_df = pd.read_csv(os.path.join(DATA_DIR, "bp_pitchers_2021.csv"))
oaa_hitter_df = pd.read_csv(os.path.join(DATA_DIR, "outs_above_average.csv"))

fielding_df = pd.read_csv(os.path.join(DATA_DIR, "fieldingNotes(player_defensive_data).csv"))
baserunning_df = pd.read_csv(os.path.join(DATA_DIR, "baserunningNotes(player_offense_data).csv"))
war_df = pd.read_csv(os.path.join(DATA_DIR, "FanGraphs Leaderboard.csv"))

# ====== CLEANERS ======

def clean_sorted_hitter():
    df = hitter_by_game_df.drop(['H-AB', 'AB', 'H', '#P', 'Game', 'Team', 'Hitter Id'], axis=1)
    return df.sort_values(by='Hitters')

def clean_sorted_pitcher():
    df = pitcher_by_game_df.drop(['R', 'ER', 'PC', 'Game', 'Team', 'Extra', 'Pitcher Id'], axis=1)
    return df.sort_values(by='Pitchers')

def clean_warp_hitter():
    df = warp_hitter_df.drop(['bpid', 'mlbid', 'Age', 'DRC+', '+/-', 'PA', 'R', 'RBI',
                              'ISO', 'K%', 'BB%', 'Whiff%'], axis=1)
    return df.sort_values(by='WARP')

def clean_warp_pitcher():
    df = warp_pitcher_df.drop(['bpid', 'mlbid', 'DRA-', 'DRA', 'DRA SD', 'cFIP',
                               'GS', 'W', 'L', 'ERA', 'RA9', 'Whiff%'], axis=1)
    return df.sort_values(by='WARP')

def clean_war():
    df = war_df.drop(['playerid', 'Team', 'Pos'], axis=1)
    return df.sort_values(by='Total WAR')

def clean_sorted_baserunning():
    df = baserunning_by_game_df.drop(['Game'], axis=1)
    sorted_df = df.sort_values(by='Stat')

    baserunning_values = {}

    for _, row in sorted_df.iterrows():
        statlines = str(row['Stat']).split(',')
        if not statlines:
            continue

        if statlines[0] == 'SB':
            players = re.findall(capitalized_words, str(row.get('Play', '')))
            for p in players:
                baserunning_values[p] = baserunning_values.get(p, 0) + (1 / 3)

        elif statlines[0] in ['CS', 'Picked Off']:
            players = re.findall(capitalized_words, str(row.get('Play', '')))
            for p in players:
                baserunning_values[p] = baserunning_values.get(p, 0) - (1 / 3)

    return baserunning_values

def clean_defensive_players():
    df = fielding_df.drop(['Game', 'Team'], axis=1)
    sorted_df = df.sort_values(by='Stat')

    defensive_values = {}

    for _, row in sorted_df.iterrows():
        statlines = str(row['Stat']).split(',')
        if not statlines:
            continue

        players = re.findall(capitalized_words, str(row.get('Play', '')))

        if statlines[0] == 'DP':
            for p in players:
                defensive_values[p] = defensive_values.get(p, 0) + (1 / 3)

        elif statlines[0] == 'Assists':
            for p in players:
                defensive_values[p] = defensive_values.get(p, 0) + (0.5 / 3)

        elif statlines[0] == 'E':
            for p in players:
                defensive_values[p] = defensive_values.get(p, 0) - (1 / 3)

    return defensive_values


