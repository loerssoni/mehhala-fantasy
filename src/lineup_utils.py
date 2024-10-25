from itertools import combinations, product, chain
from ast import literal_eval
import collections
import pandas as pd
from process_data import PRED_COLS


POSITIONS_QUOTA = {
        'D': 4,
        'C': 2,
        'RW': 2,
        'LW': 2,
        'G': 2,
        'WC': 1
}

def all_elements_smaller(a, b):
    # Check each key in Counter a
    for key in a:
        # If key is missing in b or a[key] is not less than b[key], return False
        if key not in b:
            return False
        if a[key] > b[key]:
            return False
    return True

def check_valid_lineup(lineup, position_lookup, positions_quota=None):
    if len(lineup) < 3:
        return True
    if positions_quota is None:
        positions_quota = {k: v for k, v in POSITIONS_QUOTA.items()}
    
    positions_map = {k:[] for k in positions_quota}
    multi_positions = []
    for player in lineup:
        if len(position_lookup[player]) == 1:
            pos = position_lookup[player][0]
            if len(positions_map[pos]) >= positions_quota[pos]:
                if len(positions_map['WC']) == 0:
                    positions_map['WC'] = [player]
                else:
                    return False
            else:
                positions_map[pos] += [player]     
        else:
            multi_positions.append(player)

    positions_left = [p for p, v in positions_map.items() for i in range(positions_quota[p]-len(v))]
    multi_pos_positions = [position_lookup[p] for p in multi_positions]
    if len(multi_pos_positions) == 0:
        return True

    cc = collections.Counter(positions_left))
        
    for c in list(product(*multi_pos_positions)):
        if all_elements_smaller(collections.Counter(c), cc):
            return True
    return False
    

def get_valid_lineups(day_teams, min_players=13):
    position_lookup = day_teams.set_index(['playerId'], drop=False).pos.apply(literal_eval).to_dict()
    n_combs = min(min_players, len(position_lookup))
    lineups = list(combinations(position_lookup.keys(), n_combs))

    
    valid_lineups = [l for l in lineups if check_valid_lineup(l, position_lookup) != False and len(l) > 0]
    if len(valid_lineups) == 0:
        if n_combs > 2:
            return get_valid_lineups(day_teams, min_players=n_combs-1)
    return valid_lineups

def get_lineup_teams(lineups, day_teams):
    teams = []
    for lineup in lineups:
        match_filter = (day_teams.playerId.isin(lineup))
        teams.append(day_teams[match_filter])
    return pd.concat(teams, keys=range(len(teams)), names=['lineup'])



def get_resulted_points(lineup):
    metrics = [m for metrics in PRED_COLS.values() for m in metrics]
    res = lineup[metrics].sum()
    return res

def get_pct_lineups(day_teams, team_id):
    lineups = get_valid_lineups(day_teams[day_teams.team_id == team_id])
    if len(lineups) == 0:
        return [()]
    lineup_teams = get_lineup_teams(lineups, day_teams)
    ranksum = lineup_teams.groupby('lineup')['rank'].sum()
    best = ranksum.rank(pct=True).sort_values().index.tolist()
    return [lineups[i] for i in best]


def get_positions_quota(selected_team, pos_lookup):
    positions_quota = {k: v for k, v in POSITIONS_QUOTA.items()}
    lineup_check_players = []
    for player in selected_team:
        if len(pos_lookup[player]) == 1:
            positions_quota[pos_lookup[player][0]] -= 1
        else:
            lineup_check_players.append(player)
    return positions_quota, lineup_check_players


def get_rest_of_season_games(date_now, player_games, selected_team, position_lookup):
    player_rest_of_season_games = {p:0 for p in player_games.playerId.unique()}
    for date in pd.date_range(date_now, player_games.ts.max()):
        available_games = player_games[(player_games.ts == date)]
        day_sel_team = available_games[(available_games.playerId.isin(selected_team))]
        positions_quota, lineup_check_players = get_positions_quota(day_sel_team.playerId.tolist(), position_lookup)
        for player in available_games.playerId.tolist():
            if check_valid_lineup(lineup_check_players + [player], position_lookup, positions_quota):
                player_rest_of_season_games[player] += 1
    return pd.Series(player_rest_of_season_games)
