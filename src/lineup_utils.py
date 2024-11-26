from itertools import product
import collections
import pandas as pd


POSITIONS_QUOTA = {
        'D': 4,
        'C': 2,
        'RW': 2,
        'LW': 2,
        'WC': 1
}

def get_available_positions(selected_team, position_lookup):
    combs = product(*[position_lookup.loc[player, 'pos_l'] + ['WC'] if position_lookup.loc[player, 'pos_l'] != ['G'] else position_lookup.loc[player, 'pos_l'] for player in selected_team])
    valid_positions = set()
    min_slots_left = sum(POSITIONS_QUOTA.values())
    for c in combs:
        left_pos = collections.Counter(POSITIONS_QUOTA) - collections.Counter(c)
        comb_slots_left = sum(left_pos.values())
        if comb_slots_left < min_slots_left:
            valid_positions = set(left_pos.keys())
            min_slots_left = comb_slots_left
        elif comb_slots_left == min_slots_left:
            valid_positions = valid_positions.union(set(left_pos.keys()))
    return valid_positions

def can_include_player(player_to_check, available_positions, players):
    for pos in players.loc[player_to_check, 'pos_l']:
        if pos in available_positions:
            return True
    return False

def goalie_probability(p_current_selected):
    # Compute p_none (probability that none are true in p_current_selected)
    p_none = (1 - p_current_selected).product()

    # Compute p_one (probability that exactly one is true in p_current_selected)
    p_one = sum(
        p * (1 - p_current_selected.drop(index)).product()
        for index, p in p_current_selected.items()
    )

    # Compute the total probability
    p_result = (p_none + p_one)

    return p_result


def get_rest_of_period_games(date_now, player_games, selected_team, players, preds):
    goalie_start_prob = preds.icetime.dropna()
    
    player_rest_of_season_games = {p:0 for p in player_games.playerId.unique() if p not in selected_team}
    for date in pd.date_range(date_now, player_games.date.max()):
        
        available_games = player_games[(player_games.date == date)]
        day_sel_team = available_games[(available_games.playerId.isin(selected_team))].playerId.tolist()
        available_positions = get_available_positions(day_sel_team, players)

        day_goalie_prob = goalie_start_prob[[p for p in day_sel_team if p in goalie_start_prob]]

        available_players = [p for p in available_games.playerId.tolist() if p not in selected_team]
        for player in available_players:
            if player in goalie_start_prob:
                player_rest_of_season_games[player] += goalie_probability(day_goalie_prob)
            elif can_include_player(player, available_positions, players):
                player_rest_of_season_games[player] += 1

    return pd.Series(player_rest_of_season_games)
