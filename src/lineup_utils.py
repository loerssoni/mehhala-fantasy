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
    combs = product(*[position_lookup[player] + ['WC'] if position_lookup[player] != ['G'] else position_lookup[player] for player in selected_team])
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

def can_include_player(player_to_check, available_positions, position_lookup):
    for pos in position_lookup[player_to_check]:
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


def get_rest_of_season_games(date_now, player_games, selected_team, position_lookup, goalie_start_prob):
    player_rest_of_season_games = {p:0 for p in player_games.playerId.unique()}
    for date in pd.date_range(date_now, player_games.ts.max()):
        available_games = player_games[(player_games.ts == date)]
        day_sel_team = available_games[(available_games.playerId.isin(selected_team))].playerId.tolist()
        available_positions = get_available_positions(day_sel_team, position_lookup)

        day_goalie_prob = goalie_start_prob[[p for p in day_sel_team if p in goalie_start_prob]]

        for player in available_games.playerId.tolist():
            if player in goalie_start_prob:
                player_rest_of_season_games[player] += goalie_probability(day_goalie_prob)

            elif can_include_player(player, available_positions, position_lookup):
                player_rest_of_season_games[player] += 1

    return pd.Series(player_rest_of_season_games)

