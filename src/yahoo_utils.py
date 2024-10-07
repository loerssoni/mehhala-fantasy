import pandas as pd
from yfpy.query import YahooFantasySportsQuery
import json
with open('../creds/token.json', 'r') as f:
    yahoo_access_token = json.loads(f.read())



league_ids = {
     427:21834,
     453: 15482
}
          
def get_q(game_id):
    q = YahooFantasySportsQuery(
        game_id = game_id,
        league_id = league_ids[game_id],
        game_code = 'nhl',
        yahoo_access_token_json = yahoo_access_token,
        browser_callback = False,
    )
    return q

def fetch_player_data(game_id):
    q = get_q(game_id)
    players = q.get_league_players()
    pdf = pd.DataFrame([{
    'pos':p.display_position.split(','), 
    'name': p.name.full, 
    'player_key':p.player_key, 
    'team': p.editorial_team_abbr,
    'status': p.status 
     } for p in players])
    names_map = pd.read_csv('data/name_mapping.csv', index_col='name_match')
    bios = pd.read_hdf('data/bios.h5')
    bios['name'] = bios[['name']].replace(names_map.to_dict())['name']
    pdf = pdf.merge(bios, how='inner', on='name', suffixes=('_yh', ''))
    return pdf

import os

def get_players(game_id):
    filepath = f'data/players{game_id}.csv'
    if not os.path.isfile(filepath):
        players = fetch_player_data(game_id)
        players.to_csv(filepath)
    else:
        players = pd.read_csv(filepath,index_col=0)
    players[~players.playerId.duplicated()].copy()
    return players

def get_ts(t):
    return pd.Timestamp(t, unit='s', tz='UTC')\
        .tz_convert('US/Pacific')

def iso_get_ts(t):
    return pd.Timestamp(t).tz_convert('US/Pacific')

def get_all_matchups(game_id):
    q = get_q(game_id)
    all_matchups = []
    for i in range(1, 27):
        all_matchups += q.get_league_matchups_by_week(i)
    return all_matchups

def get_matchup_results(matchups):
    stats_map = {
        1: 'goals',
        2: 'assists',
        4: 'plusmin',
        5: 'pim',
        8: 'ppp',
        14: 'shots',
        16: 'fow',
        31: 'hits',
        32: 'blocks'
    }
    stats = []
    for match in matchups:
        for mt in match.teams:
            for s in mt.team_stats.stats:
                stats.append({'id': s.stat_id, 'val': s.value})
    stats = pd.DataFrame(stats)
    stats['name'] = stats.id.apply(lambda x: stats_map.get(x, None))
    return stats


def get_gameweek(date):
    import requests
    games = []
    url = f'https://api-web.nhle.com/v1/schedule/{date}'
    print(url)
    r = requests.get(url)
    print(r.status_code)
    if r.status_code != 200:
        print(r.text)
    data = r.json()
    for week in data['gameWeek']:
        for game in week['games']:
            games.append({
                'gameId': game['id'],
                'home': game['homeTeam']['abbrev'],
                'away': game['awayTeam']['abbrev'],
                'ts': iso_get_ts(game['startTimeUTC']).date()
            })
    return games, data.get('nextStartDate')

def get_schedule(dates):
    games = []
    next_start = dates[0].strftime('%Y-%m-%d')
    for date in dates:
        if date >= pd.Timestamp(next_start):
            g, next_start = get_gameweek(date.strftime('%Y-%m-%d'))
            games += g
    games = [g for g in games if dates[0].date() <= g['ts'] <= dates[-1].date()]
    return games

import json
import pickle

def get_games_by_week(game_id):
    filepath = f'data/schedule{game_id}.pickle'
    if not os.path.isfile(filepath):
        q = get_q(game_id)
        weeks = q.get_game_weeks_by_game_id(game_id)
        weekly_schedule = {}
        for week in gweeks:
            dates = pd.date_range(week.start, week.end)
            weekly_schedule[week.display_name] = get_schedule(dates)
        with open(filepath, 'wb') as f:
            pickle.dump(weekly_schedule, f)
    else:
        with open(filepath, 'rb') as f:
            weekly_schedule = pickle.load(f)

    return weekly_schedule

def get_moves(t):
    moves = []
    for p in t.players:
        if p.transaction_data.destination_team_key:
            moves.append({'player': p.player_key, 'to': p.transaction_data.destination_team_key})
        if p.transaction_data.source_team_key:
            moves.append({'player': p.player_key, 'from': p.transaction_data.source_team_key})
    return moves


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def get_transactions(game_id):
    q = get_q(game_id)
    transactions = q.get_league_transactions()
    transactions = [transactions[i] for i in argsort([t.timestamp for t in transactions])]
    tr_by_date = {}
    for t in transactions:
        k = (get_ts(t.timestamp) + pd.Timedelta('1d')).date()
        if k not in tr_by_date.keys():
            tr_by_date[k] = []
        tr_by_date[k] += get_moves(t)
    return tr_by_date


def get_initial_teams(game_id):
    q = get_q(game_id)
    players = q.get_league_draft_results()
    teams = {}
    for p in players:
        if p.team_key not in teams:
            teams[p.team_key] = []
        teams[p.team_key].append(p.player_key)
    return teams

def get_teams(game_id, force=False):
    filepath = f'data/teams{game_id}.csv'
    if not os.path.isfile(filepath) or force:
        q = get_q(game_id)
        s = q.get_league_settings()
        weeks = q.get_game_weeks_by_game_id(game_id)
        dates = pd.date_range(get_ts(s.draft_time).strftime('%Y-%m-%d'),
                  weeks[-1].end, freq='d').date

        from copy import deepcopy

        teams = {}
        transactions = get_transactions(game_id)
        current_teams = get_initial_teams(game_id)
        for date in dates:
            trs = transactions.get(date, [])
            for move in trs:
                if 'to' in move:
                    current_teams[move['to']].append(move['player'])
                if 'from' in move:
                    current_teams[move['from']].remove(move['player'])
            teams[date] = deepcopy(current_teams)

        df = pd.DataFrame(teams).T.melt(var_name='team_id', value_name='player', ignore_index=False)
        df = df.reset_index()
        df = df.join(df.player.explode(), rsuffix='_key', validate='one_to_many')
        df['index'] = pd.to_datetime(df['index'])
        df = df.drop('player', axis=1).set_index('index')
        df.index.name = 'date'
        df.to_csv(filepath)
    else:
        df = pd.read_csv(filepath,index_col=0)
    df = df.set_index(pd.to_datetime(df.index))

    return df