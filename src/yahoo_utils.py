import pandas as pd

def get_players(q):
    players = q.get_league_players()
    pdf = pd.DataFrame([{
    'pos':p.display_position.split(','), 
    'name': p.name.full, 
    'player_key':p.player_key, 
    'team': p.editorial_team_abbr,
    'status': p.status 
     } for p in players])
    return pdf

def get_ts(t):
    return pd.Timestamp(t, unit='s', tz='UTC')\
        .tz_convert('US/Pacific')

def iso_get_ts(t):
    return pd.Timestamp(t).tz_convert('US/Pacific')

def get_gameweek(date):
    import requests
    games = []
    url = f'https://api-web.nhle.com/v1/schedule/{date}'
    r = requests.get(url)
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

def get_transactions(q):
    transactions = q.get_league_transactions()
    transactions = [transactions[i] for i in argsort([t.timestamp for t in transactions])]
    tr_by_date = {}
    for t in transactions:
        k = (get_ts(t.timestamp) + pd.Timedelta('1d')).date()
        if k not in tr_by_date.keys():
            tr_by_date[k] = []
        tr_by_date[k] += get_moves(t)
    return tr_by_date