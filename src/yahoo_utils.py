import pandas as pd
from yfpy.query import YahooFantasySportsQuery
import json

from ast import literal_eval

with open('../creds/token.json', 'r') as f:
    yahoo_access_token = json.loads(f.read())

GAME_ID = 453
TEAM_KEY = "453.l.15482.t.3"

league_ids = {
     453: 15482
}

def get_q():
    q = YahooFantasySportsQuery(
        game_id = GAME_ID,
        league_id = league_ids[GAME_ID],
        game_code = 'nhl',
        yahoo_access_token_json = yahoo_access_token,
        browser_callback = False,
    )
    return q

def get_matchup(team_key, date=None):
    q = get_q()
    matchups = q.get_team_info(team_key.split('.')[-1]).matchups
    if date:
        matchup = [m for m in matchups if pd.to_datetime(m.week_end).date() > date + pd.Timedelta('1d')][0]
    else:
        return None
    
    matchup = {
        'week': matchup.week,
        'opponent': [t.team_key for t in matchup.teams if t.team_key != team_key][0],
        'start': pd.to_datetime(matchup.week_start).date(),
        'end': pd.to_datetime(matchup.week_end).date()
    }
    return matchup

def refresh_players():
    q = get_q()
    players = q.get_league_players()
    pdf = pd.DataFrame([{
    'pos': str(p.display_position.split(',')) , 
    'name': p.name.full, 
    'player_key':p.player_key, 
    'team': p.editorial_team_abbr,
    'status': p.status 
     } for p in players])
    names_map = pd.read_csv('data/name_mapping.csv', index_col='name_match')
    bios = pd.read_hdf('data/bios.h5')
    bios['name'] = bios[['name']].replace(names_map.to_dict())['name']
    pdf = pdf.merge(bios, how='inner', on='name', suffixes=('_yh', ''))
    
    faulty_keys = [
        '453.p.32762',
        '453.p.7654',
        '453.p.5774'
    ]
    pdf = pdf.loc[~pdf.player_key.isin(faulty_keys)]
    
    faulty_ids = [
        8483575
    ]
    pdf = pdf.loc[~pdf.playerId.isin(faulty_ids)]
    return pdf.to_csv('data/players.csv', index=False)

def iso_get_ts(t):
    return pd.Timestamp(t).tz_convert('US/Pacific')

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
                'date': iso_get_ts(game['startTimeUTC']).date()
            })
    return games, data.get('nextStartDate')

def get_schedule(dates):
    games = []
    next_start = dates[0].strftime('%Y-%m-%d')
    for date in dates:
        if date >= pd.Timestamp(next_start):
            g, next_start = get_gameweek(date.strftime('%Y-%m-%d'))
            games += g
    games = [g for g in games if dates[0].date() <= g['date'] <= dates[-1].date()]
    return games

def refresh_player_games():
    q = get_q()
    weeks = q.get_team_info(TEAM_KEY.split('.')[-1]).matchups
    current_schedule = {}
    for week in weeks:
        dates = pd.date_range(week.week_start, week.week_end)
        current_schedule[week.week] = get_schedule(dates)

    games_list = []
    for k, v in current_schedule.items():
        for r in v:
            a = {'week':k}
            a.update(r)
            games_list.append(a)
    games_df = pd.DataFrame(games_list)
    games_df['date'] = pd.to_datetime(games_df['date'])
    
    players = pd.read_csv('data/players.csv')
    players['team_yh'] = players.team_yh.replace({
        'SJ':'SJS',
        'LA':'LAK',
        'TB':'TBL',
        'NJ':'NJD'
    })
    
    player_games = pd.concat([
        games_df.merge(players, how='left', left_on='home', right_on='team_yh'),
        games_df.merge(players, how='left', left_on='away', right_on='team_yh')
    ])[['week','gameId','date','pos','name','playerId']]
    player_games = player_games.dropna()
    player_games.to_csv('data/player_games.csv', index=False)
    return

def load_player_games(mode, date, matchup):
    player_games = pd.read_csv('data/player_games.csv')
    player_games = player_games.set_index('playerId')
    player_games['date'] = pd.to_datetime(player_games.date)
        
    date_filter = (player_games.date.dt.date >= date)
    if mode == 'week':
        date_filter = date_filter & (player_games.date.dt.date <= matchup['end'])
    return player_games.loc[date_filter].copy()

def refresh_teams():
    q = get_q()
    teams = q.get_league_teams()
    team_info = []
    rosters = []
    for t in teams:
        t_info = q.get_team_info(t.team_key.split('.')[-1])
        for player in t_info.roster.players:
            rosters.append({
                'team_key':t_info.team_key,
                'team_name':t_info.name,
                'player_key': player.player_key,
                'selected_position':player.selected_position.position,
            })
    teams = pd.DataFrame(rosters)
    teams.to_csv('data/teams.csv', index=False)
    return

def load_players(team_key, matchup=None):
    players = pd.read_csv('data/players.csv').set_index('playerId')
    teams = pd.read_csv('data/teams.csv')

    players = players.reset_index().merge(teams, how='left', on='player_key').set_index('playerId')
    
    players['pos_l'] = players.pos.apply(literal_eval)
    
    players['current_lineup'] = players.team_key == team_key
    players['available'] = players.team_key.isna() | players.current_lineup
    players['is_rostered'] = players.team_key.notna()
    
    if matchup is not None:
        players['opp_lineup'] = players.team_key == matchup['opponent']

    return players
