def load_player_data(start_season, end_season, filename, player_type):
    import requests
    from bs4 import BeautifulSoup
    urls = []
    for season in range(start_season, end_season):
        base_url = f'https://moneypuck.com/moneypuck/playerData/teamPlayerGameByGame/{season}/regular/{player_type}/'
        print(base_url)
        r = requests.get(base_url)
        soup = BeautifulSoup(r.text)
        season_as = soup.find_all('a', href=True)
        urls += [base_url + u.text for u in season_as if '.csv' in u.text]
    print('urls:', urls)
    import time
    import io
    import pandas as pd
    dfs = []
    for i, url in enumerate(urls):
        r = requests.get(url)
        s = io.StringIO(r.text)
        df = pd.read_csv(s)
        dfs.append(df)
        time.sleep(0.1)
        if r.status_code != 200:
            print(r.text)
            break
        if i % 10 == 9:
            print(r.status_code)
            print(i, '/', len(urls))
    print('received:', len(dfs), 'csvs of', len(urls))
    if len(dfs) > 0:
        df = pd.concat(dfs)
        df.to_hdf(filename, 'data', index=False)

def load_history():
    filename = 'data/gamebygame21.h5'
    load_player_data(2021, 2023, filename, 'skaters')
    filename = 'data/goalies21.h5'
    load_player_data(2021, 2023, filename, 'goalies')

def load_current():
    filename = 'data/gamebygame24.h5'
    load_player_data(2023, 2024, filename, 'skaters')
    filename = 'data/goalies24.h5'
    load_player_data(2023, 2024, filename, 'goalies')

def combine_history():
    import pandas as pd
    df1 = pd.read_hdf('data/gamebygame21.h5', 'data')
    df2 = pd.read_hdf('data/gamebygame24.h5', 'data')
    df1 = pd.concat([df1, df2])
    df1.to_hdf('data/gamebygame.h5', key='data')

    del df1, df2
    df1 = pd.read_hdf('data/goalies21.h5', 'data')
    df2 = pd.read_hdf('data/goalies24.h5', 'data')
    df1 = pd.concat([df1, df2])
    df1.to_hdf('data/goalies.h5', key='data')
    

def process_y():
    import pandas as pd
    df = pd.read_hdf('data/gamebygame.h5')

    df = df.pivot(index=['playerId',
                 'gameId', 
                'playerTeam'], columns=['situation'], values=df.columns[11:])
    df.columns = [f'{s[1]}_{s[0]}' for s in df.columns]
    df.to_hdf('data/games_p.h5', key='data')
    
    ys = {
        'g': ['all_I_F_goals'],
        'a': ['all_I_F_primaryAssists', 'all_I_F_secondaryAssists'],
        'sog': ['all_I_F_shotsOnGoal'],
        'hit': ['all_I_F_hits'],
        'block': ['all_shotsBlockedByPlayer'],
        'pim': ['all_penalityMinutes'],
        'fow': ['all_I_F_faceOffsWon'],
        'goalsfor':['5on5_OnIce_F_goals'],
        'goalsaga':['5on5_OnIce_A_goals'],
        'ppp':['5on4_I_F_goals', '5on4_I_F_primaryAssists', '5on4_I_F_secondaryAssists']
    }
    for k, v in ys.items():
        ys[k] = df[v].sum(1)
    ys = pd.DataFrame(ys)
    print('Processed y, writing to csv')
    ys.to_hdf('data/y.h5', key='data')
    


    df = pd.read_hdf('data/goalies.h5')
    df = df[df.situation == 'all'].copy()
    df = df.sort_values('gameId').reset_index(drop=True)
    df['season'] = (df.gameId // 1e6).astype(int)
    df['n_team'] = df.playerTeam != df.groupby('playerId')['playerTeam'].transform(lambda x: x.shift(1).bfill())
    df['n_team'] = df.groupby(['playerId', 'season'])['n_team'].cumsum()
    df['first_w_team'] = df.groupby(['playerTeam','season'])['gameId'].transform('first')
    df.loc[df.n_team > 0, 'first_w_team'] = df.loc[df.n_team > 0].groupby(['playerId','season','n_team'])['gameId'].transform('first')
    df['last_w_team'] = df.groupby(['playerTeam','season'])['gameId'].transform('last')
    df.loc[df.n_team < df.groupby('playerId').n_team.transform('max'), 'last_w_team']   =\
        df.loc[df.n_team < df.groupby('playerId').n_team.transform('max')].\
        groupby(['playerId', 'season','n_team'])['gameId'].transform('last')

    df.groupby('playerId').n_team.max()

    dt = pd.read_hdf('data/teams.h5')
    dt = dt[(dt.situation == 'all')&(dt.gameId > 2021000000)].copy()
    dt = dt.sort_values(['gameId','playerTeam']).reset_index(drop=True)
    dt['win'] = (dt['goalsFor'] - dt['goalsAgainst']) > 0

    games = dt[['season','gameId','playerTeam','home_or_away', 'win']].drop_duplicates()
    pts = df[['season','playerTeam','playerId','first_w_team', 'last_w_team', 'name']].drop_duplicates()
    games = games.merge(pts, how='left', on=['season','playerTeam'])
    games = games[(games.first_w_team <= games.gameId)&(games.last_w_team >= games.gameId)].copy()
    games = games.drop(['first_w_team','last_w_team',], axis=1)
    games = games.merge(df, how='left', on=['season','gameId','playerTeam','playerId', 'name', 'home_or_away'])
    games = games.drop([ 'opposingTeam', 'gameDate', 'position', 'situation'], axis=1)
    games = games.fillna(0)

    y_df = games.set_index(['playerId',
                 'gameId', 
                'playerTeam'])
    y_df['win'] = (y_df['win']&(y_df['icetime'] > 1800)).astype(int)
    ys = {
        'ga': ['goals'],
        'save': ['ongoal'],
        'win':['win'],
        'icetime':['icetime'],
    }
    for k, v in ys.items():
        ys[k] = y_df[v].sum(1)
        
    ys['win'] = (ys['win'] > 0).astype(int)
    ys['so'] = ys['ga'] == 0
    ys['save'] = ys['save'] - ys['ga']
    ys['icetime'] /= 360
    ys = pd.DataFrame(ys)
    
    print('Processed y, writing to csv')
    ys.to_hdf('data/y_g.h5', key='data')
    y_df = y_df.drop(['season','home_or_away','name'], axis=1)
    y_df.to_hdf('data/games_g.h5', key='data')
    

def load_bios():
    import requests
    import io
    import pandas as pd
    url = 'https://moneypuck.com/moneypuck/playerData/playerBios/allPlayersLookup.csv'
    r = requests.get(url)
    s = io.StringIO(r.text)
    df = pd.read_csv(s)
    print('Loaded bios, writing to csv.')
    df.to_hdf('data/bios.h5', key='data', index=False)


def load_team_data():
    import requests
    import io
    import pandas as pd
    url = 'https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv'
    r = requests.get(url)
    s = io.StringIO(r.text)
    df = pd.read_csv(s)
    df.to_hdf('data/teams.h5', key='data', index=False)

