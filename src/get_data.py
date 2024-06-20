def load_skater_data(start_season, end_season, filename):
    import requests
    from bs4 import BeautifulSoup
    urls = []
    for season in range(start_season, end_season):
        base_url = f'https://moneypuck.com/moneypuck/playerData/teamPlayerGameByGame/{season}/regular/skaters/'
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
    print('received:', len(dfs), 'csvs')
    if len(dfs) > 0:
        df = pd.concat(dfs)
        df.to_csv(filename, index=False)

def load_history():
    filename = 'data/gamebygame16.csv'
    load_skater_data(2017, 2023, filename)

def load_current():
    filename = 'data/gamebygame24.csv'
    load_skater_data(2023, 2024, filename)

def combine_history():
    import pandas as pd
    df1 = pd.read_csv('data/gamebygame16.csv')
    df2 = pd.read_csv('data/gamebygame24.csv')
    df1 = pd.concat([df1, df2])
    df1.to_csv('data/gamebygame.csv')

def process_y():
    import pandas as pd
    df = pd.read_csv('data/gamebygame.csv')
    y_df = df[df['situation'] == 'all'].set_index(['playerId',
                         'gameId', 
                         'gameDate',
                        'playerTeam',
                        'opposingTeam'])
    ys = {
        'goals': ['I_F_goals'],
        'assists': ['I_F_primaryAssists', 'I_F_secondaryAssists'],
        'shots': ['I_F_shotsOnGoal'],
        'hits': ['I_F_hits'],
        'blocks': ['shotsBlockedByPlayer'],
        'pim': ['penalityMinutes'],
        'fow': ['I_F_faceOffsWon'],
    }
    for k, v in ys.items():
        ys[k] = y_df[v].sum(1)
    ys = pd.DataFrame(ys)
    y_df = df[df['situation'] == '5on5'].set_index(['playerId',
                         'gameId', 
                         'gameDate',
                        'playerTeam',
                        'opposingTeam'])[['OnIce_F_goals', 'OnIce_A_goals']]
    y_df.columns =['goalsfor', 'goalsaga']
    ys = ys.join(y_df)
    y_df = df[df['situation'] == '5on4'].set_index(['playerId',
                         'gameId', 
                         'gameDate',
                        'playerTeam',
                        'opposingTeam'])
    ys = ys.join(pd.DataFrame(y_df[['I_F_goals', 'I_F_primaryAssists', 'I_F_secondaryAssists']].sum(1), columns=['ppp']))
    print('Processed y, writing to csv')
    ys.to_csv('data/y.csv')

def process_pca():
    import pandas as pd
    df = pd.read_csv('data/gamebygame.csv')
    df = df.pivot(index=['playerId',
                         'gameId', 
                         'gameDate',
                        'playerTeam',
                        'opposingTeam'], columns=['situation'], values=df.columns[11:])
    print('Read in game data.')

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Lasso

    train_idx = df.index.get_level_values(2) < 20210800

    pipe = Pipeline([
        ('scl', StandardScaler()),
        ('pca', PCA(n_components=35))
    ])
    pipe.fit(df[train_idx])
    pcad = pipe.transform(df)
    pcad = pd.DataFrame(pcad, index=df.index)
    print('Processed pca, writing to csv')

    pcad.to_csv('data/gbgpca.csv')

def load_bios():
    import requests
    import io
    import pandas as pd
    url = 'https://moneypuck.com/moneypuck/playerData/playerBios/allPlayersLookup.csv'
    r = requests.get(url)
    s = io.StringIO(r.text)
    df = pd.read_csv(s)
    print('Loaded bios, writing to csv.')
    df.to_csv('data/bios.csv', index=False)


def load_team_data():
    import requests
    import io
    import pandas as pd
    url = 'https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv'
    r = requests.get(url)
    s = io.StringIO(r.text)
    df = pd.read_csv(s)
    df.to_csv('data/teams.csv', index=False)

