import pandas as pd
from get_data import INDEX_COLS

PRED_COLS = {
    'skater': ['g', 'a', 'sog', 'fow', 'hit', 'block', 'pim', 'goalsfor', 'goalsaga', 'ppp'],
    'goalie': ['ga','win','so', 'save', 'icetime']
}

def get_player_stats(player_type, cols=None, schedule=None):
    
    filenames = {
        'skater':('data/games_p.h5','data/y.h5'),
        'goalie':('data/games_g.h5','data/y_g.h5')
    }

    x_f, y_f = filenames[player_type]
    X = pd.read_hdf(x_f)
    X = X.sort_index(level='gameId')
    if cols is not None:
        X = X[cols].copy()
    X = X.groupby('playerId', as_index=False).rolling(30).mean().drop('playerId', axis=1)
    y = pd.read_hdf(y_f)
    
    if schedule is not None:
        X_train, X_test = get_schedule_adjusted_stats(X, schedule)
        X_train = X_train.groupby('playerId').shift(1).dropna()
        X = pd.concat([X_train, X_test])
    else:
        X = X.groupby('playerId').shift(1)
        X = X.dropna()
    return X, y

def get_rest_of_season_player_stats(player_type, cols=None, schedule=None):
    
    filenames = {
        'skater':('data/games_p.h5','data/y.h5'),
        'goalie':('data/games_g.h5','data/y_g.h5')
    }

    x_f, y_f = filenames[player_type]
    X = pd.read_hdf(x_f)
    X = X.sort_index(level='gameId')
    if cols is not None:
        X = X[cols].copy()
    X = X.groupby('playerId', as_index=False).rolling(30).mean().drop('playerId', axis=1)
    y = pd.read_hdf(y_f)
    X = X.groupby('playerId').shift(1)
    y['date'] = pd.to_datetime(y.index.get_level_values('gameDate'), format='%Y%m%d')
    y['season'] = (y.index.get_level_values('gameId') // 1000000)
    y = y.sort_values('date', ascending=False)
    y_r = y.sort_values('date', ascending=False).groupby(['playerId','season'], as_index=False)[PRED_COLS[player_type]].expanding().mean()
    y_r = y_r.set_index(X.index)
    return X, y_r

def get_schedule_adjusted_stats(X, schedule):
    gdates = pd.to_datetime(X.index.get_level_values('gameDate'), format='%Y%m%d')
    dfs = []
    first_week = True
    for gameweek, games in schedule.items():
        games_df = pd.DataFrame(games)
        start_date = games_df.ts.min()

        if first_week:
            first_week = False
            test_period_start = start_date
            X_train = X.loc[(gdates.date < start_date)].copy()
        latest_stats = X.reset_index(level='playerTeam')\
            .loc[(gdates.date < start_date)&(gdates.date > start_date - pd.Timedelta(days=365))]
        if 'playerId' in latest_stats.index.names:
            latest_stats = latest_stats.groupby('playerId').last()
        else:
            latest_stats = latest_stats.groupby('playerTeam').last()
        home_players = games_df.merge(latest_stats.reset_index(), left_on='home', right_on='playerTeam')
        away_players = games_df.merge(latest_stats.reset_index(), left_on='away', right_on='playerTeam')
        home_players['opposingTeam'] = home_players['away'].copy()
        home_players['home'] = 1
        away_players['opposingTeam'] = away_players['home'].copy()
        away_players['home'] = 0
        games_df = pd.concat([home_players, away_players])
        games_df['gameDate'] = pd.to_datetime(games_df.ts).dt.strftime('%Y%m%d').astype(int)
        games_df = games_df.set_index(X_train.index.names).drop(['away','ts'], axis=1)
        dfs.append(games_df)
    X_test = pd.concat(dfs)
    return X_train, X_test

def get_team_stats(schedule=None):
    df = pd.read_hdf('data/teams.h5')
    df = df.sort_values(['gameId', 'home_or_away'])
    df = df.pivot(index=['team', 'gameId', 'gameDate', 'opposingTeam', 'home_or_away'], columns=['situation'], values=df.columns[11:])
    df['home'] = (df.index.get_level_values('home_or_away') == 'HOME').astype(int)
    df = df.reset_index(level='home_or_away', drop=True)
    df = df.set_index(df.index.rename(['playerTeam','gameId', 'gameDate', 'opposingTeam']))
    df.columns = [f'{s[1]}_{s[0]}' for s in df.columns]
    df = df.sort_values(['playerTeam','gameId'])

    df = df.groupby('playerTeam', as_index=False).rolling(10, min_periods=5).mean().drop('playerTeam', axis=1)
    
    if schedule is not None:
        X_train, X_test = get_schedule_adjusted_stats(df, schedule)
        X_train = X_train.groupby('playerTeam').shift(1).dropna()
        X = pd.concat([X_train, X_test])
    else:
        X = df.groupby('playerTeam').shift(1).dropna()
    return X

def get_bios():
    df = pd.read_hdf('data/bios.h5')
    df = df.join(pd.get_dummies(df[['position', 'shootsCatches']]).set_index(df.index))

    ecols = ['playerId', 'name', 'position_C', 'position_D', 'position_G',
    'position_L', 'position_R', 'shootsCatches_L', 'shootsCatches_R']
    X_b = df[ecols]
    return X_b

def get_projections():
    df = pd.read_csv('data/all_projections.csv')
    df.columns = df.columns.str.lower()
    return df

def get_weekly_stats(player_type, cols=None, schedule=None, fillna=True):
    X_p, y = get_player_stats(player_type, cols=cols, schedule=schedule)
    team_stats = get_team_stats(schedule=schedule)
    X_p = X_p.merge(team_stats, on=['gameId','playerTeam'], how='left').set_index(X_p.index)
    
    X_p = X_p.merge(team_stats, left_on=['gameId','opposingTeam'], right_on=['gameId','playerTeam'], how='left', suffixes=('','_opp')).set_index(X_p.index)
    
    bios = get_bios()
    X_p = X_p.merge(bios, on='playerId', how='left').set_index(X_p.index)
    
    X_proj = get_projections()
    X_p['season'] = X_p.index.get_level_values('gameId') // (1000*1000)
    X_p = X_p.merge(X_proj, left_on=['name', 'season'], right_on=['player', 'season'], how='left').set_index(X_p.index)
    
    X_p = X_p.loc[:,~X_p.isna().all()].copy()
    X_p = X_p.drop(['name','player','playerId'], axis=1)
    if fillna:
        X_p = X_p.fillna(0)
    return X_p, y

def get_rest_of_season_stats(player_type, cols=None, schedule=None, fillna=True):
    X_p, y = get_rest_of_season_player_stats(player_type, cols=cols, schedule=schedule)
    bios = get_bios()
    X_p = X_p.merge(bios, on='playerId', how='left').set_index(X_p.index)
    
    X_proj = get_projections()
    X_p['season'] = X_p.index.get_level_values('gameId') // (1000*1000)
    X_p = X_p.merge(X_proj, left_on=['name', 'season'], right_on=['player', 'season'], how='left').set_index(X_p.index)
    X = X.drop(['playerId','name','player'], axis=1)
    return X_p, y
