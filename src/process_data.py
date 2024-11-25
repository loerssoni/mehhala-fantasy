import pandas as pd
from get_data import INDEX_COLS

PRED_COLS = {
    'skater': ['g', 'a', 'sog', 'fow', 'hit', 'block', 'pim', 'goalsfor', 'goalsaga', 'ppp'],
    'goalie': ['ga','win','so', 'save', 'icetime']
}
cats = ['g','a','sog','fow','hit','block','pim','plusmin','ppp', 'ga','win','so','save']


def get_rest_of_season_player_stats(player_type, cols=None, window=30, shift=True):
    
    filenames = {
        'skater':('data/games_p.h5','data/y.h5'),
        'goalie':('data/games_g.h5','data/y_g.h5')
    }

    x_f, y_f = filenames[player_type]
    X = pd.read_hdf(x_f)
    X = X.sort_index(level='gameId')

    if cols is not None:
        used_cols = list(set([c.replace('_played', '') for c in cols]))
        X = X[used_cols].copy()
        
    min_periods = min(window, 20)
    
    if player_type == 'goalie':
        X = X.fillna(0).groupby('playerId', as_index=False).rolling(window, min_periods=min_periods).mean().drop('playerId', axis=1)
        X_played = X.dropna().groupby('playerId', as_index=False).rolling(window, min_periods=min_periods).mean().drop('playerId', axis=1)
        X = X.join(X_played, rsuffix='_played')
        if cols is not None:
            X = X[cols].copy()
    else:
        X = X.dropna().groupby('playerId', as_index=False).rolling(window, min_periods=min_periods).mean().drop('playerId', axis=1)
        
    X = X.groupby('playerId').ffill()

    if shift:
        X = X.groupby('playerId').shift(1)
        
    y = pd.read_hdf(y_f)    
    y = y.sort_values('gameId', ascending=False)
    
    y_std = y.copy()

    if player_type == 'skater':
        y_std['plusmin'] = y_std['goalsfor'] - y_std['goalsaga']
    
    if player_type == 'goalie':
        y_std.ga = (y.icetime / y_std.ga).clip(0,1.5)
    
    y_std = ((y_std - y_std.groupby('playerId').transform('mean')) ** 2).mean() ** 0.5
    y_std = y_std[[c for c in y_std.index if c in cats]].copy()

    min_periods = 5

    if player_type == 'goalie':
        window = 5
        stat_cols = [c for c in PRED_COLS[player_type] if c != 'icetime']
        y[stat_cols] = y[stat_cols].apply(lambda x: x / y.icetime)
        y_played = y.dropna().groupby(['playerId'], as_index=False)[stat_cols].rolling(window, min_periods=min_periods).mean().drop('playerId', axis=1)
        y_all = y.groupby(['playerId'], as_index=False)[['icetime']].rolling(window, min_periods=min_periods).mean().drop('playerId', axis=1)
        y_r = y_all[['icetime']].join(y_played)
    
    if player_type == 'skater':
        window = 20
        y_r = y.groupby(['playerId'], as_index=False)[PRED_COLS[player_type]].rolling(window, min_periods=min_periods).mean()
    
    y_r = y_r.loc[X.index]
    
    X['dummyvar'] = 1
    return X, y_r, y_std

def get_team_stats(shift=True):
    df = pd.read_hdf('data/teams.h5')
    df = df.sort_values(['gameId', 'home_or_away'])
    df = df.pivot(index=['team', 'gameId', 'gameDate', 'opposingTeam', 'home_or_away'], columns=['situation'], values=df.columns[11:])
    df['home'] = (df.index.get_level_values('home_or_away') == 'HOME').astype(int)
    df = df.reset_index(level='home_or_away', drop=True)
    df = df.set_index(df.index.rename(['playerTeam','gameId', 'gameDate', 'opposingTeam']))
    df.columns = [f'{s[1]}_{s[0]}' for s in df.columns]
    df = df.sort_values(['playerTeam','gameId'])

    df = df.groupby('playerTeam', as_index=False).rolling(10, min_periods=5).mean().drop('playerTeam', axis=1)
    
    if shift:
        return df.groupby('playerTeam').shift(1).dropna()
    else:
        return df
    return

def get_bios():
    df = pd.read_hdf('data/bios.h5')
    df = df.join(pd.get_dummies(df[['position', 'shootsCatches']]).set_index(df.index))

    ecols = ['playerId', 'name', 'position_C', 'position_D', 'position_G',
    'position_L', 'position_R', 'shootsCatches_L', 'shootsCatches_R']
    X_b = df[ecols]
    return X_b