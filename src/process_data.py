import pandas as pd
from get_data import INDEX_COLS

PRED_COLS = {
    'skater': ['g', 'a', 'sog', 'fow', 'hit', 'block', 'pim', 'goalsfor', 'goalsaga', 'ppp'],
    'goalie': ['ga','win','so', 'save', 'icetime']
}

def get_rest_of_season_player_stats(player_type, cols=None, window=30, shift=True):
    
    filenames = {
        'skater':('data/games_p.h5','data/y.h5'),
        'goalie':('data/games_g.h5','data/y_g.h5')
    }

    x_f, y_f = filenames[player_type]
    X = pd.read_hdf(x_f)
    X = X.sort_index(level='gameId')
    if cols is not None:
        X = X[cols].copy()
    X = X.groupby('playerId', as_index=False).rolling(window).mean().drop('playerId', axis=1)
    y = pd.read_hdf(y_f)
    
    if shift:
        X = X.groupby('playerId').shift(1)
    
    y['date'] = pd.to_datetime(y.index.get_level_values('gameDate'), format='%Y%m%d')
    y['season'] = (y.index.get_level_values('gameId') // 1000000)
    y = y.sort_values('date', ascending=False)
    
    if player_type == 'goalie':
        window = 15
        
    if player_type == 'skater':
        window = 80
        
    y_r = y.groupby(['playerId'], as_index=False)[PRED_COLS[player_type]].rolling(window, min_periods=1).mean()
    y_r = y_r.loc[X.index]
    
    # alternative windows for specific items
    if player_type == 'goalie':
        new_window = 75
        cols = ['so']
        
    if player_type == 'skater':
        new_window = 60
        cols = ['fow', 'hit', 'goalsfor','goalsaga']
        
    y_so = y.groupby(['playerId'], as_index=False)[cols].rolling(new_window, min_periods=1).mean().loc[X.index]
    y_r[cols] = y_so[cols]
    
    return X, y_r

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