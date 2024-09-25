import pandas as pd


SKATERS_PRED = ['g', 'a', 'sog', 'fow', 'hit', 'block', 'pim', 'goalsfor', 'goalsaga', 'ppp']
GOALIES_PRED = ['ga','win','so', 'save', 'icetime']
PRED_COLS = {
    'skater': SKATERS_PRED,
    'goalie': GOALIES_PRED
}

def get_player_stats(player_type, cols=None):
    
    filenames = {
        'skater':('data/games_p.h5','data/y.h5'),
        'goalie':('data/games_g.h5','data/y_g.h5')
    } 
    
    x_f, y_f = filenames[player_type]
    X = pd.read_hdf(x_f)
    X = X.sort_index(level=1)
    if cols is not None:
        X = X[cols].copy()
    X = X.groupby('playerId', as_index=False).rolling(30).mean().drop('playerId', axis=1)
    X = X.groupby('playerId').shift(1)
    X = X.dropna()

    y = pd.read_hdf(y_f)
    y = y.loc[X.index].copy()
    return X, y



def get_team_stats():
    df = pd.read_hdf('data/teams.h5')
    df = df.sort_values(['gameId', 'home_or_away'])
    df = df.pivot(index=['team', 'gameId', 'home_or_away'], columns=['situation'], values=df.columns[11:])
    df['home'] = (df.index.get_level_values('home_or_away') == 'HOME').astype(int)
    df = df.reset_index(level=2, drop=True)
    df = df.set_index(df.index.rename(['playerTeam','gameId']))
    df.columns = [f'{s[1]}_{s[0]}' for s in df.columns]
    df = df.sort_values(['playerTeam','gameId'])

    df = df.groupby('playerTeam', as_index=False).rolling(10, min_periods=5).mean().drop('playerTeam', axis=1)
    df = df.groupby('playerTeam').shift(1)
    return df

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

def get_full_stats(player_type, cols):
    X_p, y = get_player_stats(player_type, cols=cols)
    team_stats = get_team_stats()
    X_p = X_p.merge(team_stats, on=['gameId','playerTeam'], how='left').set_index(X_p.index)
    bios = get_bios()
    X_p = X_p.merge(bios, on='playerId', how='left').set_index(X_p.index)
    X_proj = get_projections()
    X_p['season'] = X_p.index.get_level_values(1) // (1000*1000)
    X_p = X_p.merge(X_proj, left_on=['name', 'season'], right_on=['player', 'season'], how='left').set_index(X_p.index)
    X_p = X_p.loc[:,~X_p.isna().all()].copy()
    X_p = X_p.fillna(0)
    X_p = X_p.drop(['name','player','playerId'], axis=1)
    y = y.loc[X_p.index,:].copy()
    return X_p, y


