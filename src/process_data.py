import pandas as pd


PRED_COLS = ['fow', 'goals', 'assists', 'shots', 'hits', 'blocks', 'pim', 'goalsfor', 'goalsaga', 'ppp']
INDEX_COLS = ['playerId','gameId','gameDate', 'playerTeam','opposingTeam']

def get_player_stats():

    X_p = pd.read_csv('data/gbgpca.csv')
    y = pd.read_csv('data/y.csv')
    X_p = X_p.set_index(INDEX_COLS)
    y = y.set_index(INDEX_COLS)

    component_cols = list(range(25))
    component_cols = [str(i) for i in component_cols]
    X_p[component_cols] = X_p.groupby('playerId')[component_cols].apply(lambda x: x.ewm(com=30).mean()).values
    
    return X_p, y

def get_team_stats(retrain=False):
    df = pd.read_csv('data/teams.csv')
    df = df.sort_values(['gameId', 'home_or_away'])
    df = df.pivot(index=['team', 'gameId', 'gameDate', 'home_or_away'], columns=['situation'], values=df.columns[11:])
    df['home'] = (df.index.get_level_values('home_or_away') == 'HOME').astype(int)
    df = df.reset_index(level=3, drop=True)

    pt = df.copy()

    pt[pt.columns] = pt.reset_index(level='team').groupby('team')[pt.columns].apply(lambda x: x.ewm(com=30).mean()).values

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    train_idx = pt.index.get_level_values('gameId').astype(int) // (10 ** 6) < 2023
    
    n_components = 15
    pipe = Pipeline([
        ('scl', StandardScaler()),
        ('pca', PCA(n_components=n_components))
    ])
    pipe.fit(pt[train_idx])
    pt = pd.DataFrame(pipe.transform(pt), pt.index)
    pt.columns = [str(c) for c in pt.columns]
    pt['home'] = df['home']
    return pt

def get_bios(player_stats):
    df = pd.read_csv('data/bios.csv')
    df.birthDate = pd.to_datetime(df.birthDate)
    df = df.merge(player_stats.reset_index().iloc[:,:5], how='left', on='playerId').set_index(INDEX_COLS)
    df['gameDate'] = pd.to_datetime(df.index.get_level_values('gameDate'), format='%Y%m%d').values
    df['age'] = ((df.gameDate - df.birthDate).dt.days / 365)
    df = df.join(pd.get_dummies(df[['position', 'shootsCatches']]).set_index(df.index))
    ht = df['height'].str.split('\'|"')
    df['ht'] = ht.str[0].astype(float) * 12 + ht.str[1].astype(float)

    ecols = ['weight', 'age', 'position_C', 'position_D', 'position_G',
    'position_L', 'position_R', 'shootsCatches_L', 'shootsCatches_R', 'ht']
    X_b = df[ecols]
    return X_b

def run_training(X, y, retrain=False):
    from sklearn.ensemble import HistGradientBoostingRegressor
    import numpy as np
    from sklearn.metrics import r2_score, mean_absolute_error

    train_idx = X.index.get_level_values('gameId').astype(int) // (10 ** 6) < 2023

    y = y.loc[X.index,:]
    pipes = {}
    for c in PRED_COLS:
        pipe = HistGradientBoostingRegressor()
        pipe.fit(X[train_idx], y[train_idx][c])

        preds = np.clip(pipe.predict(X[~train_idx]), 0, np.inf)
        score = r2_score(y[~train_idx][c], preds)
        print(c, '--', score)
        mae = mean_absolute_error(y[~train_idx][c], preds)
        print(c, '--', mae)
        if retrain:
            pipe.fit(X, y)
        pipes[c] = pipe
    return pipes
