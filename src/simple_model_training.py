import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from process_data import get_player_stats, get_weekly_stats, PRED_COLS
import pickle

def forward_feature_selection_with_elimination(X, y, col, tol=1e-6, cv=3):
    
    n_features = X.shape[1]
    selected_features = []
    remaining_features = X.columns.tolist()
    best_score = 0
    lasso_cv = LinearRegression()
    
    while remaining_features:
        scores = []
        n = len(remaining_features)
        for i, feature in enumerate(remaining_features):
            # Try adding the current feature to the selected features
            features_to_try = selected_features + [feature]
            X_subset = X.loc[X.index.isin(y.index),features_to_try]

            # Use cross-validation to evaluate the model
            score = np.mean(cross_val_score(lasso_cv, X_subset, y.loc[X_subset.index, col], cv=cv))
            scores.append((score, feature))
            if i % 100 == 0:
                print(i, '/', n)

        # Sort features by score
        scores.sort(key=lambda x: x[0], reverse=True)

        best_scores = [s for s in scores if s[0] - best_score > tol]
        
        # If the score improves, add the best feature
        if len(best_scores) > 0:
            selected_features.append(best_scores[0][1])
            remaining_features = [f for _, f in best_scores[1:]]  # Keep top n_to_keep - 1 remaining features
            best_score = best_scores[0][0]
            print(f"Selected feature {best_scores[0][1]} with score {best_score}")
        else:
            # Stop if no improvement
            break
        
    return selected_features, best_score

def run_fselection(X, y):
    features_map = {}
    for col in y.columns:
        print(f'Fitting for {col}')
        features_map[col] = forward_feature_selection_with_elimination(X, y, col)
    return feature_map

import json

def load_player_feature_map(player_type):
    player_feature_map_file = f'data/{player_type}_player_features.json'
    try:
        with open(player_feature_map_file, 'r') as f:
            player_features_map = json.loads(f.read())
    except FileNotFoundError:
        print('File not found. Fitting...')
        X_p, y = get_player_stats(player_type)
        print('Data loaded')
        player_features_map = {}
        for col in y.columns:
            print(f'Fitting for {col}')
            player_features_map[col] = forward_feature_selection_with_elimination(X_p, y, col)
        with open(player_feature_map_file, 'w') as f:
            f.write(json.dumps(player_features_map))
    return player_features_map


def load_feature_maps(player_type='skater'):
    player_features_map = load_player_feature_map(player_type)
    feature_map_file = f'data/full_{player_type}_features.json'
    try:
        with open(feature_map_file, 'r') as f:
            full_features_map = json.loads(f.read())
    except FileNotFoundError:
        print('File not found. Fitting...')
        full_features_map = {}
        
        for col in PRED_COLS[player_type]:
            print(f'Fitting for {col}')
            X_p, y = get_weekly_stats(player_type, cols=player_features_map[col][0])       
            print('Loaded data')
            full_features_map[col] = forward_feature_selection_with_elimination(X_p, y, col)
        with open(feature_map_file, 'w') as f:
            f.write(json.dumps(full_features_map))
            
    return full_features_map, player_features_map




def run_training(player_type, full_features_map, player_features_map, schedule=None, retrain=False):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    import numpy as np
    from sklearn.metrics import r2_score, mean_absolute_error

    
    pipes = {}
    for c in PRED_COLS[player_type]:
        pipe = Pipeline([
            ('scl', StandardScaler()),
            ('reg', Lasso(alpha=3e-3))
        ])
        
        X, y = get_weekly_stats(player_type, cols=player_features_map[c][0], schedule=schedule)
        
        X = X[X.index.isin(y.index)].copy()
        y = y.loc[X.index].copy()

        if schedule is not None:
            min_date = min([g['ts'] for w in schedule.values() for g in w])
            X_dates = pd.to_datetime(X.index.get_level_values('gameDate'), format='%Y%m%d').date
            train_idx = X_dates < min_date
        else:
            train_idx = X.index.get_level_values('gameId').astype(int) < 2023020800

        X = X[full_features_map[c][0]].copy()
        pipe.fit(X.loc[train_idx], y[train_idx][c])
        preds = np.clip(pipe.predict(X[~train_idx]), 0, np.inf)
        score = r2_score(y[~train_idx][c], preds)
        print(c, '--', score)
        mae = mean_absolute_error(y[~train_idx][c], preds)
        print(c, '--', mae)
        if retrain:
            if schedule is not None:
                X, y = get_weekly_stats(player_type, cols=player_features_map[c][0], schedule=None)
                X = X.loc[X.index.isin(y.index), full_features_map[c][0]].copy()
                y = y.loc[X.index].copy()

            pipe.fit(X, y[c])
        pipes[c] = pipe
    return pipes

def get_pipelines():
    skater_models_file = 'data/models_skater.pkl'
    full_features_map, player_features_map = load_feature_maps('skater')
    try:
        with open(skater_models_file, 'rb') as f:
            pipes = pickle.load(f)
    except FileNotFoundError:
        pipes = run_training('skater', full_features_map, player_features_map, retrain=True)
    
    with open(skater_models_file, 'wb') as f:
        pickle.dump(pipes, f)
    
    goalie_features_map, g_player_features_map = load_feature_maps('goalie')
    
    goalie_models_file = 'data/models_goalie.pkl'
    try:
        with open(goalie_models_file, 'rb') as f:
            g_pipes = pickle.load(f)
    except FileNotFoundError:
        g_pipes = run_training('goalie', goalie_features_map, g_player_features_map, retrain=True)
    
    with open(goalie_models_file, 'wb') as f:
        pickle.dump(g_pipes, f)
    
    return {'goalie':g_pipes, 'skater':pipes}

