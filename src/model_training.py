import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from process_data import get_rest_of_season_player_stats, PRED_COLS
import pickle

def forward_feature_selection_with_elimination(X, y, col, tol=1e-4, cv=3):
    
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

def load_player_feature_map(player_type, data=None):
    player_feature_map_file = f'data/{player_type}_player_features2.json'
    try:
        with open(player_feature_map_file, 'r') as f:
            player_features_map = json.loads(f.read())
    except FileNotFoundError:
        print('File not found. Fitting...')
        if data is None:
            X_p, y = get_rest_of_season_player_stats(player_type)
        else:
            X_p, y = data
        print('Data loaded')
        player_features_map = {}
        for col in y.columns:
            print(f'Fitting for {col}')
            player_features_map[col] = forward_feature_selection_with_elimination(X_p, y, col)
        with open(player_feature_map_file, 'w') as f:
            f.write(json.dumps(player_features_map))
    return player_features_map

def run_simple_training(player_features_map, data=None, player_type=None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    import numpy as np
    from sklearn.metrics import r2_score, mean_absolute_error

    if data is not None:
        X, y = data
    else:
        X, y = get_rest_of_season_player_stats(player_type, cols=player_features_map[c][0])
        
    pipes = {}
    for c in PRED_COLS[player_type]:
        pipe = Pipeline([
            ('scl', StandardScaler()),
            ('reg', Lasso(alpha=3e-3))
        ])
                
        X_tr = X[X.index.isin(y.index)].copy()
        y_tr = y.loc[X.index].copy()
        X_tr = X_tr[player_features_map[c][0]].copy()
        pipe.fit(X_tr, y_tr[c])
        preds = pipe.predict(X_tr)
        score = r2_score(y_tr[c], preds)
        print(c, '--', score)
        mae = mean_absolute_error(y[c], preds)
        print(c, '--', mae)
        pipes[c] = pipe
    return pipes

def get_simple_pipelines(data_p=None, data_g=None):
    skater_models_file = 'data/models_skater3.pkl'
    player_features_map = load_player_feature_map('skater', data=data_p)
    try:
        with open(skater_models_file, 'rb') as f:
            pipes = pickle.load(f)
    except FileNotFoundError:
        pipes = run_simple_training(player_features_map, data=data_p, player_type='skater')
    
    with open(skater_models_file, 'wb') as f:
        pickle.dump(pipes, f)
    
    g_player_features_map = load_player_feature_map('goalie', data=data_g)
    
    goalie_models_file = 'data/models_goalie3.pkl'
    try:
        with open(goalie_models_file, 'rb') as f:
            g_pipes = pickle.load(f)
    except FileNotFoundError:
        g_pipes =  run_simple_training(g_player_features_map, data=data_g, player_type='goalie')
        
    with open(goalie_models_file, 'wb') as f:
        pickle.dump(g_pipes, f)
    
    return {'goalie':g_pipes, 'skater':pipes}
    