import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from process_data import get_rest_of_season_player_stats, PRED_COLS
import pickle

def forward_feature_selection_with_elimination(X, y, col, tol=1e-4, cv=3):
    X = X.dropna()
    y = y.loc[X.index]
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

def load_player_feature_map(player_type, data, window=30, force_retrain=False):
    player_feature_map_file = f'data/{player_type}_features_{window}.json'
    try:
        with open(player_feature_map_file, 'r') as f:
            player_features_map = json.loads(f.read())
    except FileNotFoundError:
        print('File not found. Fitting...')
        force_retrain = True
    
    if force_retrain:
        X, y = data
        
        player_features_map = {}
        for col in y.columns:
            print(f'Fitting for {col}')
            player_features_map[col] = forward_feature_selection_with_elimination(X, y, col)
        with open(player_feature_map_file, 'w') as f:
            f.write(json.dumps(player_features_map))
    return player_features_map

def get_data_by_windows(player_type, windows=[], force_retrain=False, shift=True):
    Xs = []
    full_feature_map = {}
    for window in windows:
        if force_retrain:
            X, y = get_rest_of_season_player_stats(player_type, window=window, shift=shift)
        else:
            X, y = None, None
        
        feature_map = load_player_feature_map(player_type, (X, y), window=window, force_retrain=force_retrain)
            
        all_cat_cols = []
        original_cols = []
        for cat, values in feature_map.items():
            original_cols += [c for c in values[0] if c not in original_cols]
            cat_cols = [f'{c}_{window}' for c in values[0]]
            full_feature_map[cat] = full_feature_map.get(cat, []) + cat_cols
            all_cat_cols += cat_cols
            
        
        if not force_retrain:
            X, y = get_rest_of_season_player_stats(player_type, window=window, cols=original_cols)

        X = X[original_cols].copy()
        X.columns = [f'{c}_{window}' for c in original_cols]
        Xs.append(X)
        print('retrieved data for window', window)

    X = pd.concat(Xs, axis=1).dropna()
    return X, y, full_feature_map

def run_simple_training(player_features_map, data, player_type):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.metrics import r2_score, mean_absolute_error

    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y.loc[X.index], test_size=0.2, shuffle=False)

        
    pipes = {}
    for c in PRED_COLS[player_type]:
        
        print('fitting', c)
        lr = Pipeline([
            ('scl', StandardScaler()),
            ('reg', Ridge(alpha=1))
        ])
        pipe = lr
        pipe.fit(X_train[player_features_map[c]], y_train[c])
        preds = pipe.predict(X_test[player_features_map[c]])
        preds = np.clip(preds, 0, np.inf)
        score = r2_score(y_test[c], preds)
        print(c, '--', score)
        mae = mean_absolute_error(y_test[c], preds)
        print(c, '--', mae)
        
        # refit
        pipe.fit(X[player_features_map[c]], y[c])
        pipes[c] = pipe
    return pipes

def get_simple_pipelines(player_type, data, feature_map, force_retrain=False):
    model_files = {
        'skater':'data/models_skater3.pkl',
        'goalie':'data/models_goalie4.pkl',
    } 

    try:
        with open(model_files[player_type], 'rb') as f:
            pipes = pickle.load(f)
    except FileNotFoundError:
        force_retrain = True
        
    if force_retrain:
        pipes = run_simple_training(feature_map, data, player_type)
        with open(model_files[player_type], 'wb') as f:
            pickle.dump(pipes, f)
    
    return pipes
    