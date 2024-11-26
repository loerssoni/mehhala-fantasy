import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from process_data import get_rest_of_season_player_stats, PRED_COLS
import pickle
import logging
from sklearn.model_selection import KFold

def forward_feature_selection_with_elimination(X, y, col, tol=1e-4, cv=3):

    non_nan = X.notna().all(axis=1) & y[col].notna()
    X = X.loc[non_nan]
    y = y.loc[X.index]
    
    n_features = X.shape[1]
    selected_features = []
    remaining_features = X.columns.tolist()
    best_score = 0
    lasso_cv = LinearRegression()
    
    cvs = KFold(n_splits=cv, shuffle=False)
    while remaining_features:
        scores = []
        n = len(remaining_features)
        for i, feature in enumerate(remaining_features):
            # Try adding the current feature to the selected features
            features_to_try = selected_features + [feature]
            X_subset = X.loc[X.index.isin(y.index),features_to_try]

            # Use cross-validation to evaluate the model
            score = np.mean(cross_val_score(lasso_cv, X_subset, y.loc[X_subset.index, col], cv=cvs))
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
            logging.info(f"Selected feature {best_scores[0][1]} with score {best_score}")
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
        logging.info('File not found. Fitting...')
        force_retrain = True
    
    if force_retrain:
        X, y = data
        
        player_features_map = {}
        for col in PRED_COLS[player_type]:
            logging.info(f'Fitting for {col}')
            player_features_map[col] = forward_feature_selection_with_elimination(X, y, col)
        with open(player_feature_map_file, 'w') as f:
            f.write(json.dumps(player_features_map))
    return player_features_map

def get_data_by_windows(player_type, windows=[], force_retrain=False, shift=True):
    Xs = []
    full_feature_map = {}
    for window in windows:
        if force_retrain:
            X, y, y_std = get_rest_of_season_player_stats(player_type, window=window, shift=shift)
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
            X, y, y_std = get_rest_of_season_player_stats(player_type, window=window, cols=original_cols, shift=shift)
        original_cols = X.columns
        X = X[original_cols].copy()
        X.columns = [f'{c}_{window}' for c in original_cols]
        Xs.append(X)
        logging.info(f'retrieved data for window  {window}')

    X = pd.concat(Xs, axis=1).dropna()
    X['dummyvar'] = 1
    return X, y, y_std,full_feature_map


# Define transformation functions
def log_tr(x):
    return np.log(x + 1)
def exp_tr(x):
    return np.exp(x) - 1

from sklearn.preprocessing import PowerTransformer

transformers = {
    'ga':{'func':log_tr, 'inverse_func':exp_tr},
    'win':{},
    'so':{'func':log_tr, 'inverse_func':exp_tr},
    'save':{'func':log_tr, 'inverse_func':exp_tr},
    'icetime':{},
    'g':{'func':log_tr, 'inverse_func':exp_tr},
    'a':{'func':log_tr, 'inverse_func':exp_tr},
    'sog':{},
    'fow':{},
    'hit':{},
    'block':{},
    'pim':{},
    'goalsfor':{},
    'goalsaga':{},
    'ppp':{'func':log_tr, 'inverse_func':exp_tr},
}

def run_simple_training(player_features_map, data, player_type):
    from sklearn.linear_model import RidgeCV, LinearRegression
    from sklearn.model_selection import KFold 
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.metrics import r2_score, mean_absolute_error

    import numpy as np

    X, y = data
    y = y.dropna()
    X = X.loc[y.index]
    X['dummyvar'] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y.loc[X.index], test_size=0.2, shuffle=True)

    pipes = {}
    for c in PRED_COLS[player_type]:
        if len(player_features_map[c]) == 0:
            player_features_map[c] = ['dummyvar']

        print('fitting ' + c)
        lr = Pipeline([
            ('scl', StandardScaler()),
            ('reg', (TransformedTargetRegressor(RidgeCV(cv=KFold(n_splits=3, shuffle=False), alphas=[1e-4, 1e-2, 1e-1, 10]), **transformers.get(c, {}))))
            # ('reg', LinearRegression()),
        ])
        
        pipe = lr
        pipe.fit(X_train[player_features_map[c]], y_train[c])
        preds = pipe.predict(X_test[player_features_map[c]])
        preds = np.clip(preds, 0, np.inf)
        score = r2_score(y_test[c], preds)
        print(f'{c}  --  {score}')
        mae = mean_absolute_error(y_test[c], preds)
        print(f'{c}  --  {mae}')

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
    except (FileNotFoundError, EOFError):
        force_retrain = True
        
    if force_retrain:
        pipes = run_simple_training(feature_map, data, player_type)
        with open(model_files[player_type], 'wb') as f:
            pickle.dump(pipes, f)
    
    return pipes


def get_latest_predictions(player_type, windows):
    X, y, y_std, feature_map = get_data_by_windows(player_type, windows, force_retrain=False, shift=False)
    X_latest = X.groupby('playerId').last()
    X = X.groupby('playerId').shift(1).dropna()
    y = y.loc[X.index]

    logging.info('loaded data')
    pipelines = get_simple_pipelines(player_type, (X, y), feature_map, force_retrain=True)
    logging.info('trained pipelines')
    preds = {}

    for col in PRED_COLS[player_type]:
        cols = feature_map[col]
        if len(cols) == 0:
            cols = ['dummyvar']
        preds[col] = pipelines[col].predict(X_latest.dropna()[cols])
    preds_df = pd.DataFrame(preds, index=X_latest.dropna().index)
    return preds_df, y_std

def make_predictions():
    skater_preds, skater_std = get_latest_predictions('skater', [30, 15, 10, 5, 3])
    logging.info('made skater preds')
    logging.info(skater_preds.shape)

    goalie_preds, goalie_std = get_latest_predictions('goalie', [50, 30, 20, 15, 10, 8, 3, 1])
    logging.info('made goalie preds')
    logging.info(goalie_preds.shape)

    player_stds = pd.concat([skater_std, goalie_std])
    player_stds.to_csv('data/player_stds.csv')

    preds = pd.concat([skater_preds, goalie_preds], axis=0)
    preds['plusmin'] = preds['goalsfor'] - preds['goalsaga']

    preds.icetime = (preds.icetime-preds.icetime.min())/(preds.icetime.max()-preds.icetime.min())
    preds.so = preds.so * preds.icetime
    preds.win = preds.win * preds.icetime
    preds.save = preds.save * preds.icetime
    preds.ga = 1 / preds.ga
    
    preds.to_csv('data/latest_preds.csv')
    logging.info('preds processed.' ) 
    logging.info(preds.shape)
    
def load_predictions():
    preds = pd.read_csv('data/latest_preds.csv', index_col=0)
    player_stds = pd.read_csv('data/player_stds.csv', index_col=0)
    return preds, player_stds['0']

    
