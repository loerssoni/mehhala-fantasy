def main():
    from get_data import load_history, load_current, combine_history, process_y, load_bios, load_team_data

    # load_history()

    load_current()
    combine_history()
    load_team_data()
    load_bios()
    process_y()

    import pandas as pd
    from process_data import get_rest_of_season_player_stats, get_rest_of_season_stats, PRED_COLS
    from model_training import get_simple_pipelines, load_player_feature_map


    X, y = get_rest_of_season_player_stats('skater')
    skater_latest = X.groupby('playerId').last()


    preds = {}
    skaters_p_feats = load_player_feature_map('skater', data=(X,y))
    sk_p_cols = list(set([c for s in skaters_p_feats.values() for c in s[0]]))


    X_g, y_g = get_rest_of_season_player_stats('goalie')
    goalie_latest = X_g.groupby('playerId').last()

    goalie_preds = {}
    goalies_p_feats = load_player_feature_map('goalie', (X_g, y_g))

    pipelines = get_simple_pipelines((X, y), (X_g, y_g))

    preds = {}
    goalie_preds = {}

    for col in PRED_COLS['skater']:    
        preds[col] = pipelines['skater'][col].predict(skater_latest.dropna()[skaters_p_feats[col][0]])
    preds_df = pd.DataFrame(preds, index=skater_latest.dropna().index)

    gl_p_cols = list(set([c for s in goalies_p_feats.values() for c in s[0]]))
    for col in PRED_COLS['goalie']:
        goalie_preds[col] = pipelines['goalie'][col].predict(goalie_latest.dropna()[goalies_p_feats[col][0]])
    goalie_preds_df = pd.DataFrame(goalie_preds, index=goalie_latest.dropna().index)

    preds = pd.concat([preds_df, goalie_preds_df], axis=0)
    preds['plusmin'] = preds['goalsfor'] - preds['goalsaga']
    preds['ga'] = -preds['ga'] / preds['icetime']


    import yahoo_utils
    import lineup_utils
    from process_data import PRED_COLS

    from ast import literal_eval
    import pandas as pd

    game_id = 453
    players = yahoo_utils.get_players(game_id)
    players = players[~players.playerId.duplicated()]
    player_info = players.set_index('playerId')[['name','pos','team']]
    players['team_yh'] = players.team_yh.replace({
        'SJ':'SJS',
        'LA':'LAK',
        'TB':'TBL',
        'NJ':'NJD'
    })
    current_schedule = yahoo_utils.get_games_by_week(game_id)
    teams = yahoo_utils.get_teams(game_id, True)

    q = yahoo_utils.get_q(game_id)
    info = []
    for team in teams.team_id.drop_duplicates():
        info.append(q.get_team_info(team.split('.')[-1]))


    current_team = [t for t in info if t.name.decode('utf8')=='Kiitos ryhmään pääsystä'][0]
    position_lookup = players.set_index(['playerId'], drop=False).pos.apply(literal_eval).to_dict()
    metrics = [m for metrics in PRED_COLS.values() for m in metrics]



    games_list = []
    for k, v in current_schedule.items():
        for r in v:
            a = {'week':k}
            a.update(r)
            games_list.append(a)
    games_df = pd.DataFrame(games_list)
    games_df['ts'] = pd.to_datetime(games_df['ts'])
    player_games = pd.concat([
        games_df.merge(players, how='left', left_on='home', right_on='team_yh'),
        games_df.merge(players, how='left', left_on='away', right_on='team_yh')
    ])[['week','gameId','ts','pos','name','playerId']]
    player_games = player_games.dropna()



    date_now = pd.Timestamp.now().date()
    cats = ['g','a','sog','fow','hit','block','pim','plusmin','ga','win','so','save']
    m = [m for m in current_team.matchups if pd.to_datetime(m.week_end).date() >= date_now + pd.Timedelta('2d')][0]
    print(m.week)
    opponent_id = [t.team_key for t in m.teams if t.team_key != current_team.team_key][0]
    dates = pd.date_range(max(pd.to_datetime(m.week_start).date(), date_now), m.week_end)


    ranks = preds.copy()
    ranks['plusmin'] = ranks['goalsfor'] - ranks['goalsaga']
    ranks['ga'] = -ranks['ga'] / ranks['icetime']
    ranks = ranks.drop('icetime', axis=1)
    ranks = ((ranks - ranks.mean())/(ranks.std())).mean(1).rank(pct=True)
    ranks.name = 'rank'

    week_teams = teams.loc[(pd.to_datetime(teams.index) >= m.week_start)&(pd.to_datetime(teams.index) <= m.week_end)]

    ir = [p.player_key for t in info for p in t.roster.players if p.selected_position.date == date_now.strftime('%Y-%m-%d') and 'IR' in p.selected_position.position]
    ir = []
    current_lineup = teams[(teams.team_id == current_team.team_key)&(teams.index.get_level_values('date') == date_now.strftime('%Y-%m-%d'))&(~teams.player_key.isin(ir))]
    current_lineup = current_lineup.merge(players, how='left', on='player_key').playerId.tolist()
    current_lineup.remove(8478445)
    current_lineup.append(8477935)

    selected_team = []

    for date in dates:
        rankings = []
        print('\n\n\n', date.date())
        if len(selected_team) > 0:
        #     current_lineup = [p for p in selected_team] # this is inactive since we use our existing lineup as starting point for each day of the week
            selected_team = []

        starting_teams = teams.loc[(pd.to_datetime(teams.index) == date)]
        all_available_players = players[(~players.player_key.isin(starting_teams.player_key))|(players.playerId.isin(current_lineup))]
        all_available_players = all_available_players.playerId.tolist()

        week_games = player_games[(player_games.ts >= date)&(player_games.ts <= m.week_end)]



        while len(selected_team) < 35:
            print(str(len(selected_team)), end='\r')

            if len(selected_team) < 14:
                available = [p for p in current_lineup if p not in selected_team]

            else:
                available = [p for p in all_available_players if p not in selected_team]

            rest_games = lineup_utils.get_rest_of_season_games(date, player_games, selected_team, position_lookup)
            stats_available = rest_games[rest_games.index.isin(preds.index)].index
            lineup_preds = preds.loc[stats_available, cats].apply(lambda x: x * rest_games[stats_available])
            preds_st = ((lineup_preds - lineup_preds.mean())/(lineup_preds.std()))
            ranks = preds_st.mean(1).rank(pct=True)
            ranks.name = 'rank'

            week_rest_games = lineup_utils.get_rest_of_season_games(date, week_games, selected_team, position_lookup)
            week_stats_available = week_rest_games[week_rest_games.index.isin(preds.index)].index
            week_lineup_preds = preds.loc[stats_available, cats].apply(lambda x: x * week_rest_games[week_stats_available])
            week_preds_st = ((week_lineup_preds - week_lineup_preds.mean())/(week_lineup_preds.std()))
            week_ranks = week_preds_st.mean(1).rank(pct=True)
            week_ranks.name = 'week_rank'

            if len(selected_team) < 14:
                selected_player = ranks.loc[[p for p in available if p in ranks]].idxmax()
                selected_team.append(selected_player)
                rankings.append({'playerId':selected_player, 'rank': round(ranks.loc[selected_player], 3), 'week_rank': round(week_ranks.loc[selected_player], 3), 'games':rest_games.loc[selected_player], 'week_games':week_rest_games.loc[selected_player]})

            else:
                rest_of_them = ranks[[p for p in available if p in stats_available]].sort_values(ascending=False).iloc[:(35-len(selected_team))].index.tolist()
                for p in rest_of_them:
                    rankings.append({'playerId':p, 'rank': round(ranks.loc[p], 3), 'week_rank': round(week_ranks.loc[p], 3), 'games':rest_games.loc[p], 'week_games':week_rest_games.loc[p]})
                selected_team += rest_of_them

        for p in current_lineup:
            if p not in selected_team:
                rankings.append({'playerId':p, 'rank': round(ranks.loc[p], 3), 'week_rank': round(week_ranks.loc[p], 3), 'games':rest_games.loc[p], 'week_games':week_rest_games.loc[p]})
        rankings = pd.DataFrame(rankings).set_index('playerId')

        n_games = week_rest_games[week_rest_games.index.isin(preds.index)]
        n_games.name = 'n_games'
        print(player_info.loc[current_lineup])
        print('DROPS')
        for p in current_lineup:
            if p not in selected_team:

                print(player_info.join(n_games).join(rankings).loc[p].to_dict())
        print('ADDS')
        for p in selected_team:
            if p not in current_lineup:
                print(player_info.join(n_games).join(rankings).loc[p].to_dict())

        break

    output = player_info.join(rankings, how='inner').sort_values(['rank'], ascending=False).join(preds_st.round(3)).join(week_preds_st.round(3), rsuffix='_week')
    output['current_lineup'] = output.index.isin(current_lineup)
    output['selection'] = output.index.isin(selected_team)
    output = output.sort_values(['current_lineup', 'rank','week_rank'], ascending=False)
    output.to_csv('data/player_data.csv')

    ranks = preds.copy()
    ranks = ranks.drop('icetime', axis=1)
    # ranks = ((ranks - ranks.mean())/(ranks.std())).mean(1).rank(pct=True)
    full_n_games = player_games[player_games.ts.dt.date > date_now].groupby('playerId').gameId.count()
    full_n_games = pd.DataFrame(ranks).join(full_n_games).fillna(0).gameId
    week_n_games = player_games[(player_games.ts.dt.date > date_now)&(player_games.ts <= m.week_end)].groupby('playerId').gameId.count()
    week_n_games = pd.DataFrame(ranks).join(week_n_games).fillna(0).gameId

    week_ranks = ranks[cats].apply(lambda x: x * week_n_games)
    ranks = ranks[cats].apply(lambda x: x * full_n_games)
    ranks.name = 'rank'
    week_ranks.name ='week_rank'

    prankings = players.join(ranks, how='inner', on='playerId').join(week_ranks, on='playerId', rsuffix='_week').merge(starting_teams, how='left', on='player_key').set_index('playerId')

    ss = []
    for t in info:
        a = prankings.loc[prankings.team_id == t.team_key].copy()
        a['team'] = t.name.decode()
        a['manager'] = t.managers[0].nickname
        ss.append(a)

    ss = pd.concat(ss)
    ss = ss.groupby(['team','manager'])[cats+[c+'_week' for c in cats]].mean().round(2)
    ss['matchup'] = ss.index.get_level_values('team').isin([t.name.decode() for t in m.teams])
    ss['own'] = ss.index.get_level_values('team') == current_team.name.decode()
    ss = ss.sort_values(['own','matchup'], ascending=False)
    ss.to_csv('data/team_data.csv')

if __name__ == '__main__':
    main()
