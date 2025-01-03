import pandas as pd

project_id = 'mehhala-fantasy'
# Read file from Google Cloud Storage
player_data = pd.read_csv('gs://{}.appspot.com/player_data.csv'.format(project_id))
team_data = pd.read_csv('gs://{}.appspot.com/team_data.csv'.format(project_id))

skater_data = player_data.loc[player_data.pos != "['G']", ['is_base', 'is_available', 'current_lineup','name','pos','rank','week_rank',
                              'week_games','g','a','sog','ppp', 'fow','hit','block','pim','plusmin','team', 'games']]
goalie_data = player_data.loc[player_data.pos == "['G']",['is_base', 'is_available', 'current_lineup','name','rank','week_rank','week_games', 
                                          'ga','win','so','save','team', 'games']]
team_season_data = team_data[['team','manager','g', 'a', 'ppp', 'sog', 'fow', 'hit', 'block', 'pim', 'plusmin',
       'ga', 'win', 'so', 'save']]
team_week_data = team_data[['team','manager','g_week', 'a_week', 'ppp_week', 'sog_week', 'fow_week',
       'hit_week', 'block_week', 'pim_week', 'plusmin_week', 'ga_week',
       'win_week', 'so_week', 'save_week']]
