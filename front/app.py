
import streamlit as st
import numpy as np
import pandas as pd
import os

project_id = os.environ['GOOGLE_CLOUD_PROJECT']

# Read file from Google Cloud Storage
player_data = pd.read_csv('gs://{}.appspot.com/player_data.csv'.format(project_id))
team_data = pd.read_csv('gs://{}.appspot.com/team_data.csv'.format(project_id))

st.subheader("Mehhala Fantasy")

tab1, tab2, tab3, tab4  = st.tabs(["Player predictions", "Goalie predictions", "Team comparison", "Team week comparison"])
with tab1:
    st.dataframe(player_data.loc[player_data.pos != "['G']"], column_order=['current_lineup','name','pos','rank','week_rank',
                              'week_games','g','a','sog','fow','hit','block','pim','plusmin','team'],hide_index=True)
with tab2:
    st.dataframe(player_data.loc[player_data.pos == "['G']"], column_order=['current_lineup','name','rank','week_rank','week_games', 
                                          'ga','win','so','save','team'],hide_index=True)

with tab3:
    st.dataframe(team_data, column_order=['team','manager','g', 'a', 'sog', 'fow', 'hit', 'block', 'pim', 'plusmin',
       'ga', 'win', 'so', 'save'],hide_index=True)

with tab4:
    st.dataframe(team_data, column_order=['team','manager','g_week', 'a_week', 'sog_week', 'fow_week',
       'hit_week', 'block_week', 'pim_week', 'plusmin_week', 'ga_week',
       'win_week', 'so_week', 'save_week'], hide_index=True)
