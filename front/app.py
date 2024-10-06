
import streamlit as st
import numpy as np
import pandas as pd
import os

project_id = os.environ['GOOGLE_CLOUD_PROJECT']

# Read file from Google Cloud Storage
player_data = pd.read_csv('gs://{}.appspot.com/player_data.csv'.format(project_id))
team_data = pd.read_csv('gs://{}.appspot.com/team_data.csv'.format(project_id))

st.subheader("Mehhala Fantasy")

tab1, tab2 = st.tabs(["Player predictions", "Team comparison"])
with tab1:

    st.dataframe(player_data)
with tab2:
    st.dataframe(team_data)
