
import streamlit as st
import numpy as np
import pandas as pd

st.subheader("Mehhala Fantasy")
player_data = pd.read_csv('player_data.csv')
team_data = pd.read_csv('team_data.csv')

tab1, tab2 = st.tabs(["Player predictions", "Team comparison"])
with tab1:

    st.dataframe(player_data)
with tab2:
    st.dataframe(team_data)
