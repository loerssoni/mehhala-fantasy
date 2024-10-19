import os

from dash.dependencies import Input, Output
from dash import dcc, html
# Dash Bootstrap components
import dash_bootstrap_components as dbc

# Import app
from app import app

# Import server for deployment
from app import srv as server

import dash_table
import dash_ag_grid as dag
from data import skater_data, goalie_data, team_season_data, team_week_data

import pandas as pd

app_name = os.getenv("DASH_APP_PATH", "/mehhala-fantasy")


# Layout for Team Analysis page
tableLayout = html.Div(
    [

        dbc.Row(
            dbc.Col(
                html.Div(id="team-data")
            ),
            justify="center",
        )
    ],
    className="app-page",
)

tabs = html.Div([
    dcc.Tabs(id="page-selection", value='tab-1-example-graph', children=[
        dcc.Tab(label='Skater predictions', value='skater_data'),
        dcc.Tab(label='Goalie predictions', value='goalie_data'),
        dcc.Tab(label='Team season analysis', value='team_season_data'),
        dcc.Tab(label='Team week analysis', value='team_week_data'),
    ]),
    tableLayout
])

# Div to hide/show the checkbox (hidden by default)
checkbox = html.Div(id="checkbox-container", children=[
        dcc.Checklist(
            id='available-checkbox',
            options=[{'label': 'Only Show Available', 'value': 'available'}],
            value=[],  # Initially unchecked
            style={'marginBottom': 20}
        )
    ], style={'display': 'none'})  # Hide by default


header = dbc.Row(
    dbc.Col(
        html.Div(
            [
                html.H2(children="Mehhala-fantasy")
            ]
        )
    ),
    className="banner",
)

container = dbc.Container([header, tabs, checkbox])

@app.callback(
    Output("team-data", "children"),
    [Input('page-selection', 'value'),
     Input('available-checkbox', 'value')]  # Capture the state of the checkbox
)
def render_content(tab, available_filter):
    data_asset = pd.DataFrame({})
    
    # Load data based on the selected tab
    if tab == 'skater_data':
        data_asset = skater_data
    elif tab == 'goalie_data':
        data_asset = goalie_data
    elif tab == 'team_season_data':
        data_asset = team_season_data
    elif tab == 'team_week_data':
        data_asset = team_week_data
    
    # Apply filter if 'is_available' checkbox is checked and tab is skater_data or goalie_data
    if tab in ['skater_data', 'goalie_data'] and 'available' in available_filter:
        data_asset = data_asset[data_asset['is_available'] == True]

    # Return the table layout
    return html.Div(
            [dag.AgGrid(
                rowData=data_asset.to_dict("records"),
                columnDefs=[{"field": col} for col in data_asset.columns],
                columnSize='autoSize'
            )]
        )

@app.callback(
    Output("checkbox-container", "style"),
    Input('page-selection', 'value')
)
def toggle_checkbox_visibility(tab):
    # Show the checkbox only for skater_data or goalie_data tabs
    if tab in ['skater_data', 'goalie_data']:
        return {'display': 'block'}  # Show checkbox
    else:
        return {'display': 'none'}  # Hide checkbox


# Main index function that will call and return all layout variables
def index():
    layout = html.Div([container])
    return layout


# Set layout to index function
app.layout = index()

# Call app server
if __name__ == "__main__":
    # set debug to false when deploying app
    app.run_server(debug=True)

