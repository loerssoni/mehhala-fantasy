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
table_layout = html.Div(
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

# Div to hide/show the checkbox (hidden by default)
checkbox = html.Div(id="checkbox-container", children=[
        dcc.Checklist(
            id='available-checkbox',
            options=[{'label': 'Only Show Available', 'value': 'available'}],
            value=[],  # Initially unchecked
            style={'marginBottom': 20}
        )
    ], style={'display': 'none'})  # Hide by default

tabs = html.Div([
    dcc.Tabs(id="page-selection", value='tab-1-example-graph', children=[
        dcc.Tab(label='Skater predictions', value='skater_data'),
        dcc.Tab(label='Goalie predictions', value='goalie_data'),
        dcc.Tab(label='Team season analysis', value='team_season_data'),
        dcc.Tab(label='Team week analysis', value='team_week_data'),
    ]),
    checkbox, 
    table_layout
])

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

container = dbc.Container([header, tabs])

assets_to_render = {
    'skater_data': skater_data,
    'goalie_data': goalie_data,
    'team_season_data': team_season_data, 
    'team_week_data': team_week_data,
    'available_skaters': skater_data[skater_data['is_available'] == True],
    'available_goalies': goalie_data[goalie_data['is_available'] == True]
}
rendered_assets = {}
for k, data_asset in assets_to_render.items():
    rendered_assets[k] = html.Div(
            [dag.AgGrid(
                rowData=data_asset.to_dict("records"),
                columnDefs=[{"field": col} for col in data_asset.columns],
                columnSize='autoSize', 
                columnSizeOptions={
                'defaultMinWidth': 50
                }
            )]
    )

@app.callback(
    Output("team-data", "children"),
    [Input('page-selection', 'value'),
     Input('available-checkbox', 'value')]  # Capture the state of the checkbox
)
def render_content(tab, available_filter):
    content = html.Div()
    
    # Apply filter if 'is_available' checkbox is checked and tab is skater_data or goalie_data
    if tab in ['skater_data', 'goalie_data'] and 'available' in available_filter:
        if tab == 'skater_data':
            content = rendered_assets['available_skaters'] 
        else:
            content = rendered_assets['available_goalies']
    else:
        content = rendered_assets.get(tab, html.Div())
    # Return the table layout
    return content

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

