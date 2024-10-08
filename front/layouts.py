# import dash-core, dash-html, dash io, bootstrap
import os

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Dash Bootstrap components
import dash_bootstrap_components as dbc

# Import app
from app import app

# Import server for deployment
from app import srv as server

from data import skater_data, goalie_data, team_season_data, team_week_data


app_name = os.getenv("DASH_APP_PATH", "/mehhala-fantasy")

# Layout variables, navbar, header, content, and container
nav = Navbar()

# Layout for Team Analysis page
tableLayout = html.Div(
    [
        dbc.Row(dbc.Col(html.H3(children="Data"))),
        # Display Championship titles in datatable
        dbc.Row(
            dbc.Col(
                html.Div(id="team-data"),
                xs={"size": "auto", "offset": 0},
                sm={"size": "auto", "offset": 0},
                md={"size": 7, "offset": 0},
                lg={"size": "auto", "offset": 0},
                xl={"size": "auto", "offset": 0},
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

    


@callback(Output("team-data", "children"),
          Input('page-selection', 'value'))
def render_content(tab):
    if tab == 'skater_data':
        return skater_data
    if tab == 'goalie_data':
        return goalie_data
    if tab == 'team_season_data':
        return team_season_data
    if tab == 'team_week_data':
        return team_week_data



# Main index function that will call and return all layout variables
def index():
    layout = html.Div([nav, container])
    return layout


# Set layout to index function
app.layout = index()

# Call app server
if __name__ == "__main__":
    # set debug to false when deploying app
    app.run_server(debug=True)

