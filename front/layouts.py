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
from data import skater_data, goalie_data, team_season_data, team_week_data


app_name = os.getenv("DASH_APP_PATH", "/mehhala-fantasy")


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

@app.callback(Output("team-data", "children"),
          Input('page-selection', 'value'))
def render_content(tab):
    if tab == 'skater_data':
        data_asset = skater_data
    elif tab == 'goalie_data':
        data_asset = goalie_data
    elif tab == 'team_season_data':
        data_asset = team_season_data
    elif tab == 'team_week_data':
        data_asset = team_week_data
    return html.Div(
            dash_table.DataTable(
                data=data_asset.to_dict("records"),
                columns=[{"name": col, "id": col} for col in data_asset.columns],
                style_as_list_view=True,
                editable=False,
                style_table={
                    "overflowY": "scroll",
                    "width": "100%",
                    "minWidth": "100%",
                },
                style_header={"backgroundColor": "#f8f5f0", "fontWeight": "bold"},
                style_cell={"textAlign": "center", "padding": "8px"},
            )
        )


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

