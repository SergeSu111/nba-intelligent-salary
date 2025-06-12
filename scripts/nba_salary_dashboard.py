import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import joblib


random_forest_pipeline = joblib.load("nba_salary_rf_model.pkl")

team_mapping = {
    "LAL": "Los Angeles Lakers",
    "PHO": "Phoenix Suns",
    "DAL": "Dallas Mavericks",
    "WSB": "Washington Bullets",
    "BOS": "Boston Celtics",
    "HOU": "Houston Rockets",
    "SAC": "Sacramento Kings",
    "DEN": "Denver Nuggets",
    "CHI": "Chicago Bulls",
    "IND": "Indiana Pacers",
    "ORL": "Orlando Magic",
    "NOH": "New Orleans Hornets",
    "TOR": "Toronto Raptors",
    "CHO": "Charlotte Hornets",
    "POR": "Portland Trail Blazers",
    "DET": "Detroit Pistons",
    "PHI": "Philadelphia 76ers",
    "VAN": "Vancouver Grizzlies",
    "SEA": "Seattle SuperSonics",
    "WAS": "Washington Wizards",
    "NJN": "New Jersey Nets",
    "NOK": "New Orleans/Oklahoma City Hornets",
    "LAC": "Los Angeles Clippers",
    "OKC": "Oklahoma City Thunder",
    "MIL": "Milwaukee Bucks",
    "ATL": "Atlanta Hawks",
    "GSW": "Golden State Warriors",
    "CHA": "Charlotte Bobcats",
    "UTA": "Utah Jazz",
    "MEM": "Memphis Grizzlies",
    "MIN": "Minnesota Timberwolves",
    "NYK": "New York Knicks",
    "NOP": "New Orleans Pelicans",
    "BRK": "Brooklyn Nets",
    "CLE": "Cleveland Cavaliers",
    "MIA": "Miami Heat",
    "SAS": "San Antonio Spurs",
    "CHH": "Charlotte Hornets",
    "TOT": "Multiple Teams"
}

reverse_team_mapping = {v: k for k, v in team_mapping.items()}

app = dash.Dash(__name__, external_stylesheets=[
    "https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap"
])

feature_notes = html.Div(
    [
        html.H4("NOTE: Variable Inputs (Some are feature engineering & need to be calculated)"),
        html.Ul(
            [
                html.Li(
                    [
                        html.B("Shooting Efficiency: "),
                        "Measures a player's shooting accuracy by averaging their field goal percentage ",
                        "(baskets made divided by baskets attempted) and effective field goal percentage ",
                        "(which accounts for the added value of three-point shots).",
                        html.Br(),
                        html.Br(),
                        html.I("Formula: "),
                        "Shooting Efficiency = (Field Goal Percentage + Effective Field Goal Percentage) / 2",
                    ]
                ),
                html.Br(),
                html.Li(
                    [
                        html.B("Weighted Efficiency (WEFF): "),
                        "Evaluates a player's overall performance. Rewards positive contributions like points scored, assists, total rebounds, steals, and blocks. ",
                        "Penalizes inefficiencies like missed field goal attempts, missed free throw attempts, and turnovers. Normalized by games played.",
                        html.Br(),
                        html.Br(),
                        html.I("Formula: "),
                        "WEFF = ((2 * Points Scored) + (1.5 * Assists) + (1.2 * Total Rebounds) + Steals + Blocks - 0.5 * (Field Goals Attempted - Field Goals Made) - 0.5 * (Free Throws Attempted - Free Throws Made) - Turnovers) / Games Played",
                    ]
                ),
                html.Br(),
                html.Li(
                    [
                        html.B("Offensive Contribution: "),
                        "Reflects a player's contribution to their team's offense by combining points scored, ",
                        "assists (weighted more heavily), and offensive rebounds.",
                        html.Br(),
                        html.Br(),
                        html.I("Formula: "),
                        "Offensive Contribution = Points Scored + (1.5 * Assists) + Offensive Rebounds",
                    ]
                ),
                html.Br(),
                html.Li(
                    [
                        html.B("Defensive Contribution: "),
                        "Quantifies a player's defensive impact by summing up defensive rebounds, steals, and blocks.",
                        html.Br(),
                        html.Br(),
                        html.I("Formula: "),
                        "Defensive Contribution = Defensive Rebounds + Steals + Blocks",
                    ]
                ),
            ]
        ),
    ]
)


app.layout = html.Div([

    html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src="/assets/logo.jpg", 
                        style={"height": "120px", "marginRight": "15px"}
                    ),
                    html.Span("The Dashboard", className="navbar-title"),
                ],
                className="navbar-content"
            ),
        ],
        className="navbar"
    ),

    html.Div(
        [
            html.H1("NBA Salary Predictor", className="header-title"),
            html.Img(
                src="/assets/bask.png", 
                className="floating-basketball",
                style={"height": "70px", "marginLeft": "15px"}
            ),
        ],
        className="title-container",
        style={"display": "flex", "alignItems": "center", "justifyContent": "center"}
    ),

    html.Div([
        html.Div([
            html.Label("Season:", className="label"),
            dcc.Slider(id='season', min=1990, max=2030, step=1, value=1990, marks={1990: '1990', 2030: '2030'}),
            html.Div(id='season-display', className="slider-display"),

            html.Label("Player Team:", className="label"),
            dcc.Dropdown(
                id='team',
                options=[{"label": name, "value": abbrev} for name, abbrev in reverse_team_mapping.items()],
                value="LAL",
                placeholder="Select a Team",
                searchable=True,
                className="dropdown"
            ),

            html.Label("Player's Age:", className="label"),
            dcc.Slider(id='age', min=18, max=50, step=1, value=18, marks={18: '18', 50: '50'}),
            html.Div(id='age-display', className="slider-display"),

            html.Label("Games Started:", className="label"),
            dcc.Slider(id='gs', min=0, max=83, step=1, value=0, marks={0: '0', 83: '83'}),
            html.Div(id='gs-display', className="slider-display"),
        ], className="input-group"),

        html.Div([
            html.Label("Field Goals Made:", className="label"),
            dcc.Slider(id='fg', min=0, max=830, step=50, value=0, marks={0: '0', 830: '830'}),
            html.Div(id='fg-display', className="slider-display"),

            html.Label("Field Goal Attempts:", className="label"),
            dcc.Slider(id='fga', min=0, max=1724, step=100, value=0, marks={0: '0', 1724: '1724'}),
            html.Div(id='fga-display', className="slider-display"),

            html.Label("Effective Field Goal Percentage:", className="label"),
            dcc.Slider(id='efg', min=0, max=1, step=0.1, value=0, marks={0: '0.0', 1: '1.0'}),
            html.Div(id='efg-display', className="slider-display"),

            html.Label("Weighted Efficiency:", className="label"),
            dcc.Slider(id='weff', min=0, max=72, step=5, value=0, marks={0: '0', 72: '72'}),
            html.Div(id='weff-display', className="slider-display"),
        ], className="input-group"),

        html.Div([
            html.Label("Points Per Game:", className="label"),
            dcc.Slider(id='ppg', min=0, max=31, step=1, value=0, marks={0: '0', 31: '31'}),
            html.Div(id='ppg-display', className="slider-display"),

            html.Label("Rebounds Per Game:", className="label"),
            dcc.Slider(id='rpg', min=0, max=16, step=0.1, value=0, marks={0: '0', 16: '16'}),
            html.Div(id='rpg-display', className="slider-display"),

            html.Label("Turnovers Per Game:", className="label"),
            dcc.Slider(id='tpg', min=0, max=6, step=0.1, value=0, marks={0: '0', 6: '6'}),
            html.Div(id='tpg-display', className="slider-display"),

            html.Label("Assists Per Game:", className="label"),
            dcc.Slider(id='apg', min=0, max=13, step=0.1, value=0, marks={0: '0', 13: '13'}),
            html.Div(id='apg-display', className="slider-display"),
        ], className="input-group"),

        html.Div([
            html.Label("Shooting Efficiency:", className="label"),
            dcc.Slider(id='shooting_efficiency', min=0, max=1, step=0.1, value=0, marks={0: '0.0', 1: '1.0'}),
            html.Div(id='shooting-efficiency-display', className="slider-display"),

            html.Label("Offensive Contribution:", className="label"),
            dcc.Slider(id='offensive_contribution', min=0, max=3042, step=100, value=0, marks={0: '0', 3042: '3042'}),
            html.Div(id='offensive-contribution-display', className="slider-display"),

            html.Label("Defensive Contribution:", className="label"),
            dcc.Slider(id='defensive_contribution', min=0, max=973, step=50, value=0, marks={0: '0', 973: '973'}),
            html.Div(id='defensive-contribution-display', className="slider-display"),

            html.Label("Games Started Percentage:", className="label"),
            dcc.Slider(id='gs_percentage', min=0, max=1, step=0.1, value=0, marks={0: '0.0', 1: '1.0'}),
            html.Div(id='gs-percentage-display', className="slider-display"),

            html.Label("Minutes Played Per Game:", className="label"),
            dcc.Slider(id='mpg', min=0, max=44, step=0.1, value=0, marks={0: '0', 44: '44'}),
            html.Div(id='mpg-display', className="slider-display"),
        ], className="input-group"),
    ], className="form-container"),

    html.Div([
        html.Button("Predict Salary", id="predict-button", n_clicks=0, className="predict-button"),
        html.Div(id="predicted-salary", className="salary-display"),
    ], className="action-container"),

    html.Div(
    feature_notes,
    className="notes-container"),

    html.Footer(
        "© 2025 Ved Patel. All rights reserved.",
        style={
            "textAlign": "center",
            "padding": "15px",
            "marginTop": "50px",
            "backgroundColor": "black",
            "color": "white",
            "fontSize": "1em",
        },
    ),

], className="main-container")



# Callback to display slider values
@app.callback(
    Output('season-display', 'children'), [Input('season', 'value')]
)
def update_season_display(value):
    return f"Selected Season: {value}"

@app.callback(
    Output('age-display', 'children'), [Input('age', 'value')]
)
def update_age_display(value):
    return f"Selected Age: {value}"

@app.callback(
    Output('gs-display', 'children'), [Input('gs', 'value')]
)
def update_gs_display(value):
    return f"Selected Games Started: {value}"

@app.callback(
    Output('fg-display', 'children'), [Input('fg', 'value')]
)
def update_fg_display(value):
    return f"Selected Field Goals Made: {value}"

@app.callback(
    Output('fga-display', 'children'), [Input('fga', 'value')]
)
def update_fga_display(value):
    return f"Selected Field Goal Attempts: {value}"

@app.callback(
    Output('efg-display', 'children'), [Input('efg', 'value')]
)
def update_efg_display(value):
    return f"Selected Effective Field Goal %: {value:.1f}"

@app.callback(
    Output('weff-display', 'children'), [Input('weff', 'value')]
)
def update_weff_display(value):
    return f"Selected Weighted Efficiency: {value}"

@app.callback(
    Output('ppg-display', 'children'), [Input('ppg', 'value')]
)
def update_ppg_display(value):
    return f"Selected Points Per Game: {value}"

@app.callback(
    Output('rpg-display', 'children'), [Input('rpg', 'value')]
)
def update_rpg_display(value):
    return f"Selected Rebounds Per Game: {value}"

@app.callback(
    Output('tpg-display', 'children'), [Input('tpg', 'value')]
)
def update_tpg_display(value):
    return f"Selected Turnovers Per Game: {value}"

@app.callback(
    Output('apg-display', 'children'), [Input('apg', 'value')]
)
def update_apg_display(value):
    return f"Selected Assists Per Game: {value}"

@app.callback(
    Output('shooting-efficiency-display', 'children'), [Input('shooting_efficiency', 'value')]
)
def update_shooting_efficiency_display(value):
    return f"Selected Shooting Efficiency: {value:.1f}"

@app.callback(
    Output('offensive-contribution-display', 'children'), [Input('offensive_contribution', 'value')]
)
def update_offensive_contribution_display(value):
    return f"Selected Offensive Contribution: {value}"

@app.callback(
    Output('defensive-contribution-display', 'children'), [Input('defensive_contribution', 'value')]
)
def update_defensive_contribution_display(value):
    return f"Selected Defensive Contribution: {value}"

@app.callback(
    Output('gs-percentage-display', 'children'), [Input('gs_percentage', 'value')]
)
def update_gs_percentage_display(value):
    return f"Selected GS Percentage: {value:.1f}"

@app.callback(
    Output('mpg-display', 'children'), [Input('mpg', 'value')]
)
def update_mpg_display(value):
    return f"Selected Minutes Per Game: {value}"

# Callback to handle prediction
@app.callback(
    Output("predicted-salary", "children"),
    [Input("predict-button", "n_clicks")],
    [
        State("team", "value"),
        State("season", "value"),
        State("age", "value"),
        State("gs", "value"),
        State("fg", "value"),
        State("fga", "value"),
        State("efg", "value"),
        State("weff", "value"),
        State("ppg", "value"),
        State("rpg", "value"),
        State("tpg", "value"),
        State("apg", "value"),
        State("shooting_efficiency", "value"),
        State("offensive_contribution", "value"),
        State("defensive_contribution", "value"),
        State("gs_percentage", "value"),
        State("mpg", "value")
    ]
)
def predict_salary(n_clicks, team, season, age, gs, fg, fga, efg, weff, ppg, rpg, tpg, apg, 
                   shooting_efficiency, offensive_contribution, defensive_contribution, gs_percentage, mpg):
    if n_clicks and n_clicks > 0:
        input_data = {
            "Tm": [team], 
            "Season": [season],
            "Age": [age],
            "GS": [gs],
            "FG": [fg],
            "FGA": [fga],
            "eFG%": [efg],
            "WEFF": [weff],
            "PPG": [ppg],
            "RPG": [rpg],
            "TPG": [tpg],
            "APG": [apg],
            "ShootingEfficiency": [shooting_efficiency],
            "OffensiveContribution": [offensive_contribution],
            "DefensiveContribution": [defensive_contribution],
            "GS%": [gs_percentage],
            "MPG": [mpg]
        }

        input_df = pd.DataFrame(input_data)

        input_df = input_df[[
            "Season", "Tm", "Age", "GS", "FG", "FGA", "eFG%", "WEFF", 
            "PPG", "RPG", "TPG", "APG", "ShootingEfficiency", 
            "OffensiveContribution", "DefensiveContribution", 
            "GS%", "MPG"
        ]]

        salary = random_forest_pipeline.predict(input_df)[0]
        return f"Predicted Salary: ${salary:,.2f}"
    return "Click the button to predict the salary."

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=8080)
