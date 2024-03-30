import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

# Set the Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

# Define regions, education levels, and metrics (placeholder data)
regions = ['NCR', 'Region 1', 'Region 2', 'Region 3', 'Region 4A', 'Region 4B', 'Region 5', 'Region 6', 'Region 7', 'Region 8', 'Region 9', 'Region 10', 'Region 11', 'Region 12', 'CAR', 'CARAGA', 'ARMM']
education_levels = ['Primary', 'Secondary']
education_metrics = ['Enrollment', 'Completion', 'Dropout']
years = list(range(2006, 2016))

# Define layout
app.layout = html.Div([
    dbc.Navbar(
        [
            dbc.NavbarBrand("Philippine Education Dashboard", className="ml-1", style={"margin-left": "10px", "font-weight": "550"}),   # Align brand to the left
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Home", href="#")),
                ],
                className="ml-auto",
                navbar=True,
            ),
        ],
        color="primary",
        dark=True,
    ),
    html.H3("Filters"),
    html.Div([
        html.Div([
            html.Label('Select Region:'),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': region, 'value': region} for region in regions],
                value=regions[0]
            ),
        ], style={'width': '30%', 'margin-left': '10px'}),
        
        html.Div([
            html.Label('Select Education Level:'),
            dcc.Dropdown(
                id='education-level-dropdown',
                options=[{'label': level, 'value': level} for level in education_levels],
                value=education_levels[0]
            ),
        ], style={'width': '30%', 'margin-left': '10px'}),
        
        html.Div([
            html.Label('Select Education Metric:'),
            dcc.Dropdown(
                id='education-metric-dropdown',
                options=[{'label': metric, 'value': metric} for metric in education_metrics],
                value=education_metrics[0]
            ),
        ], style={'width': '30%', 'margin-left': '10px'}),
        
        html.Div([
            html.Label('Select Year:'),
            dcc.Slider(
                id='year-slider',
                min=min(years),
                max=max(years),
                value=min(years),
                marks={str(year): str(year) for year in years},  # Align years to the left
                step=None
            )
        ], style={'width': '30%', 'margin-left': '10px', 'margin-top': '20px'}),
    ]),
    html.Div(id='output-container', style={'margin-top': '20px'})
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)







