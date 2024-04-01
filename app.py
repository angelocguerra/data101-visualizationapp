import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

# Set the Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

# Define regions, education levels, and metrics (options)
regions = ['NCR', 'Region 1', 'Region 2', 'Region 3', 'Region 4A', 'Region 4B', 'Region 5', 'Region 6', 'Region 7', 'Region 8', 'Region 9', 'Region 10', 'Region 11', 'Region 12', 'CAR', 'CARAGA', 'ARMM']
education_levels = ['Primary', 'Secondary']
education_metrics = ['Enrollment', 'Completion', 'Dropout']
years = list(range(2006, 2016))

# Define layout
app.layout = html.Div([
    # Navbar
    dbc.Navbar(
        [
            dbc.NavbarBrand("Philippine Education Dashboard", className="ml-1", style={"margin-left": "10px", "font-weight": "550"}),   
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
    
    # Filters Card
    dbc.Card(
        [
            dbc.CardHeader(html.H3("Filters", className="mb-0", style={'color': 'black'})),
            dbc.CardBody(
                [
                    # Regions
                    html.Div([
                        html.Label('Select Region:', style={'color': 'black'}),  # Change font color to black
                        dcc.Dropdown(
                            id='region-dropdown',
                            options=[{'label': region, 'value': region} for region in regions],
                            value=regions[0],
                            multi=True,
                        ),
                    ], style={'margin-bottom': '10px'}),
                    
                    # Education Level
                    html.Div([
                        html.Label('Select Education Level:', style={'color': 'black'}),  # Change font color to black
                        dcc.Dropdown(
                            id='education-level-dropdown',
                            options=[{'label': level, 'value': level} for level in education_levels],
                            value=education_levels[0]
                        ),
                    ], style={'margin-bottom': '10px'}),
                    
                    # Education Metric
                    html.Div([
                        html.Label('Select Education Metric:', style={'color': 'black'}),  # Change font color to black
                        dcc.Dropdown(
                            id='education-metric-dropdown',
                            options=[{'label': metric, 'value': metric} for metric in education_metrics],
                            value=education_metrics[0]
                        ),
                    ], style={'margin-bottom': '10px'}),
                    
                    # Year
                    html.Div([
                        html.Label('Select Year Range:', style={'color': 'black'}),  # Change font color to black
                        dcc.RangeSlider(
                            id='year-slider',
                            min=min(years),
                            max=max(years),
                            value=[min(years), max(years)],  # Default range from min to max years
                            marks={str(year): str(year) for year in years}, 
                            step=None,  # Allow any step within the range
                        )
                    ], style={'margin-bottom': '10px'}),
                ]
            ),
        ],
        style={"width": "30%", "margin": "20px", "background-color": "#C3DCBC"},  # Set card color to C3DCBC
    ),
    
    html.Div(id='output-container', style={'margin-top': '20px'})
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
