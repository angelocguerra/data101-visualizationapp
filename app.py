import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pathlib import Path

# Set the Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

dataset_folder = Path('cleaned-datasets/')
# Primary Completion Rates
primary_completion = pd.read_csv(dataset_folder / 'Primary/' 'Primary_Completion_Rate_by_Region_and_Year.csv', index_col='Region')

# Primary Drop-out Rates
primary_dropout = pd.read_csv(dataset_folder / 'Primary/' 'Primary_Drop-out_Rate_by_Region_and_Year.csv', index_col='Region')

# Primary Net Enrollment Rates
primary_enrollment = pd.read_csv(dataset_folder / 'Primary/' 'Primary_Net_Enrollment_Rate_by_Region_and_Year.csv', index_col='Region')

# Secondary Completion Rates
secondary_completion = pd.read_csv(dataset_folder / 'Secondary/' 'Secondary_Completion_Rate_by_Region_and_Year.csv', index_col='Region')

# Secondary Drop-out Rates
secondary_dropout = pd.read_csv(dataset_folder / 'Secondary/' 'Secondary_Drop-out_Rate_by_Region_and_Year.csv', index_col='Region')

# Secondary Net Enrollment Rates
secondary_enrollment = pd.read_csv(dataset_folder / 'Secondary/' 'Secondary_Enrollment_Rate_by_Region_and_Year.csv', index_col='Region')

# Poverty Incidence Rates
poverty_incidence = pd.read_csv(dataset_folder / 'Interpolated_Poverty_Incidence_among_Population.csv', index_col='Region')

# Define regions, education levels, and metrics (options)
regions = ['NCR', 'Region I', 'Region II', 'Region III', 'Region IV-A', 'Region IV-B', 'Region V', 'Region VI', 'Region VII', 'Region VIII', 'Region IX', 'Region X', 'Region XI', 'Region XII', 'CAR', 'Caraga', 'ARMM']
education_levels = ['Primary', 'Secondary']
education_metrics = ['Enrollments', 'Completions', 'Dropouts']
years = list(range(2006, 2016))

# Define layout
app.layout = html.Div([
    # Navbar
    dbc.Navbar(
        [
            dbc.NavbarBrand("Philippine Education Dashboard", className="ml-1", style={"background-color": "#446C37", "margin-left": "10px", "font-weight": "550"}),   
        ],
        color="#446C37",  
        dark=True,
    ),

    dbc.Row(children=[
        # Filters Div
        dbc.Col(
            html.Div([
                html.H3("Filters", style={'color': 'black'}),
                # Regions
                html.Div([
                    html.Label('Select Region:', style={'color': 'black'}),  
                    dcc.Dropdown(
                        id='region-dropdown',
                        options=[{'label': region, 'value': region} for region in regions],
                        value=regions[0],
                        multi=True,
                    ),
                ], style={'margin-bottom': '10px'}),
                
                # Education Level
                html.Div([
                    html.Label('Select Education Level:', style={'color': 'black'}), 
                    dcc.Dropdown(
                        id='education-level-dropdown',
                        options=[{'label': level, 'value': level} for level in education_levels],
                        value=education_levels[0]
                    ),
                ], style={'margin-bottom': '10px'}),
                
                # Education Metric
                html.Div([
                    html.Label('Select Education Metric:', style={'color': 'black'}), 
                    dcc.Dropdown(
                        id='education-metric-dropdown',
                        options=[{'label': metric, 'value': metric} for metric in education_metrics],
                        value=education_metrics[0]
                    ),
                ], style={'margin-bottom': '10px'}),
                
                # Year
                html.Div([
                    html.Label('Select Year Range:', style={'color': 'black'}), 
                    dcc.RangeSlider(
                        id='year-slider',
                        min=min(years),
                        max=max(years),
                        value=[min(years), max(years)],  
                        marks={str(year): str(year) for year in years}, 
                        step=None,  
                    )
                ], style={'margin-bottom': '10px'}),
            ], style={"background-color": "#C3DCBC", "padding": "20px"}),
            width=6
        ),

        # Line Chart
        dbc.Col(
            dcc.Loading(id="line-loading", children=dcc.Graph(id='line-chart')),
            width=6
        ),
    ]),

    dbc.Row(children=[
        # Bar Chart Card and Legends
        dbc.Col(
            html.Div([
                dbc.Card(
                    [
                        dbc.CardHeader(html.H3("Bar chart", className="mb-0", style={'color': 'black', 'background-color': 'white'})),
                        dbc.CardBody(
                            [
                                html.Div([
                                    html.Div(style={'background-color': '#D5FBCB', 'height': '20px', 'width': '20px', 'display': 'inline-block', 'margin-right': '5px'}),
                                    html.Label('Enrollments', style={'color': 'black'}),
                                ]),
                                html.Div([
                                    html.Div(style={'background-color': '#00E08F', 'height': '20px', 'width': '20px', 'display': 'inline-block', 'margin-right': '5px'}),
                                    html.Label('Completions', style={'color': 'black'}),
                                ]),
                                html.Div([
                                    html.Div(style={'background-color': '#23B37F', 'height': '20px', 'width': '20px', 'display': 'inline-block', 'margin-right': '5px'}),
                                    html.Label('Dropouts', style={'color': 'black'}),
                                ]),
                            ]
                        ),
                    ],
                    style={"margin-bottom": "10px"}
                ),
                dcc.Loading(id="bar-loading", children=[
                    dbc.Col(id='bar-chart-container')
                ])
            ]),
            width=6,
            style={"background-color": "#446C37"}
        ),

        # Scatter Plot
        dbc.Col(
            dcc.Loading(id="scatter-loading", children=dcc.Graph(id='scatter-plot')),
            width=6
        ),
    ]),

    html.Div(id='output-container', style={'margin-top': '20px'})
])


@app.callback(
    Output('line-chart', 'figure'),
    [
        Input('region-dropdown', 'value'),
        Input('education-level-dropdown', 'value'),
        Input('education-metric-dropdown', 'value'),
    ]
)
def update_line_chart(regions_selected, educ_level_selected, educ_metric_selected):
    years_selected = [2006, 2015]  # Set default year range from 2006 to 2015
    years_range = list(range(years_selected[0], years_selected[1] + 1))
    years_as_strings = [str(year) for year in years_range]

    line_df = pd.DataFrame()
    # Determine which dataset to use based on education level and metric
    if educ_level_selected == 'Secondary':
        if educ_metric_selected == 'Enrollments':
            line_df = secondary_enrollment.loc[regions_selected, years_as_strings ]
        elif educ_metric_selected == 'Completions':
            line_df = secondary_completion.loc[regions_selected, years_as_strings ]
        elif educ_metric_selected == 'Dropouts':
            line_df = secondary_dropout.loc[regions_selected, years_as_strings ]

    elif educ_level_selected == 'Primary':
        if educ_metric_selected == 'Enrollments':
            line_df = primary_enrollment.loc[regions_selected, years_as_strings]
        elif educ_metric_selected == 'Completions':
            line_df = primary_completion.loc[regions_selected, years_as_strings]
        elif educ_metric_selected == 'Dropouts':
            line_df = primary_dropout.loc[regions_selected, years_as_strings ]

 
    line_df = line_df.T
    line_df = line_df.reset_index().melt(id_vars='index', var_name='Region', value_name=educ_metric_selected)
    line_df['index'] = line_df['index'].astype(str) 
    
    # Plotly express line chart
    fig = px.line(line_df, 
                  x='index', 
                  y=educ_metric_selected,
                  title=f'{educ_level_selected} {educ_metric_selected} Rates by Region',
                  labels={'index': 'Year', educ_metric_selected: educ_metric_selected},
                  color='Region',
                  color_discrete_map={region: px.colors.qualitative.Plotly[i] for i, region in enumerate(regions_selected)}
                 )
    
    return fig

@app.callback(
    Output('scatter-plot', 'figure'),
    [
        Input('region-dropdown', 'value'),
        Input('education-level-dropdown', 'value'),
        Input('education-metric-dropdown', 'value'),
    ]
)
def update_scatter_plot(regions_selected, educ_level_selected, educ_metric_selected):
    years_selected = [2006, 2015]  # Set default year range from 2006 to 2015
    years_range = list(range(years_selected[0], years_selected[1] + 1))
    years_as_strings = [str(year) for year in years_range]

    scatter_df = pd.DataFrame()

    if educ_level_selected == 'Secondary':
        if educ_metric_selected == 'Enrollments':
            scatter_df = secondary_enrollment.loc[regions_selected, years_as_strings ]
        elif educ_metric_selected == 'Completions':
            scatter_df = secondary_completion.loc[regions_selected, years_as_strings ]
        elif educ_metric_selected == 'Dropouts':
            scatter_df = secondary_dropout.loc[regions_selected, years_as_strings ]

    elif educ_level_selected == 'Primary':
        if educ_metric_selected == 'Enrollments':
            scatter_df = primary_enrollment.loc[regions_selected, years_as_strings]
        elif educ_metric_selected == 'Completions':
            scatter_df = primary_completion.loc[regions_selected, years_as_strings]
        elif educ_metric_selected == 'Dropouts':
            scatter_df = primary_dropout.loc[regions_selected, years_as_strings ]
   
 
    poverty_df = poverty_incidence.loc[regions_selected, years_as_strings]


    scatter_df = scatter_df.T
    scatter_df = scatter_df.reset_index().melt(id_vars='index', var_name='Region', value_name=educ_metric_selected)
    scatter_df['index'] = scatter_df['index'].astype(str)
    
    poverty_df = poverty_df.T
    poverty_df = poverty_df.reset_index().melt(id_vars='index', var_name='Region', value_name='Poverty Incidence')
    poverty_df['index'] = poverty_df['index'].astype(str)
    
    
    combined_df = scatter_df.merge(poverty_df, on=['index', 'Region'])

    # Plotly express scatter plot
    fig = px.scatter(combined_df, 
                     x=educ_metric_selected,
                     y='Poverty Incidence',
                     title=f'Poverty Incidence Rates and {educ_level_selected} {educ_metric_selected} Rates by Region',
                     labels={educ_metric_selected: educ_metric_selected, 'Poverty Incidence': 'Poverty Incidence'},
                     color='Region',
                     color_discrete_map={region: px.colors.qualitative.Plotly[i] for i, region in enumerate(regions_selected)}
                    )
    
    return fig

def update_bar_chart(regions_selected, educ_level_selected, educ_metric_selected, years_selected):

    if not isinstance(regions_selected, list):
        regions_selected = [regions_selected]

    years_range = list(range(years_selected[0], years_selected[1] + 1))
    years_as_strings = [str(year) for year in years_range]

    print(regions_selected)
    print(educ_level_selected)
    print(educ_metric_selected)
    print(years_as_strings)
    
    bar_enrollment_df = pd.DataFrame()
    bar_completion_df  = pd.DataFrame()
    bar_dropout_df  = pd.DataFrame()
    bar_df = pd.DataFrame()
    # Determine which dataset to use based on education level and metric
    if educ_level_selected == 'Secondary':
            bar_enrollment_df = secondary_enrollment.loc[regions_selected, years_as_strings ]
            bar_completion_df = secondary_completion.loc[regions_selected, years_as_strings ]
            bar_dropout_df = secondary_dropout.loc[regions_selected, years_as_strings ]

    elif educ_level_selected == 'Primary':
            bar_enrollment_df = primary_enrollment.loc[regions_selected, years_as_strings]
            bar_completion_df  = primary_completion.loc[regions_selected, years_as_strings]
            bar_dropout_df  = primary_dropout.loc[regions_selected, years_as_strings ]

    bar_completion_df = pd.DataFrame(bar_completion_df)
    bar_completion_df = pd.DataFrame(bar_completion_df.mean(axis=1).round(1), columns=['Completions'])

    bar_dropout_df = pd.DataFrame(bar_dropout_df )
    bar_dropout_df = pd.DataFrame(bar_dropout_df.mean(axis=1).round(1), columns=['Dropouts'])

    bar_enrollment_df = pd.DataFrame(bar_enrollment_df)
    bar_enrollment_df = pd.DataFrame(bar_enrollment_df.mean(axis=1).round(1), columns=['Enrollments'])

    print(bar_completion_df)
    print(bar_dropout_df)
    print(bar_enrollment_df)

    bar_df = pd.merge(bar_completion_df, bar_dropout_df, on='Region', how='outer')
    bar_df = pd.merge(bar_df, bar_enrollment_df, on='Region', how='outer')

    bar_df = bar_df.sort_values(by=educ_metric_selected, ascending=False)

    fig_list = []  # List to store multiple figures

    if(len(regions_selected) > 1):
        for i in range(0, len(regions_selected)):    
            fig = go.Figure()
            
            fig.add_trace(go.Bar(x=[bar_df['Dropouts'].iloc[i]],
                                orientation='h',
                                name='Drop-outs',
                                marker=dict(color='#23B37F', line=dict(width=0)),
                                text = bar_df['Dropouts'].iloc[i],
                                texttemplate = "%{text}%",
                                textposition='auto')
            )

            fig.add_trace(go.Bar(x=[bar_df['Completions'].iloc[i]],
                                orientation='h',
                                name='Completions',
                                marker=dict(color='#00E08F', line=dict(width=0)),
                                text=bar_df['Completions'].iloc[i],
                                texttemplate = "%{text}%",
                                textposition='auto')
            )

            fig.add_trace(go.Bar(x=[bar_df['Enrollments'].iloc[i]],
                                orientation='h',
                                name='Enrollments',
                                marker=dict(color='#D5FBCB',line=dict(width=0)),
                                text=bar_df['Enrollments'].iloc[i],
                                texttemplate = "%{text}%",
                                textposition='auto')
            )

            fig.update_layout(showlegend=False)
            fig.update_xaxes(autorange='reversed')
            fig.update_layout(height=100, width=550, bargroupgap=0.15)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(margin=dict(l=130, r=15, t=15, b=15, pad=130))
            fig.update_layout(
                title={
                    'text': f'{i+1}. {bar_df.index[i]}',
                    'y':0.6,
                    'x':0.06,
                    'xanchor': 'left',
                    'yanchor': 'middle'})
            fig.update_layout(plot_bgcolor='#446C37')
            fig.update_layout(title_font_color='#FFFFFF',
                                #title_font_family='Sansation-Regular',
                                title_font_size=12,
                            )
            fig.update_layout(font_color='#181717',
                                #font_family='Sansation-Regular',
                                font_size=9,
                            )
          
            fig_list.append(fig)  # Append each figure to the list

    if (len(regions_selected) == 1):
        fig = go.Figure()
        fig.add_trace(go.Bar(y=[bar_df['Enrollments'].iloc[0]],
                        name='Enrollments',
                        marker=dict(color='#D5FBCB',line=dict(width=0)),
                        text=bar_df['Enrollments'].iloc[0],
                        texttemplate = "%{text}%",
                        textposition='auto')
        )

        fig.add_trace(go.Bar(y=[bar_df['Completions'].iloc[0]],
                        name='Completions',
                        marker=dict(color='#00E08F', line=dict(width=0)),
                        text=bar_df['Completions'].iloc[0],
                        texttemplate = "%{text}%",
                        textposition='auto')
        )

        fig.add_trace(go.Bar(y=[bar_df['Dropouts'].iloc[0]],
                        name='Drop-outs',
                        marker=dict(color='#23B37F', line=dict(width=0)),
                        text=bar_df['Dropouts'].iloc[0],
                        texttemplate = "%{text}%",
                        textposition='auto')
        )

        fig.update_layout(showlegend=False)
        fig.update_xaxes(autorange='reversed')
        fig.update_layout(height=350, width=550, bargroupgap=0.15)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(margin=dict(l=15, r=15, t=15, b=15, pad=15))
        fig.update_layout(plot_bgcolor='#446C37')
        fig.update_layout(font_color='#181717',
                                #font_family='Sansation',
                                # font_size=12,
                                )
        fig_list.append(fig)  # Append the single figure to the list
        
    return fig_list  # Return the list of figures

@app.callback(
    Output('bar-chart-container', 'children'),
    [
        Input('region-dropdown', 'value'),
        Input('education-level-dropdown', 'value'),
        Input('education-metric-dropdown', 'value'),
        Input('year-slider', 'value')
    ]
)

def update_graph(regions_selected, educ_level_selected, educ_metric_selected, years_selected):
    fig_list = update_bar_chart(regions_selected, educ_level_selected, educ_metric_selected, years_selected)
    graph_list = [dcc.Graph(figure=fig) for fig in fig_list]
    return graph_list

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)