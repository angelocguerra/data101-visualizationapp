import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

import pandas as pd
import plotly.express as px
import geopandas as gpd
import plotly.graph_objects as go

from pathlib import Path

# Set Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Philippine Education Dashboard")
server = app.server

dataset_folder = Path('cleaned-datasets/')
# Primary Completion Rates
primary_completion = pd.read_csv(dataset_folder / 'Primary/' 'Primary_Completion_Rate_by_Region_and_Year.csv', index_col='Region')
gpd_primary_completion = gpd.read_file(dataset_folder / 'Spatial/primary_completion.shp')
# Primary Drop-out Rates
primary_dropout = pd.read_csv(dataset_folder / 'Primary/' 'Primary_Drop-out_Rate_by_Region_and_Year.csv', index_col='Region')
gpd_primary_dropout = gpd.read_file(dataset_folder / 'Spatial/primary_dropout.shp')
# Primary Net Enrollment Rates
primary_enrollment = pd.read_csv(dataset_folder / 'Primary/' 'Primary_Net_Enrollment_Rate_by_Region_and_Year.csv', index_col='Region')
gpd_primary_enrollment = gpd.read_file(dataset_folder / 'Spatial/primary_enrollment.shp')
# Secondary Completion Rates
secondary_completion = pd.read_csv(dataset_folder / 'Secondary/' 'Secondary_Completion_Rate_by_Region_and_Year.csv', index_col='Region')
gpd_secondary_completion = gpd.read_file(dataset_folder / 'Spatial/secondary_completion.shp')
# Secondary Drop-out Rates
secondary_dropout = pd.read_csv(dataset_folder / 'Secondary/' 'Secondary_Drop-out_Rate_by_Region_and_Year.csv', index_col='Region')
gpd_secondary_dropout = gpd.read_file(dataset_folder / 'Spatial/secondary_dropout.shp')
# Secondary Net Enrollment Rates
secondary_enrollment = pd.read_csv(dataset_folder / 'Secondary/' 'Secondary_Enrollment_Rate_by_Region_and_Year.csv', index_col='Region')
gpd_secondary_enrollment = gpd.read_file(dataset_folder / 'Spatial/secondary_enrollment.shp')
# Poverty Incidence Rates
poverty_incidence = pd.read_csv(dataset_folder / 'Interpolated_Poverty_Incidence_among_Population.csv', index_col='Region')

# Define regions, education levels, and metrics (options)
regions = ['NCR', 'Region I', 'Region II', 'Region III', 'Region IV-A', 'Region IV-B', 'Region V', 'Region VI', 'Region VII', 'Region VIII', 'Region IX', 'Region X', 'Region XI', 'Region XII', 'CAR', 'Caraga', 'ARMM']
education_levels = ['Primary', 'Secondary']
education_metrics = ['Enrollments', 'Completions', 'Dropouts']
years = list(range(2006, 2016))

px.set_mapbox_access_token(open(".mapbox_token").read())

# Define layout
app.layout = html.Div([
    # Navbar
    dbc.Navbar(
        [
            dbc.NavbarBrand("Philippine Education Dashboard",
                            style={"margin-left": '1rem', "font-family": "Sansation Bold", "font-size": "1.25rem"})
        ],
        color="#387E45",
        dark=True,
        style={"height": "3rem", "flex": "0 1 auto"}
    ),

    dbc.Row(children=[
        # Filters and Bar Chart
        dbc.Col(width=3, children=[
            # Filters
            dbc.Row([
                # Filters Label
                html.Div([
                    html.Img(alt="filter icon", src="assets/Filter icon.png", width=20, height=20),
                    html.P("Filters", style={'color': 'black', "font-family": "Sansation Bold", "margin-bottom": "auto",
                                             "font-size": "1.25rem", "margin-left": "0.7em"}),
                ], style={"display": "flex", "align-items": "center", "margin": "auto", 'margin-bottom': '5px'}),

                # Regions
                html.Div([
                    html.Label('Select Region/s:',
                               style={'color': 'black', "font-family": "Sansation Regular", "font-size": "0.8rem"}),
                    dcc.Dropdown(
                        id='region-dropdown',
                        options=[{'label': region, 'value': region} for region in regions],
                        value=regions[0],
                        multi=True,
                        className="dropdown-region",
                        style={"font-family": "Sansation Regular", "font-size": "0.95rem"}
                    ),
                ], style={'margin-bottom': '5px'}),

                # Education Level
                html.Div([
                    html.Label('Select Education Level:',
                               style={'color': 'black', "font-family": "Sansation Regular", "font-size": "0.8rem"}),
                    dcc.Dropdown(
                        id='education-level-dropdown',
                        options=[{'label': level, 'value': level} for level in education_levels],
                        value=education_levels[0],
                        className="dropdown-level",
                        style={"font-family": "Sansation Regular", "font-size": "0.95rem"}
                    ),
                ], style={'margin-bottom': '5px'}),

                # Education Metric
                html.Div([
                    html.Label('Select Education Metric:',
                               style={'color': 'black', "font-family": "Sansation Regular", "font-size": "0.8rem"}),
                    dcc.Dropdown(
                        id='education-metric-dropdown',
                        options=[{'label': metric, 'value': metric} for metric in education_metrics],
                        value=education_metrics[0],
                        className="dropdown-metric",
                        style={"font-family": "Sansation Regular", "font-size": "0.95rem"}
                    ),
                ], style={'margin-bottom': '5px'}),

                # Year
                html.Div([
                    html.Label('Select Year:',
                               style={'color': 'black', "font-family": "Sansation Regular", 'margin-bottom': '5px', "font-size": "0.85rem"}),
                    dcc.Slider(
                        id='year-slider',
                        min=min(years),
                        max=max(years),
                        value=min(years),
                        marks={str(year): str(year) for year in years},
                        included=False,
                        step=1,
                    )
                ], style={'margin-bottom': '10px'}, className="year-slider"),

                # Title Legends
                html.Div([
                    html.Div([
                        html.P("Bar Chart", style={'color': 'black', 'font-family': 'Sansation Bold', 'margin-bottom': '0.5rem'}),
                        html.P("Comparison of all educational metrics across different regions which are ranked according to the highest selected metric.", style={'color': 'black', 'font-size': '0.7rem', 'font-family': 'Sansation Regular'}),
                        html.Div([
                            html.Div([
                                html.Div(style={'background-color': '#D5FBCB', 'height': '15px', 'width': '15px', 'display': 'inline-block', 'margin-right': '10px'}),
                                html.Label('Enrollments', style={'color': 'black', 'font-family': 'Sansation Regular', 'font-size': '0.8rem'}),
                            ], style={"display": "flex", "align-items": "center", "margin-right": "10px"}),

                            html.Div([
                                html.Div(style={'background-color': '#00E08F', 'height': '15px', 'width': '15px', 'display': 'inline-block', 'margin-right': '10px'}),
                                html.P('Completions', style={'color': 'black', 'font-family': 'Sansation Regular', 'font-size': '0.8rem', 'margin-bottom': 0}),
                            ], style={"display": "flex", "align-items": "center", "margin-right": "10px"}),

                            html.Div([
                                html.Div(style={'background-color': '#23B37F', 'height': '15px', 'width': '15px', 'display': 'inline-block', 'margin-right': '10px'}),
                                html.Label('Dropouts', style={'color': 'black', 'font-family': 'Sansation Regular', 'font-size': '0.8rem'}),
                            ], style={"display": "flex", "align-items": "center", "margin-right": "10px"}),
                        ]),
                    ], style={'background-color': '#8AC278', 'padding': '1rem', 'border-radius': '1rem'}),
                ], style={'margin-top': '5px'}),

                # Bar Chart
                html.Div([
                    dcc.Loading(id="bar-loading", children=[
                        # html.Div(id='bar-chart-container')
                        html.Div(id='bar-chart-container', className="barchart",
                                 style={"height": "175px", "overflow": "scroll", "background-color": "#446C37"})
                    ])
                ], className="bar-chart", style={})
            ]),
        ], style={"background-color": "#C3DCBC", "padding": "20px", "padding-left": "30px", "flex-flow": "column"}),

        # Line Chart and Scatterplot
        dbc.Col(children=[
            # Line Chart
            dbc.Card(html.Div([dcc.Loading(id="line-loading", children=dcc.Graph(id='line-chart', style={'height': '45vh'}))]), style={'margin-bottom': '1rem', 'margin-top': '1rem'}),
            # Scatterplot
            dbc.Card(html.Div([dcc.Loading(id="scatter-loading", children=dcc.Graph(id='scatter-plot', style={'height': '45vh'}))]))
        ]),

        dbc.Col(children=[
            # Choropleth Map
            html.Div([dcc.Loading(id="choropleth-loading", children=dcc.Graph(id='choropleth-map', style={'height': '90vh'}))]),
            html.Div([dbc.Button("Reset Highlights", id="resetHighlights", style={'background-color': '#76C585', 'border': 'none'})], className="d-grid gap-2 mt-1")
        ]),
    ], style={"height": "100%", "width": "100%"})
], style={"height": "100vh", "display": "flex", "flex-flow": "column"})


def update_bar_chart(regions_selected, educ_level_selected, educ_metric_selected, years_selected):
    if not isinstance(regions_selected, list):
        regions_selected = [regions_selected]

    years_as_strings = [str(years_selected)]

    bar_enrollment_df = pd.DataFrame()
    bar_completion_df = pd.DataFrame()
    bar_dropout_df = pd.DataFrame()

    if educ_level_selected == 'Secondary':
        bar_enrollment_df = secondary_enrollment.loc[regions_selected, years_as_strings]
        bar_completion_df = secondary_completion.loc[regions_selected, years_as_strings]
        bar_dropout_df = secondary_dropout.loc[regions_selected, years_as_strings]

    elif educ_level_selected == 'Primary':
        bar_enrollment_df = primary_enrollment.loc[regions_selected, years_as_strings]
        bar_completion_df = primary_completion.loc[regions_selected, years_as_strings]
        bar_dropout_df = primary_dropout.loc[regions_selected, years_as_strings]

    bar_completion_df = pd.DataFrame(bar_completion_df)
    bar_completion_df = pd.DataFrame(bar_completion_df.mean(axis=1).round(1), columns=['Completions'])
    bar_dropout_df = pd.DataFrame(bar_dropout_df)
    bar_dropout_df = pd.DataFrame(bar_dropout_df.mean(axis=1).round(1), columns=['Dropouts'])
    bar_enrollment_df = pd.DataFrame(bar_enrollment_df)
    bar_enrollment_df = pd.DataFrame(bar_enrollment_df.mean(axis=1).round(1), columns=['Enrollments'])

    bar_df = pd.merge(bar_completion_df, bar_dropout_df, on='Region', how='outer')
    bar_df = pd.merge(bar_df, bar_enrollment_df, on='Region', how='outer')

    bar_df = bar_df.sort_values(by=educ_metric_selected, ascending=False)

    fig_list = []  # List to store multiple figures

    if len(regions_selected) > 1:
        for i in range(0, len(regions_selected)):
            fig = go.Figure()

            fig.add_trace(go.Bar(x=[bar_df['Dropouts'].iloc[i]],
                                 orientation='h',
                                 name='Drop-outs',
                                 marker=dict(color='#23B37F', line=dict(width=0)),
                                 text=bar_df['Dropouts'].iloc[i],
                                 texttemplate="%{text}%",
                                 textposition='auto')
                          )

            fig.add_trace(go.Bar(x=[bar_df['Completions'].iloc[i]],
                                 orientation='h',
                                 name='Completions',
                                 marker=dict(color='#00E08F', line=dict(width=0)),
                                 text=bar_df['Completions'].iloc[i],
                                 texttemplate="%{text}%",
                                 textposition='auto')
                          )

            fig.add_trace(go.Bar(x=[bar_df['Enrollments'].iloc[i]],
                                 orientation='h',
                                 name='Enrollments',
                                 marker=dict(color='#D5FBCB', line=dict(width=0)),
                                 text=bar_df['Enrollments'].iloc[i],
                                 texttemplate="%{text}%",
                                 textposition='auto')
                          )

            fig.update_layout(height=100)
            fig.update_layout(showlegend=False)
            fig.update_xaxes(autorange='reversed')
            fig.update_layout(bargroupgap=0.15)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(margin=dict(l=130, r=15, t=15, b=15, pad=130))
            fig.update_layout(
                title={
                    'text': f'{i + 1}. {bar_df.index[i]}',
                    'y': 0.6,
                    'x': 0.06,
                    'xanchor': 'left',
                    'yanchor': 'middle'})
            fig.update_layout(plot_bgcolor='#446C37')
            fig.update_layout(title_font_color='#FFFFFF',
                              title_font_family='Sansation Regular',
                              title_font_size=12,
                              )
            fig.update_layout(font_color='#181717',
                              font_family='Sansation Regular',
                              font_size=9,
                              )

            fig_list.append(fig)  # Append each figure to the list

    elif len(regions_selected) == 1:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=[bar_df['Enrollments'].iloc[0]],
                             name='Enrollments',
                             marker=dict(color='#D5FBCB', line=dict(width=0)),
                             text=bar_df['Enrollments'].iloc[0],
                             texttemplate="%{text}%",
                             textposition='auto')
                      )

        fig.add_trace(go.Bar(y=[bar_df['Completions'].iloc[0]],
                             name='Completions',
                             marker=dict(color='#00E08F', line=dict(width=0)),
                             text=bar_df['Completions'].iloc[0],
                             texttemplate="%{text}%",
                             textposition='auto')
                      )

        fig.add_trace(go.Bar(y=[bar_df['Dropouts'].iloc[0]],
                             name='Drop-outs',
                             marker=dict(color='#23B37F', line=dict(width=0)),
                             text=bar_df['Dropouts'].iloc[0],
                             texttemplate="%{text}%",
                             textposition='auto')
                      )

        fig.update_layout(height=175)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(autorange='reversed')
        fig.update_layout(bargroupgap=0.15)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(margin=dict(l=15, r=15, t=15, b=15, pad=15))
        fig.update_layout(plot_bgcolor='#446C37')
        fig.update_layout(font_color='#181717',
                          font_family='Sansation Regular',
                          font_size=12,
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


@app.callback(
    Output('line-chart', 'figure'),
    Output('scatter-plot', 'figure'),
    [
        Input('region-dropdown', 'value'),
        Input('education-level-dropdown', 'value'),
        Input('education-metric-dropdown', 'value'),
        Input('choropleth-map', 'clickData'),

    ]
)
def update_line_chart_scatterplot(regions_selected, educ_level_selected, educ_metric_selected, selected_region_choropleth):
    years_selected = [2006, 2015]  # Set default year range from 2006 to 2015
    years_range = list(range(years_selected[0], years_selected[1] + 1))
    years_as_strings = [str(year) for year in years_range]

    line_scatter_df = pd.DataFrame()
    # Determine which dataset to use based on education level and metric
    if educ_level_selected == 'Secondary':
        if educ_metric_selected == 'Enrollments':
            line_scatter_df = secondary_enrollment.loc[regions_selected, years_as_strings]
        elif educ_metric_selected == 'Completions':
            line_scatter_df = secondary_completion.loc[regions_selected, years_as_strings]
        elif educ_metric_selected == 'Dropouts':
            line_scatter_df = secondary_dropout.loc[regions_selected, years_as_strings]
    elif educ_level_selected == 'Primary':
        if educ_metric_selected == 'Enrollments':
            line_scatter_df = primary_enrollment.loc[regions_selected, years_as_strings]
        elif educ_metric_selected == 'Completions':
            line_scatter_df = primary_completion.loc[regions_selected, years_as_strings]
        elif educ_metric_selected == 'Dropouts':
            line_scatter_df = primary_dropout.loc[regions_selected, years_as_strings]

    line_scatter_df = line_scatter_df.T
    line_scatter_df = line_scatter_df.reset_index().melt(id_vars='index', var_name='Region',
                                                         value_name=educ_metric_selected)
    line_scatter_df['index'] = line_scatter_df['index'].astype(str)

    poverty_df = poverty_incidence.loc[regions_selected, years_as_strings]

    poverty_df = poverty_df.T
    poverty_df = poverty_df.reset_index().melt(id_vars='index', var_name='Region', value_name='Poverty Incidence')
    poverty_df['index'] = poverty_df['index'].astype(str)

    combined_df = line_scatter_df.merge(poverty_df, on=['index', 'Region'])

    # Changing the opacity of data points
    # Set default all data points opaque
    opacity_values = [1.0] * len(regions_selected)
    # Set opacity of selected region to 1 and other regions to 0.3
    if selected_region_choropleth is not None:
        print(selected_region_choropleth['points'][0]['location'], regions_selected)
        opacity_values = [1.0 if region == selected_region_choropleth['points'][0]['location'] else 0.3 for region in regions_selected]

    color_discrete_map = {
        "NCR": "#47bcc4",
        "Region I": "#003f5c",
        "Region II": "#2f4b7c",
        "Region III": "#665191",
        "Region IV-A": "#a05195",
        "Region IV-B": "#c2c2ff",
        "Region V": "#ff7c43",
        "Region VI": "#ffa600",
        "Region VII": "#ffbf99",
        "Region VIII": "#9fbfdf",
        "Region IX": "#54738d",
        "Region X": "#37618a",
        "Region XI": "#2a4858",
        "Region XII": "#1e3230",
        "CAR": "#7a9eb1",
        "Caraga": "#0f1f1c",
        "ARMM": "#f95d6a"
    }

    # Plotly express line chart
    line_fig = px.line(line_scatter_df,
                       x='index',
                       y=educ_metric_selected,
                       title=f'{educ_level_selected} {educ_metric_selected} Rates by Region',
                       labels={'index': 'Year', educ_metric_selected: educ_metric_selected},
                       color='Region',
                       color_discrete_map=color_discrete_map,
                       template='plotly_white'
                       )

    line_fig.update_layout(
        font_family="Sansation Regular",
        font_color="black",
        title_font_family="Sansation Bold",
        title_font_color="#333333"
    )

    # Plotly express scatter plot
    scatter_fig = px.scatter(combined_df,
                             x=educ_metric_selected,
                             y='Poverty Incidence',
                             title=f'Poverty Incidence Rates and {educ_level_selected}<br>{educ_metric_selected} Rates by Region',
                             labels={educ_metric_selected: educ_metric_selected,
                                     'Poverty Incidence': 'Poverty Incidence'},
                             color='Region',
                             color_discrete_map=color_discrete_map,
                             template='plotly_white'
                             )

    scatter_fig.update_layout(
        font_family="Sansation Regular",
        font_color="black",
        title_font_family="Sansation Bold",
        title_font_color="#333333"
    )

    return line_fig, scatter_fig


@app.callback(
    Output('choropleth-map', 'figure'),
    [
        Input('region-dropdown', 'value'),
        Input('education-level-dropdown', 'value'),
        Input('education-metric-dropdown', 'value'),
        Input('year-slider', 'value')
    ]
)
def update_choropleth_map(regions_selected, educ_level_selected, educ_metric_selected, years_selected):
    if not isinstance(regions_selected, list):
        regions_selected = [regions_selected]

    years_as_strings = str(years_selected)
    filtered_data = pd.DataFrame()

    if educ_level_selected == 'Secondary':
        if educ_metric_selected == 'Enrollments':
            filtered_data = gpd_secondary_enrollment
        elif educ_metric_selected == 'Completions':
            filtered_data = gpd_secondary_completion
        elif educ_metric_selected == 'Dropouts':
            filtered_data = gpd_secondary_dropout

    elif educ_level_selected == 'Primary':
        if educ_metric_selected == 'Enrollments':
            filtered_data = gpd_primary_enrollment
        elif educ_metric_selected == 'Completions':
            filtered_data = gpd_primary_completion
        elif educ_metric_selected == 'Dropouts':
            filtered_data = gpd_primary_dropout

    filtered_data = filtered_data[filtered_data['Region'].isin(regions_selected)]

    custom_color_scale = [
        [0, '#D6E5D2'],
        [1, '#24361E']
    ]

    geodf = filtered_data.set_index('Region')
    map_fig = px.choropleth_mapbox(geodf,
                                   geojson=geodf.geometry,
                                   locations=geodf.index,
                                   color=years_as_strings,
                                   center={'lat': 12.099568, 'lon': 122.733168},
                                   zoom=4,
                                   color_continuous_scale=custom_color_scale,
                                   range_color=(0, 100))

    map_fig.update_layout(
        font_family="Sansation Regular",
        font_color="black",
        margin=dict(t=0, b=0, r=100, l=0)
    )

    return map_fig


@app.callback(
    Output('choropleth-map', 'clickData'),
    [Input('resetHighlights', 'n_clicks')]
)
def reset_clickdata(nclick):
    if nclick is not None and nclick > 0:
        return None  # Reset click data to None
    return dash.no_update


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
