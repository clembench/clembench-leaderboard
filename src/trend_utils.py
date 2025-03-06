## Fetch Model Registry and clemscores
import requests
import pandas as pd
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from src.assets.text_content import REGISTRY_URL, REPO, BENCHMARK_FILE
from src.leaderboard_utils import get_github_data

# Cut-off date from where to start the trendgraph
START_DATE = '2023-06-01'

# Graph colours
COLOUR_OPEN = 'red'
COLOUR_COMM = 'blue'

def get_param_size(params: str) -> float:
    """Convert parameter size from string to float.

    Args:
        params (str): The parameter size as a string (e.g., '1000B', '1T').

    Returns:
        float: The size of parameters in float.
    """
    if not params:
        param_size = 0
    else:
        if params[-1] == "B":
            param_size = params[:-1]
            param_size = float(param_size)
        elif params[-1] == "T":
            param_size = params[:-1]
            param_size = float(param_size)
            param_size *= 1000
        else:
            print("Not a valid parameter size")

    return param_size

def date_difference(date_str1: str, date_str2: str) -> int:
    """Calculate the difference in days between two dates.

    Args:
        date_str1 (str): The first date as a string in 'YYYY-MM-DD' format.
        date_str2 (str): The second date as a string in 'YYYY-MM-DD' format.

    Returns:
        int: The difference in days between the two dates.
    """
    date_format = "%Y-%m-%d"
    date1 = datetime.strptime(date_str1, date_format)
    date2 = datetime.strptime(date_str2, date_format)
    return (date1 - date2).days


def populate_list(df: pd.DataFrame, abs_diff: float) -> list:
    """Create a list of models based on clemscore differences.

    Args:
        df (pd.DataFrame): DataFrame containing model data.
        abs_diff (float): The absolute difference threshold for clemscore.

    Returns:
        list: A list of model names that meet the criteria.
    """
    l = [df.iloc[0]['model']]
    prev_clemscore = df.iloc[0]['clemscore']
    prev_date = df.iloc[0]['release_date']

    for i in range(1, len(df)):
        curr_clemscore = df.iloc[i]['clemscore']
        curr_date = df.iloc[i]['release_date']
        date_diff = date_difference(curr_date, prev_date)

        if curr_clemscore - prev_clemscore >= abs_diff:
            if date_diff == 0:
                l[-1] = df.iloc[i]['model']
            else:
                l.append(df.iloc[i]['model'])

            prev_clemscore = curr_clemscore
            prev_date = curr_date

    # # Add the last model if the difference between the last and previous date is greater than 15 days
    # last_date = df.iloc[-1]['release_date']
    # if date_difference(last_date, prev_date) > 15:
    #     l.append(df.iloc[-1]['model'])

    return l


def get_models_to_display(result_df: pd.DataFrame, open_dip: float = 0, comm_dip: float = 0) -> tuple:
    """Retrieve models to display based on clemscore differences.

    Args:
        result_df (pd.DataFrame): DataFrame containing model data.
        open_dip (float, optional): Threshold for open models. Defaults to 0.
        comm_dip (float, optional): Threshold for commercial models. Defaults to 0.

    Returns:
        tuple: Two lists of model names (open and commercial).
    """
    open_model_df = result_df[result_df['open_weight']==True]
    comm_model_df = result_df[result_df['open_weight']==False]

    open_model_df = open_model_df.sort_values(by='release_date', ascending=True)
    comm_model_df = comm_model_df.sort_values(by='release_date', ascending=True)
    open_models = populate_list(open_model_df, open_dip)
    comm_models = populate_list(comm_model_df, comm_dip)
    return open_models, comm_models

# Function to interpolate between two colors
def interpolate_color(rank_val, start_color):
    """
    """
    if start_color == 'red':
        hue = 0
    elif start_color == 'blue':
        hue = 240
    else:
        raise KeyError(f"Invalid color selected for trend graph: {start_color}. Please set either red or blue. Alternatively, set hue value in src.trend_utils.interpolate_colour")
    
    saturation = rank_val*100
    value = 70 if rank_val == 1 else 100

    return f"hsv({hue},{saturation},{value})"


def get_trend_data(text_data: dict, model_registry_data: list) -> pd.DataFrame:
    """Process text data frames to extract model information.

    Args:
        text_data (dict): Dict containing DataFrames and version deatils.
        model_registry_data (list): List of dictionaries containing model registry data.

    Returns:
        pd.DataFrame: DataFrame containing processed model data.
    """
    visited = set()  # Track models that have been processed
    result_df = pd.DataFrame(columns=['model', 'clemscore', 'open_weight', 'release_date', 'parameters', 'est_flag', 'version'])

    text_dfs = text_data['dataframes']
    for i in range(len(text_dfs)):
        df = text_dfs[i]
        version = text_data['version_data'][i]['name']
        for i in range(len(df)):
            model_name = df['Model'].iloc[i]
            if model_name not in visited:
                visited.add(model_name)
                for dict_obj in model_registry_data:
                    if dict_obj["model_name"] == model_name:
                        if dict_obj["parameters"] == "" :
                            params = "1000B"
                            est_flag = True
                        else:
                            params = dict_obj['parameters']
                            est_flag = False

                        param_size = get_param_size(params)
                            
                        new_data = {'model': model_name, 'clemscore': df['Clemscore'].iloc[i], 'open_weight':dict_obj['open_weight'],
                                    'release_date': dict_obj['release_date'], 'parameters': param_size, 'est_flag': est_flag, 'version': version}
                        result_df.loc[len(result_df)] = new_data
                        break
    
    return result_df  # Return the compiled DataFrame


def get_plot(df: pd.DataFrame, start_date: str = '2023-06-01', end_date: str = '2024-12-30',
             benchmark_ticks: dict = {}, benchmark_update = {}, **plot_kwargs) -> go.Figure:
    """Generate a scatter plot for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing model data.
        start_date (str, optional): Start date for filtering. Defaults to '2023-06-01'.
        end_date (str, optional): End date for filtering. Defaults to '2024-12-30'.
        benchmark_ticks (dict, optional): Custom benchmark ticks for the version dates. Defaults to {}.
        benchmark_update (dict, optional): Custom benchmark metadata containing last_updated date for the versions. Defaults to {}.
    
    Keyword Args:
        open_dip (float, optional): Threshold for open models' clemscore differences. Max dip in clemscore allowed to be considered in trend.
        comm_dip (float, optional): Threshold for commercial models' clemscore differences. Max dip in clemscore allowed to be considered in trend.
        height (int, optional): Height of the plot in pixels. Adjusted for mobile or desktop views.
        mobile_view (bool, optional): Flag to indicate if the plot should be optimized for mobile display. Defaults to False.

    Returns:
        go.Figure: The generated plot.
    """

    open_dip = plot_kwargs['open_dip']
    comm_dip = plot_kwargs['comm_dip']
    height = plot_kwargs['height']
    width = plot_kwargs['width']

    mobile_view = True if plot_kwargs['mobile_view'] else False

    max_clemscore = df['clemscore'].max()
    # Convert 'release_date' to datetime
    df['Release Date (Model and & Benchmark Version)'] = pd.to_datetime(df['release_date'], format='ISO8601')

    # Filter out data before April 2023/START_DATE
    df = df[df['Release Date (Model and & Benchmark Version)'] >= pd.to_datetime(start_date)]
    open_model_list, comm_model_list = get_models_to_display(df, open_dip, comm_dip)    
    models_to_display = open_model_list + comm_model_list

    # Create a column to indicate if the model should be labeled
    df['label_model'] = df['model'].apply(lambda x: x if x in models_to_display else "")

    # If mobile_view, then show only the models in models_to_display i.e. on the trend line #minimalistic
    if mobile_view:
        df = df[df['model'].isin(models_to_display)]

    versions = df['version'].unique()
    version_names = sorted(
        [ver for ver in versions],
        key=lambda v: list(map(int, v[1:].split('_')[0].split('.'))),  
        reverse=True
    ) 

    version_names = version_names[:3] # Select 3 latest benchmark versions
    df = df[df['version'].isin(tuple(version_names))]

    rank = 2
    max_rank = len(version_names)
    rank_value = {version_names[0]: 1}
    for ver in version_names:
        if ver not in rank_value:
            rank_value[ver] = 1 - (rank-1-(max_rank/15))/(max_rank-1)
            rank += 1

    df['color_value'] = df.apply(
        lambda row: rank_value[row['version']],
        axis=1
    )

    # Add an identifier column to each DataFrame
    df['Model Type & Benchmark Version'] = df.apply(
        lambda row: f"Open-Weight {row['version']}" if row['open_weight'] else f"Commercial {row['version']}",
        axis=1
    )

    color_map = {}
    for i in range(len(df)):
        if df.iloc[i]['Model Type & Benchmark Version'] not in color_map:
            if df.iloc[i]['open_weight']:
                color_map[df.iloc[i]['Model Type & Benchmark Version']] = interpolate_color(df.iloc[i]['color_value'], COLOUR_OPEN)
            else:
                color_map[df.iloc[i]['Model Type & Benchmark Version']] = interpolate_color(df.iloc[i]['color_value'], COLOUR_COMM)

    
    marker_size = df['parameters'].apply(lambda x: np.sqrt(x) if x > 0 else np.sqrt(400)).astype(float)  # Arbitrary sqrt value to scale marker size based on parameter size

    # Create the scatter plot
    fig = px.scatter(df,
                    x="Release Date (Model and & Benchmark Version)",
                    y="clemscore",
                    color="Model Type & Benchmark Version",  # Differentiates the datasets by color
                    color_discrete_map=color_map,  # Map colors to the defined subclasses
                    hover_name="model",
                    size=marker_size,
                    size_max=40,  # Max size of the circles
                    template="plotly_white",
                    hover_data={  # Customize hover information
                        "Release Date (Model and & Benchmark Version)": True,  # Show the Release Date (Model and & Benchmark Version)
                        "clemscore": True,  # Show the clemscore
                        "version": True
                    },
                    custom_data=["model", "Release Date (Model and & Benchmark Version)", "clemscore", "version"],  # Specify custom data columns for hover
                    opacity=0.8
                    )

    fig.update_traces(
        hovertemplate='Model Name: %{customdata[0]}<br>Release Date (Model and & Benchmark Version): %{customdata[1]}<br>Clemscore: %{customdata[2]}<br>Benchmark Version: %{customdata[3]}<br>'
    )
    
    # Sort dataframes for line plotting
    df_open = df[df['model'].isin(open_model_list)].sort_values(by='Release Date (Model and & Benchmark Version)')
    df_commercial = df[df['model'].isin(comm_model_list)].sort_values(by='Release Date (Model and & Benchmark Version)')

    ## Custom tics for x axis
    # Define the start and end dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # Generate ticks every two months
    date_range = pd.date_range(start=start_date, end=end_date, freq='2MS')  # '2MS' stands for 2 Months Start frequency
    # Create labels for these ticks
    custom_ticks = {date: date.strftime('%b %Y') for date in date_range}

    ## Benchmark Version ticks
    benchmark_tickvals = list(pd.to_datetime(list(benchmark_ticks.keys())))
    custom_ticks = {k:v for k,v in custom_ticks.items() if k not in benchmark_tickvals}
    custom_tickvals = list(custom_ticks.keys())

    if mobile_view:
        # Remove custom_tickvals within -1 month to +1 month of benchmark_tickvals for better visibility
        one_month = pd.DateOffset(months=1)
        filtered_custom_tickvals = [
            date for date in custom_tickvals 
            if not any((benchmark_date - one_month <= date <= benchmark_date + one_month) for benchmark_date in benchmark_tickvals)
        ]
        # Alternate <br> for benchmark ticks based on date difference (Eg. v1.6, v1.6.5 too close to each other for MM benchmark)
        benchmark_tick_texts = []
        for i in range(len(benchmark_tickvals)):
            if i == 0:
                benchmark_tick_texts.append(f"<br><br><b>{benchmark_ticks[benchmark_tickvals[i]]}</b>")
            else:
                date_diff = (benchmark_tickvals[i] - benchmark_tickvals[i - 1]).days
                if date_diff <= 75:
                    benchmark_tick_texts.append(f"<br><br><br><b>{benchmark_ticks[benchmark_tickvals[i]]}</b>")
                else:
                    benchmark_tick_texts.append(f"<br><br><b>{benchmark_ticks[benchmark_tickvals[i]]}</b>")
        fig.update_xaxes(
            tickvals=filtered_custom_tickvals + benchmark_tickvals,  # Use filtered_custom_tickvals
            ticktext=[f"{date.strftime('%b')}<br>{date.strftime('%y')}" for date in filtered_custom_tickvals] + 
                      benchmark_tick_texts,  # Use the new benchmark tick texts
            tickangle=0,
            tickfont=dict(size=10)
        )
        fig.update_yaxes(range=[0, 110]) # Set y-axis range to 110 for better visibility of legend and avoiding overlap with interactivity block of plotly on top-right
        display_mode = 'lines+markers'
    else:
        fig.update_xaxes(
            tickvals=custom_tickvals + benchmark_tickvals,  # Use filtered_custom_tickvals
            ticktext=[f"{date.strftime('%b')} {date.strftime('%Y')}" for date in custom_tickvals] + 
                    [f"<br><span style='font-size:12px;'><b>{benchmark_ticks[date]}</b></span>" for date in benchmark_tickvals],  # Added <br> for vertical alignment
            tickangle=0,
            tickfont=dict(size=10)  
        )
        fig.update_yaxes(range=[0, max_clemscore+10])
        display_mode = 'lines+markers+text'


    # Add lines connecting the points for open models
    fig.add_trace(go.Scatter(x=df_open['Release Date (Model and & Benchmark Version)'], y=df_open['clemscore'],
                            mode=display_mode,  # Include 'text' in the mode
                            name='Open Models Trendline',
                            text=df_open['label_model'],  # Use label_model for text labels
                            textposition='top center',  # Position of the text labels
                            line=dict(color='red'), showlegend=False))

    # Add lines connecting the points for commercial models
    fig.add_trace(go.Scatter(x=df_commercial['Release Date (Model and & Benchmark Version)'], y=df_commercial['clemscore'],
                            mode=display_mode,  # Include 'text' in the mode
                            name='Commercial Models Trendline',
                            text=df_commercial['label_model'],  # Use label_model for text labels
                            textposition='top center',  # Position of the text labels
                            line=dict(color='blue'), showlegend=False))


    # Update layout to ensure text labels are visible   
    fig.update_traces(textposition='top center')

    # Update the Legend Position and plot dimensions
    fig.update_layout(height=height,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ) 
    )

    if width:
        print("Custom Seting the Width :")
        fig.update_layout(width=width)

    return fig

def get_final_trend_plot(benchmark: str = "Text", mobile_view: bool = False) -> go.Figure:
    """Fetch and generate the final trend plot for all models.

    Args:
        benchmark (str, optional): The benchmark type to use. Defaults to "Text".
        mobile_view (bool, optional): Flag to indicate mobile view. Defaults to False.

    Returns:
        go.Figure: The generated trend plot for selected benchmark.
    """
    # Fetch Model Registry
    response = requests.get(REGISTRY_URL)
    model_registry_data = response.json()
    # Custom tick labels
    json_url = REPO + BENCHMARK_FILE
    response = requests.get(json_url)

    # Check if the JSON file request was successful
    if response.status_code != 200:
        print(f"Failed to read JSON file {json_url}: Status Code: {response.status_code}")

    json_data = response.json()
    versions = json_data['versions']

    if mobile_view:
        height = 450
        width = 375
    else:
        height = 1000
        width = None

    plot_kwargs = {'height': height, 'width': width, 'open_dip': 0, 'comm_dip': 0,
                   'mobile_view': mobile_view}

    benchmark_ticks = {}
    benchmark_update = {}
    if benchmark == "Text":
        text_data = get_github_data()['text']
        text_result_df = get_trend_data(text_data, model_registry_data)
        ## Get benchmark tickvalues as dates for X-axis
        for ver in versions:
            if 'multimodal' not in ver['version']: # Skip MM specific benchmark dates
                benchmark_ticks[pd.to_datetime(ver['release_date'])] = ver['version']
                if pd.to_datetime(ver['last_updated']) not in benchmark_update:
                    benchmark_update[pd.to_datetime(ver['last_updated'])] = [ver['version']]
                else:
                    benchmark_update[pd.to_datetime(ver['last_updated'])].append(ver['version'])

        fig =  get_plot(text_result_df, start_date=START_DATE, end_date=datetime.now().strftime('%Y-%m-%d'), benchmark_ticks=benchmark_ticks, benchmark_update=benchmark_update, **plot_kwargs)
    else:
        mm_data = get_github_data()['multimodal']
        result_df = get_trend_data(mm_data, model_registry_data)
        df = result_df
        for ver in versions:
            if 'multimodal' in ver['version']:
                temp_ver = ver['version']
                temp_ver = temp_ver.replace('_multimodal', '')
                benchmark_ticks[pd.to_datetime(ver['release_date'])] = temp_ver ## MM benchmark dates considered after v1.6 (incl.)
                benchmark_update[pd.to_datetime(ver['last_updated'])] = temp_ver

        fig = get_plot(df, start_date=START_DATE, end_date=datetime.now().strftime('%Y-%m-%d'), benchmark_ticks=benchmark_ticks, benchmark_update=benchmark_update, **plot_kwargs)

    return fig
