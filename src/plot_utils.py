import pandas as pd
import plotly.express as px

from src.assets.text_content import SHORT_NAMES

def plotly_plot(df:pd.DataFrame, LIST:list, ALL:list, NAMES:list):
    '''
    Takes in a list of models for a plotly plot
    Args:
        df: A dummy dataframe of latest version
        LIST: List of models to plot
        ALL: Either [] or ["Show All Models"] - toggle view to plot all models 
        NAMES: Either [] or ["Show Names"] - toggle view to show model names on plot 
    Returns:
        Fig: plotly figure
    '''
    
    # Get list of all models and append short names column to df
    list_columns = list(df.columns)
    ALL_LIST = list(df[list_columns[0]].unique())
    short_names = label_map(ALL_LIST)
    list_short_names = list(short_names.values())
    df["Short"] = list_short_names

    if ALL:
        LIST = ALL_LIST
    # Filter dataframe based on the provided list of models
    df = df[df[list_columns[0]].isin(LIST)]
    

    if NAMES:
        fig = px.scatter(df, x=list_columns[2], y=list_columns[3], color=list_columns[0], symbol=list_columns[0],
                 color_discrete_map={"category1": "blue", "category2": "red"},
                 hover_name=list_columns[0], template="plotly_white", text="Short")
        fig.update_traces(textposition='top center')
    else:
        fig = px.scatter(df, x=list_columns[2], y=list_columns[3], color=list_columns[0], symbol=list_columns[0],
                    color_discrete_map={"category1": "blue", "category2": "red"},
                    hover_name=list_columns[0], template="plotly_white")
        
    fig.update_layout(
        xaxis_title='% Played',
        yaxis_title='Quality Score',
        title='Overview of benchmark results',
        height=1000,
        width=1500
    )

    fig.update_xaxes(range=[-5, 105])
    fig.update_yaxes(range=[-5, 105])

    return fig


# ['Model', 'Clemscore', 'All(Played)', 'All(Quality Score)']
def compare_plots(df: pd.DataFrame, LIST1: list, LIST2: list, ALL:list, NAMES:list):
    '''
    Quality Score v/s % Played plot by selecting models
    Args:
        df: A dummy dataframe of latest version
        LIST1: The list of open source models to show in the plot, updated from frontend
        LIST2: The list of commercial models to show in the plot, updated from frontend
        ALL: Either [] or ["Show All Models"] - toggle view to plot all models 
        NAMES: Either [] or ["Show Names"] - toggle view to show model names on plot 
    Returns:
        fig: The plot
    '''

    # Combine lists for Open source and commercial models
    LIST = LIST1 + LIST2
    fig = plotly_plot(df, LIST, ALL, NAMES)

    return fig
    
def shorten_model_name(full_name):
    # Split the name into parts
    parts = full_name.split('-')

    # Process the name parts to keep only the parts with digits (model sizes and versions)
    short_name_parts = [part for part in parts if any(char.isdigit() for char in part)]

    if len(parts) == 1:
        short_name = ''.join(full_name[0:min(3, len(full_name))])
    else:
        # Join the parts to form the short name
        short_name = '-'.join(short_name_parts)

        # Remove any leading or trailing hyphens
        short_name = full_name[0] + '-'+ short_name.strip('-')

    return short_name

def label_map(model_list: list) -> dict:
    '''
    Generate a map from long names to short names, to plot them in frontend graph
    Define the short names in src/assets/text_content.py
    Args: 
        model_list: A list of long model names
    Returns:
        short_name: A dict from long to short name
    '''
    short_names = {}
    for model_name in model_list:
        if model_name in SHORT_NAMES:
            short_name = SHORT_NAMES[model_name]
        else:
            short_name = shorten_model_name(model_name)

        # Define the short name and indicate both models are same
        short_names[model_name] = short_name

    return short_names
    
def split_models(MODEL_LIST: list):
    '''
    Split the models into open source and commercial
    '''
    open_models = []
    comm_models = []

    for model in MODEL_LIST:
        if model.startswith(('gpt-', 'claude-', 'command')):
            comm_models.append(model)
        else:
            open_models.append(model)

    open_models.sort(key=lambda o: o.upper())
    comm_models.sort(key=lambda c: c.upper())
    return open_models, comm_models
