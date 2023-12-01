import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.assets.text_content import SHORT_NAMES

def plotly_plot(df:pd.DataFrame, LIST:list):
    '''
    Takes in a list of models for a plotly plot
    Args:
        df: A dummy dataframe of latest version
        LIST: List of models to plot
    Returns:
        Fig: plotly figure
    '''
    
    short_names = label_map(LIST)
    list_columns = list(df.columns)

    # Filter dataframe based on the provided list of models
    df = df[df[list_columns[0]].isin(LIST)]


    fig = px.scatter(df, x=list_columns[2], y=list_columns[3], color=list_columns[0], symbol=list_columns[0],
                 color_discrete_map={"category1": "blue", "category2": "red"},
                 hover_name=list_columns[0], template="plotly_white")
        
    fig.update_layout(
        xaxis_title='% Played',
        yaxis_title='Quality Score',
        title='Overview of benchmark results',
    )

    fig.update_xaxes(range=[-10, 110])
    fig.update_yaxes(range=[-10, 110])

    return fig

def matplotlib_plot(df:pd.DataFrame, LIST:list):
    '''
    Takes in a list of models for a matplotlib plot
    Args:
        df: A dummy dataframe of latest version
        LIST: List of models to plot
    Returns:
        Fig: matplotlib figure
    '''
    short_names = label_map(LIST)
    list_columns = list(df.columns)
    
    df = df[df[list_columns[0]].isin(LIST)]

    X = df[list_columns[2]]
    fig, ax = plt.subplots()
    for model in LIST:
        short = short_names[model]
        model_df = df[df[list_columns[0]] == model]
        x = model_df[list_columns[2]]
        y = model_df[list_columns[3]]
        color = plt.cm.rainbow(x / max(X))  # Use a colormap for different colors
        plt.scatter(x, y, color=color)
        plt.annotate(f'{short}', (x, y), textcoords="offset points", xytext=(0, -15), ha='center', rotation=0)
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    ax.set_xticks(np.arange(0,110,10))
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    plt.xlabel('% Played')
    plt.ylabel('Quality Score')
    plt.title('Overview of benchmark results')
    plt.show()

    return fig


# ['Model', 'Clemscore', 'All(Played)', 'All(Quality Score)']
def compare_plots(df: pd.DataFrame, LIST1: list, LIST2: list):
    '''
    Quality Score v/s % Played plot by selecting models
    Args:
        df: A dummy dataframe of latest version
        LIST1: The list of open source models to show in the plot, updated from frontend
        LIST2: The list of commercial models to show in the plot, updated from frontend
    Returns:
        fig: The plot
    '''
    # Combine lists for Open source and commercial models
    LIST = LIST1 + LIST2
    fig = plotly_plot(df, LIST)
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
        short_name: A map from long to list of short name + indication if models are same or different
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
