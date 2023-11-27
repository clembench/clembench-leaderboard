import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.assets.text_content import SHORT_NAMES

# Set the folder name to save csv files
global csvs_path
csvs_path = 'versions'

def get_csv_data():
    '''
    Get data from csv files saved locally
    Args:
        None
    Returns: 
        latest_df: singular list containing dataframe of the latest version of the leaderboard with only 4 columns 
        all_dfs: list of dataframes for previous versions + latest version including columns for all games
        all_vnames: list of the names for the previous versions + latest version (For Details and Versions Tab Dropdown)
    '''
    list_vers = os.listdir(csvs_path)
    list_vers = [s.split('.csv')[0] for s in list_vers]
    # Sort by latest version
    float_content = [float(s[1:]) for s in list_vers]
    float_content.sort(reverse=True)
    list_vers = ['v'+str(s) for s in float_content]

    DFS = []
    for csv in list_vers:
        read_path = os.path.join(csvs_path, csv + '.csv')
        df = pd.read_csv(read_path)
        df = process_df(df)
        df = df.sort_values(by=list(df.columns)[1], ascending=False) # Sort by clemscore
        DFS.append(df)

    # Only keep relavant columns for the main leaderboard
    latest_df_dummy = DFS[0]
    all_columns = list(latest_df_dummy.columns)
    keep_columns = all_columns[0:4]
    latest_df_dummy = latest_df_dummy.drop(columns=[c for c in all_columns if c not in keep_columns])

    latest_df = [latest_df_dummy]
    all_dfs = []
    all_vnames = []
    for df, name in zip(DFS, list_vers):
        all_dfs.append(df)
        all_vnames.append(name) 

    return latest_df, all_dfs, all_vnames

def process_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Process dataframe 
    - Remove repition in model names 
    - Convert datatypes to sort by "float" instead of "str" for sorting
    - Update column names
    Args:
        df: Unprocessed Dataframe (after using update_cols)
    Returns:
        df: Processed Dataframe
    '''

    # Change column type to float from str
    list_column_names = list(df.columns)
    model_col_name = list_column_names[0]
    for col in list_column_names:
        if col != model_col_name:
            df[col] = df[col].astype(float)

    # Remove repetition in model names, if any
    models_list = []
    for i in range(len(df)):
        model_name = df.iloc[i][model_col_name]
        splits = model_name.split('--')
        splits = [split.replace('-t0.0', '') for split in splits] # Comment to not remove -t0.0
        if splits[0] == splits[1]:
            models_list.append(splits[0])
        else:
            models_list.append(splits[0] + "--" + splits[1])
    df[model_col_name] = models_list

    # Update column names
    update = ['Model', 'Clemscore', '% Played', 'Quality Score']
    game_metrics = list_column_names[4:]

    for col in game_metrics:
        splits = col.split(',')
        update.append(splits[0].capitalize() + "" + splits[1])
    
    map_cols = {}
    for i in range(len(update)):
        map_cols[list_column_names[i]] = str(update[i])

    df = df.rename(columns=map_cols)    
    return df

def filter_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    '''
    Filter the dataframe based on the search query
    Args:
        df: Unfiltered dataframe
        query: a string of queries separated by ";"
    Return:
        filtered_df: Dataframe containing searched queries in the 'Model' column 
    '''
    queries = query.split(';')
    list_cols = list(df.columns)
    df_len = len(df)
    filtered_models = []
    models_list = list(df[list_cols[0]])
    for q in queries:
        q = q.lower()
        q = q.strip()
        for i in range(df_len):
            model_name = models_list[i]
            if q in model_name.lower():
                filtered_models.append(model_name) # Append model names containing query q

    filtered_df = df[df[list_cols[0]].isin(filtered_models)]

    if query == "":
        return df

    return filtered_df

###################################FOR PLOTS##################################################

def plot_graph(df:pd.DataFrame, LIST:list):
    '''
    Takes in a list of models to plot
    Args:
        df: A dummy dataframe of latest version
        LIST: List of models to plot
    Returns:
        Fig: figure to plot
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
    fig = plot_graph(df, LIST)
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
            open_models.append(model)
        else:
            comm_models.append(model)

    open_models.sort(key=lambda o: o.upper())
    comm_models.sort(key=lambda c: c.upper())
    return open_models, comm_models




