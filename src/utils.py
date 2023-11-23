import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.assets.text_content import SHORT_NAMES

def update_cols(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Change three header rows to a single header row
    Args:
        df: Raw dataframe containing 3 separate header rows
            Remove this function if the dataframe has only one header row

    Returns:
        df: Updated dataframe which has only 1 header row instead of 3
    '''
    default_cols = list(df.columns)

    # First 4 columns are initalised in 'update', Append additional columns for games Model, Clemscore, ALL(PLayed) and ALL(Main Score)
    update = ['Model', 'Clemscore', 'Played', 'Quality Score']
    game_metrics = default_cols[4:]

    # Change columns Names for each Game
    for i in range(len(game_metrics)):
        if i%3 == 0:
            game = game_metrics[i]
            update.append(str(game).capitalize() + "(Played)")
            update.append(str(game).capitalize() + "(Quality Score)") 
            update.append(str(game).capitalize() + "(Quality Score[std])")

    # Create a dict to change names of the columns
    map_cols = {}
    for i in range(len(default_cols)):
        map_cols[default_cols[i]] = str(update[i])

    df = df.rename(columns=map_cols)
    df = df.iloc[2:]

    return df

def process_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Process dataframe - Remove repition in model names, convert datatypes to sort by "float" instead of "str"
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
    
    return df

def get_data(path: str, flag: bool):
    '''
    Get a list of all version names and respective Dataframes 
    Args: 
        path: Path to the directory containing CSVs of different versions -> v0.9.csv, v1.0.csv, ....
        flag: Set this flag to include the latest version in Details and Versions tab
    Returns: 
        latest_df: singular list containing dataframe of the latest version of the leaderboard with only 4 columns 
        latest_vname: list of the name of latest version 
        previous_df: list of dataframes for previous versions (can skip latest version if required) 
        previous_vname: list of the names for the previous versions (INCLUDED IN Details and Versions Tab)

    '''
    # Check if Directory is empty
    list_versions = os.listdir(path)
    if not list_versions:
        print("Directory is empty")

    else:
        files = [file for file in list_versions if file.endswith('.csv')]
        files.sort(reverse=True)
        file_names = [os.path.splitext(file)[0] for file in files]

        DFS = []
        for file in files:
            df = pd.read_csv(os.path.join(path, file))
            df = update_cols(df) # Remove if by default there is only one header row
            df = process_df(df) # Process Dataframe
            df = df.sort_values(by=list(df.columns)[1], ascending=False) # Sort by clemscore
            DFS.append(df)

        # Only keep relavant columns for the main leaderboard
        latest_df_dummy = DFS[0]
        all_columns = list(latest_df_dummy.columns)
        keep_columns = all_columns[0:4]
        latest_df_dummy = latest_df_dummy.drop(columns=[c for c in all_columns if c not in keep_columns])

        latest_df = [latest_df_dummy]
        latest_vname = [file_names[0]]
        previous_df = []
        previous_vname = []
        for df, name in zip(DFS, file_names):
            previous_df.append(df)
            previous_vname.append(name) 
        
        if not flag:
            previous_df.pop(0)
            previous_vname.pop(0)

        return latest_df, latest_vname, previous_df, previous_vname
    
    return None


# ['Model', 'Clemscore', 'All(Played)', 'All(Quality Score)']
def compare_plots(df: pd.DataFrame, LIST: list):
    '''
    Quality Score v/s % Played plot by selecting models
    Args:
        LIST: The list of models to show in the plot, updated from frontend
    Returns:
        fig: The plot
    '''
    short_names = label_map(LIST)

    list_columns = list(df.columns)
    df = df[df[list_columns[0]].isin(LIST)]

    X = df[list_columns[2]]
    fig, ax = plt.subplots()
    for model in LIST:
        short = short_names[model]
        # same_flag = short_names[model][1]
        model_df = df[df[list_columns[0]] == model]
        x = model_df[list_columns[2]]
        y = model_df[list_columns[3]]
        color = plt.cm.rainbow(x / max(X))  # Use a colormap for different colors
        plt.scatter(x, y, color=color)
        # if same_flag:
        plt.annotate(f'{short}', (x, y), textcoords="offset points", xytext=(0, -15), ha='center', rotation=0)
        # else:
        #     plt.annotate(f'{short}', (x, y), textcoords="offset points", xytext=(20, -3), ha='center', rotation=0)
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    ax.set_xticks(np.arange(0,110,10))
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    plt.xlabel('% Played')
    plt.ylabel('Quality Score')
    plt.title('Overview of benchmark results')
    plt.show()

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
        # splits = model_name.split('--')
        # if len(splits) != 1:
        #     splits[0] = SHORT_NAMES[splits[0] + '-']
        #     splits[1] = SHORT_NAMES[splits[1] + '-']
        #     # Define the short name and indicate there are two different models
        #     short_names[model_name] = [splits[0] + '--' + splits[1], 0]
        # else:
        if model_name in SHORT_NAMES:
            short_name = SHORT_NAMES[model_name]
        else:
            short_name = shorten_model_name(model_name)

        # Define the short name and indicate both models are same
        short_names[model_name] = short_name

    return short_names

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
        for i in range(df_len):
            model_name = models_list[i]
            if q in model_name.lower():
                filtered_models.append(model_name) # Append model names containing query q

    filtered_df = df[df[list_cols[0]].isin(filtered_models)]

    if query == "":
        return df

    return filtered_df
    
