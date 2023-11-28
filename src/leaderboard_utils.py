import os
import pandas as pd

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
    versions_list = os.listdir(csvs_path)
    versions_list = [s for s in versions_list if s.startswith('v') and s.endswith('.csv')]
    versions_list = [s.split('.csv')[0] for s in versions_list]
    # Sort by latest version
    float_content = [float(s[1:]) for s in versions_list]
    float_content.sort(reverse=True)
    versions_list = ['v'+str(s) for s in float_content]

    DFS = []
    for csv in versions_list:
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
    for df, name in zip(DFS, versions_list):
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