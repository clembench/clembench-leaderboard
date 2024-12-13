import os
import pandas as pd
import requests
import json
from io import StringIO
from datetime import datetime

from src.assets.text_content import REPO, BENCHMARK_FILE

def get_github_data():
    """
    Read and process data from CSV files hosted on GitHub. - https://github.com/clembench/clembench-runs (REPO)
    Set the path in src/assets/text_content/REPO

    Returns:
        github_data (dict): Dictionary containing:
            - "text": List of DataFrames for each version's textual leaderboard data.
            - "multimodal": List of DataFrames for each version's multimodal leaderboard data.
            - "date": Formatted date of the latest version in "DD Month YYYY" format.
    """
    json_url = REPO + BENCHMARK_FILE
    response = requests.get(json_url)

    # Check if the JSON file request was successful
    if response.status_code != 200:
        print(f"Failed to read JSON file - {BENCHMARK_FILE} in repo {REPO}: Status Code: {response.status_code}")
        return None, None, None, None

    json_data = response.json()
    versions = json_data['versions']

    # Sort the versions in benchmark by latest first
    version_names = sorted(
        [ver['version'] for ver in versions],
        key=lambda v: list(map(int, v[1:].split('_')[0].split('.'))),  
        reverse=True
    )   

    # Collect Dataframes - Text and Multimodal Only - Ignoring _quantized, _backends, _ascii
    text_data = {
        'version_data': [],
        'dataframes': []
    }
    multimodal_data = {
        'version_data': [],
        'dataframes': []
    }

    for version in version_names:
        results_url = f"{REPO}{version}/results.csv"
        csv_response = requests.get(results_url)
        if csv_response.status_code == 200:
            df = pd.read_csv(StringIO(csv_response.text))
            df = process_df(df)
            df = df.sort_values(by=df.columns[1], ascending=False) # Sort by Clemscore

            version_data = {
                'name': version,
                'last_updated': [datetime.strptime(v['last_updated'], '%Y-%m-%d').strftime("%d %b %Y") for v in versions if v['version'] == version],
                'release_date': [datetime.strptime(v['release_date'], '%Y-%m-%d').strftime("%d %b %Y") for v in versions if v['version'] == version]
            } 

            if 'multimodal' in version:
                multimodal_data['dataframes'].append(df)
                multimodal_data['version_data'].append(version_data)
            else:
                text_data['dataframes'].append(df)
                text_data['version_data'].append(version_data)

      
    github_data = {
        'text': text_data,
        'multimodal': multimodal_data
    }

    return github_data


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process dataframe:
    - Convert datatypes to sort by "float" instead of "str"
    - Remove repetition in model names
    - Update column names

    Args:
        df: Unprocessed Dataframe (after using update_cols)

    Returns:
        df: Processed Dataframe
    """

    # Convert column values to float, apart from the model names column
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove repetition in model names
    df[df.columns[0]] = df[df.columns[0]].str.replace('-t0.0', '', regex=True)
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: '--'.join(set(x.split('--'))))

    # Update column names
    custom_column_names = ['Model', 'Clemscore', '% Played', 'Quality Score']
    for i, col in enumerate(df.columns[4:]):  # Start Capitalizing from the 5th column
        parts = col.split(',')
        custom_name = f"{parts[0].strip().capitalize()} {parts[1].strip()}"
        custom_column_names.append(custom_name)

    # Rename columns
    df.columns = custom_column_names

    return df


def query_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Filter the dataframe based on the search query.

    Args:
        df (pd.DataFrame): Unfiltered dataframe.
        query (str): A string of queries separated by ";".
    Returns:
        pd.DataFrame: Filtered dataframe containing searched queries in the 'Model' column.
    """
    if not query.strip():  # Reset Dataframe if empty query is passed
        return df

    queries = [q.strip().lower() for q in query.split(';') if q.strip()]  # Normalize and split queries

    # Filter dataframe based on queries in 'Model' column
    filtered_df = df[df['Model'].str.lower().str.contains('|'.join(queries))]

    return filtered_df

if __name__=='__main__':
    data = get_github_data()
    print(data['text']['version_data'])
    print(data['multimodal']['version_data'])
