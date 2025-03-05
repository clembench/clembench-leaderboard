## REQUIRED OUTPUT ###
# A list of version names -> v1.6, v.6_multimodal, v1.6_quantized, v1.5, v0.9, etc......
# A corresponding DataFrame?

import requests
from datetime import datetime
import pandas as pd
import json
from io import StringIO

from src.leaderboard_utils import process_df
from src.assets.text_content import REPO, BENCHMARK_FILE

VARIANTS = ['ascii', 'backends', 'quantized'] # Include other variants if added in the main clembench-runs repo

def get_version_data():
    """
    Read and process data from CSV files of all available versions hosted on GitHub. - https://github.com/clembench/clembench-runs

    Returns:
        version_data:
            -
    """
    base_repo = REPO
    json_url = base_repo + BENCHMARK_FILE
    response = requests.get(json_url)

    # Check if the JSON file request was successful
    if response.status_code != 200:
        print(f"Failed to read JSON file {json_url}: Status Code: {response.status_code}")
        return None, None, None, None

    json_data = response.json()
    versions = json_data['versions']

    version_names = sorted(
        [ver['version'] for ver in versions],
        key=lambda v: list(map(int, v[1:].split('_')[0].split('.'))),  
        reverse=True
    )   

    version_data  = {
        'versions': [],
        'dataframes': []
    }

    for version in version_names:
        base_url = f"{base_repo}{version}/results.csv"
        response = requests.get(base_url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            df = process_df(df)
            df = df.sort_values(by=df.columns[1], ascending=False)  # Sort by clemscore column
            version_data['dataframes'].append(df)
            metadata = {
                'name': version,
                'last_updated': [datetime.strptime(v['last_updated'], '%Y-%m-%d').strftime("%d %b %Y") for v in versions if v['version'] == version],
                'release_date': [datetime.strptime(v['release_date'], '%Y-%m-%d').strftime("%d %b %Y") for v in versions if v['version'] == version]
            } 
            version_data['versions'].append(metadata)

        # Look for variant results file
        version = version.split('_')[0] # Remove _multimodal suffix, and check for other suffixes
        for suffix in VARIANTS:
            base_url = f"{base_repo}{version}_{suffix}/results.csv"
            response = requests.get(base_url)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                df = process_df(df)
                df = df.sort_values(by=df.columns[1], ascending=False)  # Sort by clemscore column
                version_data['dataframes'].append(df)
                metadata = {
                    'name': version + "_" + suffix # Skip Release date and last updated # Not included in becnhmark_runs.json
                } 
                version_data['versions'].append(metadata)

    return version_data


if __name__ == "__main__":
    version_data = get_version_data()
    print(version_data['versions'])
