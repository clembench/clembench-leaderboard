import pandas as pd
import plotly.express as px
import requests
import json
import gradio as gr

from src.assets.text_content import SHORT_NAMES, TEXT_NAME, MULTIMODAL_NAME
from src.leaderboard_utils import get_github_data


def plotly_plot(df: pd.DataFrame, list_op: list, list_co: list,
                show_all: list, show_names: list, show_legend: list,
                mobile_view: list):
    """
    Takes in a list of models for a plotly plot
    Args:
        df: A dummy dataframe of latest version
        list_op: The list of open source models to show in the plot, updated from frontend
        list_co: The list of commercial models to show in the plot, updated from frontend
        show_all: Either [] or ["Show All Models"] - toggle view to plot all models 
        show_names: Either [] or ["Show Names"] - toggle view to show model names on plot 
        show_legend: Either [] or ["Show Legend"] - toggle view to show legend on plot
        mobile_view: Either [] or ["Mobile View"] - toggle view to for smaller screens
    Returns:
        Fig: plotly figure of % played v/s quality score
    """

    LIST = list_op + list_co
    # Get list of all models and append short names column to df
    list_columns = list(df.columns)
    ALL_LIST = list(df[list_columns[0]].unique())
    short_names = label_map(ALL_LIST)
    list_short_names = list(short_names.values())
    df["Short"] = list_short_names

    if show_all:
        LIST = ALL_LIST
    # Filter dataframe based on the provided list of models
    df = df[df[list_columns[0]].isin(LIST)]

    if show_names:
        fig = px.scatter(df, x=list_columns[2], y=list_columns[3], color=list_columns[0], symbol=list_columns[0],
                         color_discrete_map={"category1": "blue", "category2": "red"},
                         hover_name=list_columns[0], template="plotly_white", text="Short")
        fig.update_traces(textposition='top center')
    else:
        fig = px.scatter(df, x=list_columns[2], y=list_columns[3], color=list_columns[0], symbol=list_columns[0],
                         color_discrete_map={"category1": "blue", "category2": "red"},
                         hover_name=list_columns[0], template="plotly_white")

    if not show_legend:
        fig.update_layout(showlegend=False)

    fig.update_layout(
        xaxis_title='% Played',
        yaxis_title='Quality Score',
        title='Overview of benchmark results',
        height=1000
    )

    fig.update_xaxes(range=[-5, 105])
    fig.update_yaxes(range=[-5, 105])

    if mobile_view:
        fig.update_layout(height=300)

    if mobile_view and show_legend:
        fig.update_layout(height=450)
        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=-5.52,
            xanchor="left",
            x=0.01
        ))

        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            title="% Played v/s Quality Score"
        )

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
        short_name = full_name[0] + '-' + short_name.strip('-')

    return short_name


def label_map(model_list: list) -> dict:
    """
    Generate a map from long names to short names, to plot them in frontend graph
    Define the short names in src/assets/text_content.py
    Args: 
        model_list: A list of long model names
    Returns:
        short_name: A dict from long to short name
    """
    short_names = {}
    for model_name in model_list:
        if model_name in SHORT_NAMES:
            short_name = SHORT_NAMES[model_name]
        else:
            short_name = shorten_model_name(model_name)

        # Define the short name and indicate both models are same
        short_names[model_name] = short_name

    return short_names


def split_models(model_list: list):
    """
    Split the models into open source and commercial
    """

    open_models = []
    commercial_models = []
    open_backends = {"huggingface_local", "huggingface_multimodal", "openai_compatible"}  # Define backends considered as open

    # Load model registry data from main repo
    model_registry_url = "https://raw.githubusercontent.com/clp-research/clembench/main/backends/model_registry.json"
    response = requests.get(model_registry_url)

    if response.status_code == 200:
        json_data = json.loads(response.text)
        # Classify as Open or Commercial based on the defined backend in the model registry
        backend_mapping = {}

        for model_name in model_list:
            model_prefix = model_name.split('-')[0]  # Get the prefix part of the model name
            for entry in json_data:
                if entry["model_name"].startswith(model_prefix):
                    backend = entry["backend"]
                    # Classify based on backend
                    if backend in open_backends:
                        open_models.append(model_name)
                    else:
                        commercial_models.append(model_name)
                    break

    else:
        print(f"Failed to read JSON file: Status Code : {response.status_code}")

    open_models.sort(key=lambda o: o.upper())
    commercial_models.sort(key=lambda c: c.upper())

    # Add missing model from the model_registry
    if "dolphin-2.5-mixtral-8x7b" in model_list:
        open_models.append("dolphin-2.5-mixtral-8x7b")

    return open_models, commercial_models

"""
Update Functions, for when the leaderboard selection changes
"""
def update_open_models(leaderboard: str = TEXT_NAME):
    """
    Change the checkbox group of Open Models based on the leaderboard selected

    Args:
        leaderboard: Selected leaderboard from the frontend [Default - Text Leaderboard]
    Return:
        Updated checkbox group for Open Models, based on the leaderboard selected
    """
    github_data = get_github_data()
    leaderboard_data = github_data["text" if leaderboard == TEXT_NAME else "multimodal"][0]
    models = leaderboard_data.iloc[:, 0].unique().tolist()
    open_models, commercial_models = split_models(models)
    return gr.CheckboxGroup(
        open_models,
        value=[],
        elem_id="value-select-1",
        interactive=True,
    )

def update_closed_models(leaderboard: str = TEXT_NAME):
    """
    Change the checkbox group of Closed Models based on the leaderboard selected

    Args:
        leaderboard: Selected leaderboard from the frontend [Default - Text Leaderboard]
    Return:
        Updated checkbox group for Closed Models, based on the leaderboard selected
    """
    github_data = get_github_data()
    leaderboard_data = github_data["text" if leaderboard == TEXT_NAME else "multimodal"][0]
    models = leaderboard_data.iloc[:, 0].unique().tolist()
    open_models, commercial_models = split_models(models)
    return gr.CheckboxGroup(
        commercial_models,
        value=[],
        elem_id="value-select-2",
        interactive=True,
    )

def get_plot_df(leaderboard: str = TEXT_NAME) -> pd.DataFrame:
    """
    Get the DataFrame for plotting based on the selected leaderboard.
    Args:
        leaderboard: Selected leaderboard.
    Returns:
        DataFrame with model data.
    """
    github_data = get_github_data()
    return github_data["text" if leaderboard == TEXT_NAME else "multimodal"][0]


"""
Reset Functions for when the Leaderboard selection changes
"""
def reset_show_all():
    return gr.CheckboxGroup(
            ["Select All Models"],
            label="Show plot for all models 🤖",
            value=[],
            elem_id="value-select-3",
            interactive=True,
        )

def reset_show_names():
    return gr.CheckboxGroup(
        ["Show Names"],
        label="Show names of models on the plot 🏷️",
        value=[],
        elem_id="value-select-4",
        interactive=True,
    )


def reset_show_legend():
    return gr.CheckboxGroup(
        ["Show Legend"],
        label="Show legend on the plot 💡",
        value=[],
        elem_id="value-select-5",
        interactive=True,
    )


def reset_mobile_view():
    return gr.CheckboxGroup(
        ["Mobile View"],
        label="View plot on smaller screens 📱",
        value=[],
        elem_id="value-select-6",
        interactive=True,
    )


if __name__ == '__main__':
    mm_model_list = ['gpt-4o-2024-05-13', 'gpt-4-1106-vision-preview', 'claude-3-opus-20240229', 'gemini-1.5-pro-latest',
                     'gemini-1.5-flash-latest', 'llava-v1.6-34b-hf', 'llava-v1.6-vicuna-13b-hf', 'idefics-80b-instruct',
                     'llava-1.5-13b-hf', 'idefics-9b-instruct']

    text_model_list = ['vicuna-33b-v1.3', 'gpt-4-0125-preview', 'gpt-4-turbo-2024-04-09', 'claude-3-5-sonnet-20240620', 'gpt-4-1106-preview',
                         'gpt-4-0613', 'gpt-4o-2024-05-13', 'claude-3-opus-20240229', 'gemini-1.5-pro-latest',
                         'Meta-Llama-3-70B-Instruct-hf', 'claude-2.1', 'gemini-1.5-flash-latest', 'claude-3-sonnet-20240229',
                         'Qwen1.5-72B-Chat', 'mistral-large-2402', 'gpt-3.5-turbo-0125', 'gemini-1.0-pro', 'command-r-plus', 'openchat_3.5',
                         'claude-3-haiku-20240307', 'sheep-duck-llama-2-70b-v1.1', 'Meta-Llama-3-8B-Instruct-hf', 'openchat-3.5-1210',
                         'WizardLM-70b-v1.0', 'openchat-3.5-0106', 'Qwen1.5-14B-Chat', 'mistral-medium-2312', 'Qwen1.5-32B-Chat',
                         'codegemma-7b-it', 'dolphin-2.5-mixtral-8x7b', 'CodeLlama-34b-Instruct-hf', 'command-r', 'gemma-1.1-7b-it',
                         'SUS-Chat-34B', 'Mixtral-8x22B-Instruct-v0.1', 'tulu-2-dpo-70b', 'Nous-Hermes-2-Mixtral-8x7B-SFT',
                         'WizardLM-13b-v1.2', 'Mistral-7B-Instruct-v0.2', 'Yi-34B-Chat', 'Mixtral-8x7B-Instruct-v0.1',
                         'Mistral-7B-Instruct-v0.1', 'Yi-1.5-34B-Chat', 'vicuna-13b-v1.5', 'Yi-1.5-6B-Chat', 'Starling-LM-7B-beta',
                         'sheep-duck-llama-2-13b', 'Yi-1.5-9B-Chat', 'gemma-1.1-2b-it', 'Qwen1.5-7B-Chat', 'gemma-7b-it',
                         'llama-2-70b-chat-hf', 'Qwen1.5-0.5B-Chat', 'Qwen1.5-1.8B-Chat']

    om, cm = split_models(mm_model_list)
    print("Open")
    print(om)
    print("Closed")
    print(cm)
