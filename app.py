import gradio as gr
from apscheduler.schedulers.background import BackgroundScheduler

from src.assets.text_content import TITLE, INTRODUCTION_TEXT
from src.utils import compare_plots, filter_search, get_github_data, get_csv_data, split_models

############################ For Leaderboards #############################
username = 'kushal-10'
repo = 'clembench-runs-new'

# Save github data locally 
get_github_data(username, repo)

# Get CSV data
global latest_df, all_dfs, all_vnames
latest_df, all_dfs, all_vnames = get_csv_data()

global prev_df
prev_df = all_dfs[0]
def select_prev_df(name):
    ind = all_vnames.index(name)
    prev_df = all_dfs[ind]
    return prev_df

############################ For Plots ####################################
global plot_df, MODEL_COLS, OPEN_MODELS, COMM_MODELS
plot_df = latest_df[0]
MODEL_COLS = list(plot_df['Model'].unique())
OPEN_MODELS, COMM_MODELS = split_models(MODEL_COLS)

########################## For Background Scheduler #######################
# scheduler = BackgroundScheduler()
# scheduler.start()

def update_data():
    global latest_df, all_dfs, all_vnames, ver_selection

    # Always update GitHub data and CSVs
    get_github_data(username, repo)
    latest_df, all_dfs, all_vnames = get_csv_data()

    # Update dropdown options if there's a change in available versions
    new_versions = set(all_vnames) - set(ver_selection.choices)
    if new_versions:
        # Recreate the dropdown with updated choices
        new_ver_selection = gr.Dropdown(
            choices=list(ver_selection.choices) + list(new_versions),
            label="Select Version 🕹️",
            value=ver_selection.value
        )
        # Replace the old dropdown with the new one
        tabs.children[2].children[0] = new_ver_selection

        # Update the global reference to the new dropdown
        ver_selection = new_ver_selection


    # Update leaderboard tables
    leaderboard_table.value = latest_df[0]
    dummy_leaderboard_table.value = latest_df[0]

    # Update plot data
    global plot_df
    plot_df = latest_df[0]
    plot_grdf.value = plot_df

    # Update previous table data
    prev_df = select_prev_df(ver_selection.value)
    prev_table.value = prev_df
    dummy_prev_table.value = prev_df

def restart_space():
    
# ############# MAIN APPLICATION ######################
demo = gr.Blocks()
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("🥇 CLEM Leaderboard", elem_id="llm-benchmark-tab-table", id=0):
            with gr.Row():
                search_bar = gr.Textbox(
                    placeholder=" 🔍 Search for models - separate multiple queries with `;` and press ENTER...",
                    show_label=False,
                    elem_id="search-bar",
                )
                                    
            leaderboard_table = gr.components.Dataframe(
                value=latest_df[0],
                elem_id="leaderboard-table",
                interactive=False,
                visible=True,
            )

            # Add a dummy leaderboard to handle search queries from the latest_df and not update latest_df
            dummy_leaderboard_table = gr.components.Dataframe(
                value=latest_df[0],
                elem_id="leaderboard-table",
                interactive=False,
                visible=False,
            )
                
            search_bar.submit(
                filter_search,
                [dummy_leaderboard_table, search_bar],
                leaderboard_table,
                queue=True
            )

        with gr.TabItem("📈 Plot", id=3):
            with gr.Row():
                open_model_cols = gr.CheckboxGroup(
                    OPEN_MODELS, 
                    label="Select Open Source Models 🌐", 
                    value=[],
                    elem_id="column-select",
                    interactive=True,
                )

            with gr.Row():
                comm_model_cols = gr.CheckboxGroup(
                    COMM_MODELS, 
                    label="Select Commercial Models 💼", 
                    value=[],
                    elem_id="column-select-2",
                    interactive=True,
                )

            with gr.Row():
                plot_grdf = gr.DataFrame(
                    value=plot_df,
                    visible=False
                )
            with gr.Row():
                # Output block for the plot
                plot_output = gr.Plot()

            open_model_cols.change(
                compare_plots,
                [plot_grdf, open_model_cols, comm_model_cols],
                plot_output,
                queue=True
            )

            comm_model_cols.change(
                compare_plots,
                [plot_grdf, open_model_cols, comm_model_cols],
                plot_output,
                queue=True
            )

        with gr.TabItem("🔄 Versions and Details", elem_id="details", id=2):
            with gr.Row():
                ver_selection = gr.Dropdown(
                    all_vnames, label="Select Version 🕹️", value=all_vnames[0]
                )
            with gr.Row():
                search_bar_prev = gr.Textbox(
                    placeholder=" 🔍 Search for models - separate multiple queries with `;` and press ENTER...",
                    show_label=False,
                    elem_id="search-bar-2",
                )

            prev_table = gr.components.Dataframe(
                value=prev_df,
                elem_id="leaderboard-table",
                interactive=False,
                visible=True,
            )

            dummy_prev_table = gr.components.Dataframe(
                value=prev_df,
                elem_id="leaderboard-table",
                interactive=False,
                visible=False,
            )

            search_bar_prev.submit(
                filter_search,
                [dummy_prev_table, search_bar_prev],
                prev_table,
                queue=True
            )

            ver_selection.change(
                select_prev_df,
                [ver_selection],
                prev_table,
                queue=True
            )
    demo.load(update_data, every=5)

demo.queue()

# scheduler.add_job(update_data, 'interval', seconds=10)

demo.launch()