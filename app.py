import gradio as gr

from src.assets.text_content import TITLE, INTRODUCTION_TEXT
from src.utils import compare_plots, filter_search, get_github_data, get_current_urls

############################ For Leaderboards #############################
global URLS, VERS, latest_df, all_dfs, all_vnames
username = 'clembench'
repo = 'clembench-runs'

URLS, VERS = get_current_urls(username, repo)

latest_df, all_dfs, all_vnames = get_github_data(URLS, VERS)

global prev_df
prev_df = all_dfs[0]
def select_prev_df(name):
    ind = all_vnames.index(name)
    prev_df = all_dfs[ind]
    return prev_df

############################ For Plots ####################################
global plot_df, MODEL_COLS
plot_df = latest_df[0]
MODEL_COLS = list(plot_df['Model'].unique())


########################## For Background Scheduler #######################


############# MAIN APPLICATION ######################
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
                model_cols = gr.CheckboxGroup(
                    MODEL_COLS, 
                    label="Select Models 🤖", 
                    value=[],
                    elem_id="column-select",
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

            model_cols.change(
                compare_plots,
                [plot_grdf, model_cols],
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

    demo.load()
demo.queue()
demo.launch()