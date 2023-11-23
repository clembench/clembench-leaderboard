import gradio as gr

from src.assets.text_content import TITLE, INTRODUCTION_TEXT
from src.utils import get_data, compare_plots, filter_search

############################ For Leaderboards #############################
DATA_PATH = 'versions'
latest_flag = True #Set flag to iclude latest data in Details and Versions Tab
latest_df, latest_vname, previous_df, previous_vname = get_data(DATA_PATH, latest_flag)

global prev_df
prev_df = previous_df[0]
def select_prev_df(name):
    ind = previous_vname.index(name)
    prev_df = previous_df[ind]
    return prev_df

############################ For Plots ####################################
global plot_df, MODEL_COLS
plot_df = latest_df[0]
MODEL_COLS = list(plot_df['Model'].unique())


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
                    previous_vname, label="Select Version 🕹️", value=previous_vname[0]
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