import gradio as gr

from src.assets.text_content import TITLE, INTRODUCTION_TEXT, CLEMSCORE_TEXT
from src.leaderboard_utils import filter_search, get_github_data
from src.plot_utils import split_models, compare_plots

# For Leaderboards
dataframe_height = 800 # Height of the table in pixels
# Get CSV data
global primary_leaderboard_df, version_dfs, version_names
primary_leaderboard_df, version_dfs, version_names, date = get_github_data()

global prev_df
prev_df = version_dfs[0]
def select_prev_df(name):
    ind = version_names.index(name)
    prev_df = version_dfs[ind]
    return prev_df

# For Plots 
global plot_df, OPEN_MODELS, CLOSED_MODELS
plot_df = primary_leaderboard_df[0]
MODELS = list(plot_df[list(plot_df.columns)[0]].unique())
OPEN_MODELS, CLOSED_MODELS = split_models(MODELS)


# MAIN APPLICATION s
main_app = gr.Blocks()
with main_app:
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
                                    
            leaderboard_table = gr.Dataframe(
                value=primary_leaderboard_df[0],
                elem_id="leaderboard-table",
                interactive=False,
                visible=True,
                height=dataframe_height
            )

            gr.HTML(CLEMSCORE_TEXT)
            gr.HTML(f"Last updated - {date}")

            # Add a dummy leaderboard to handle search queries from the primary_leaderboard_df and not update primary_leaderboard_df
            dummy_leaderboard_table = gr.Dataframe(
                value=primary_leaderboard_df[0],
                elem_id="leaderboard-table",
                interactive=False,
                visible=False
            )
                
            search_bar.submit(
                filter_search,
                [dummy_leaderboard_table, search_bar],
                leaderboard_table,
                queue=True
            )

        with gr.TabItem("📈 Plot", id=3):
            with gr.Row():
                open_models_selection = gr.CheckboxGroup(
                    OPEN_MODELS, 
                    label="Open-weight Models 🌐",
                    value=[],
                    elem_id="value-select",
                    interactive=True,
                )

            with gr.Row():
                closed_models_selection = gr.CheckboxGroup(
                    CLOSED_MODELS, 
                    label="Closed-weight Models 💼",
                    value=[],
                    elem_id="value-select-2",
                    interactive=True,
                )
            
            with gr.Row():
                with gr.Column():
                    show_all = gr.CheckboxGroup(
                        ["Select All Models"],
                        label="Show plot for all models 🤖",
                        value=[],
                        elem_id="value-select-3",
                        interactive=True,
                    )
                
                with gr.Column():
                    show_names = gr.CheckboxGroup(
                        ["Show Names"],
                        label ="Show names of models on the plot 🏷️",
                        value=[],
                        elem_id="value-select-4",
                        interactive=True,
                    ) 

                with gr.Column():
                    show_legend = gr.CheckboxGroup(
                        ["Show Legend"],
                        label ="Show legend on the plot 💡",
                        value=[],
                        elem_id="value-select-5",
                        interactive=True,
                    ) 
                with gr.Column():
                    mobile_view = gr.CheckboxGroup(
                        ["Mobile View"],
                        label ="View plot on smaller screens 📱",
                        value=[],
                        elem_id="value-select-6",
                        interactive=True,
                    ) 

            with gr.Row():
                dummy_plot_df = gr.DataFrame(
                    value=plot_df,
                    visible=False
                )

            with gr.Row():
                with gr.Column():
                    # Output block for the plot
                    plot_output = gr.Plot()

            open_models_selection.change(
                compare_plots,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend, mobile_view],
                plot_output,
                queue=True
            )

            closed_models_selection.change(
                compare_plots,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend, mobile_view],
                plot_output,
                queue=True
            )
            
            show_all.change(
                compare_plots,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend, mobile_view],
                plot_output,
                queue=True
            )

            show_names.change(
                compare_plots,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend, mobile_view],
                plot_output,
                queue=True
            )

            show_legend.change(
                compare_plots,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend, mobile_view],
                plot_output,
                queue=True
            )

            mobile_view.change(
                compare_plots,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend, mobile_view],
                plot_output,
                queue=True
            )

        with gr.TabItem("🔄 Versions and Details", elem_id="details", id=2):
            with gr.Row():
                version_select = gr.Dropdown(
                    version_names, label="Select Version 🕹️", value=version_names[0]
                )
            with gr.Row():
                search_bar_prev = gr.Textbox(
                    placeholder=" 🔍 Search for models - separate multiple queries with `;` and press ENTER...",
                    show_label=False,
                    elem_id="search-bar-2",
                )

            prev_table = gr.Dataframe(
                value=prev_df,
                elem_id="leaderboard-table",
                interactive=False,
                visible=True,
                height=dataframe_height
            )

            dummy_prev_table = gr.Dataframe(
                value=prev_df,
                elem_id="leaderboard-table",
                interactive=False,
                visible=False
            )

            search_bar_prev.submit(
                filter_search,
                [dummy_prev_table, search_bar_prev],
                prev_table,
                queue=True
            )

            version_select.change(
                select_prev_df,
                [version_select],
                prev_table,
                queue=True
            )
    main_app.load()

main_app.queue()

main_app.launch()