import gradio as gr
import os
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import HfApi
from datetime import datetime, timedelta

from src.assets.text_content import TITLE, INTRODUCTION_TEXT, CLEMSCORE_TEXT, MULTIMODAL_NAME, TEXT_NAME, HF_REPO
from src.leaderboard_utils import query_search, get_github_data
from src.plot_utils import split_models, plotly_plot, get_plot_df, update_open_models, update_closed_models
from src.plot_utils import reset_show_all, reset_show_names, reset_show_legend, reset_mobile_view
from src.version_utils import get_version_data
from src.trend_utils import get_final_trend_plot

""" 
CONSTANTS
"""
# For restarting the gradio application every 24 Hrs
TIME = 43200  # in seconds # Reload will not work locally - requires HFToken # The app launches locally as expected - only without the reload utility


"""
AUTO RESTART HF SPACE
"""
HF_TOKEN = os.environ.get("H4_TOKEN", None)
api = HfApi()

def restart_space():
    api.restart_space(repo_id=HF_REPO, token=HF_TOKEN)


"""
GITHUB UTILS
"""
github_data = get_github_data()
text_leaderboard = github_data["text"]["dataframes"][0]  # Get the latest version of text-only leaderboard
multimodal_leaderboard = github_data["multimodal"]["dataframes"][0]  # Get the latest version of multimodal leaderboard

# Show only First 4 columns for the leaderboards
# Should be Model Name, Clemscore, %Played, and Quality Score
text_leaderboard = text_leaderboard.iloc[:, :4]
multimodal_leaderboard = multimodal_leaderboard.iloc[:, :4]


"""
VERSIONS UTILS
"""
versions_data = get_version_data()
latest_version = versions_data['versions'][0]['name']
last_updated_date = versions_data['versions'][0]['last_updated'][0]
version_names = [v['name'] for v in versions_data['versions']]

global version_df
version_df = versions_data['dataframes'][0]
def select_version_df(name):
    for i, v in enumerate(versions_data['versions']):
        if v['name'] == name:
            return versions_data['dataframes'][i]

"""
MAIN APPLICATION
"""
hf_app = gr.Blocks()
with hf_app:

    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        """
        #######################        FIRST TAB - TEXT-LEADERBOARD       #######################
        """
        with gr.TabItem(TEXT_NAME, elem_id="llm-benchmark-tab-table", id=0):
            with gr.Row():
                search_bar = gr.Textbox(
                    placeholder=" 🔍 Search for models - separate multiple queries with `;` and press ENTER...",
                    show_label=False,
                    elem_id="search-bar",
                )

            leaderboard_table = gr.Dataframe(
                value=text_leaderboard,
                elem_id="text-leaderboard-table",
                interactive=False,
                visible=True
            )

            # Show information about the clemscore and last updated date below the table
            gr.HTML(CLEMSCORE_TEXT)
            gr.HTML(f"Last updated - {github_data['text']['version_data'][0]['last_updated'][0]}")

            # Add a dummy leaderboard to handle search queries in leaderboard_table
            # This will show a temporary leaderboard based on the searched value
            dummy_leaderboard_table = gr.Dataframe(
                value=text_leaderboard,
                elem_id="text-leaderboard-table-dummy",
                interactive=False,
                visible=False
            )

            # Action after submitting a query to the search bar
            search_bar.submit(
                query_search,
                [dummy_leaderboard_table, search_bar],
                leaderboard_table,
                queue=True
            )

        """
        #######################       SECOND TAB - MULTIMODAL LEADERBOARD     #######################
        """
        with gr.TabItem(MULTIMODAL_NAME, elem_id="mm-llm-benchmark-tab-table", id=1):
            with gr.Row():
                mm_search_bar = gr.Textbox(
                    placeholder=" 🔍 Search for models - separate multiple queries with `;` and press ENTER...",
                    show_label=False,
                    elem_id="search-bar",
                )

            mm_leaderboard_table = gr.Dataframe(
                value=multimodal_leaderboard,
                elem_id="mm-leaderboard-table",
                interactive=False,
                visible=True
            )

            # Show information about the clemscore and last updated date below the table
            gr.HTML(CLEMSCORE_TEXT)
            gr.HTML(f"Last updated - {github_data['multimodal']['version_data'][0]['last_updated'][0]}")

            # Add a dummy leaderboard to handle search queries in leaderboard_table
            # This will show a temporary leaderboard based on the searched value
            mm_dummy_leaderboard_table = gr.Dataframe(
                value=multimodal_leaderboard,
                elem_id="mm-leaderboard-table-dummy",
                interactive=False,
                visible=False
            )

            # Action after submitting a query to the search bar
            mm_search_bar.submit(
                query_search,
                [mm_dummy_leaderboard_table, mm_search_bar],
                mm_leaderboard_table,
                queue=True
            )

        """
        #######################       THIRD TAB - PLOTS - %PLAYED V/S QUALITY SCORE     #######################
        """
        with gr.TabItem("📊 Plots", elem_id="plots", id=2):
            """
            DropDown Select for Text/Multimodal Leaderboard
            """
            leaderboard_selection = gr.Dropdown(
                choices=[TEXT_NAME, MULTIMODAL_NAME],
                value=TEXT_NAME,
                label="Select Leaderboard 🎖️🔽",
                elem_id="value-select-0",
                interactive=True
            )

            """
            Accordion Groups to select individual models - Hidden by default
            """
            with gr.Accordion("Select Open-weight Models 🌐", open=False):
                open_models_selection = update_open_models()
                clear_button_1 = gr.ClearButton(open_models_selection)

            with gr.Accordion("Select Commercial Models 💰", open=False):
                closed_models_selection = update_closed_models()
                clear_button_2 = gr.ClearButton(closed_models_selection)

            """
            Checkbox group to control the layout of the plot 
            """
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
                        label="Show names of models on the plot 🏷️",
                        value=[],
                        elem_id="value-select-4",
                        interactive=True,
                    )

                with gr.Column():
                    show_legend = gr.CheckboxGroup(
                        ["Show Legend"],
                        label="Show legend on the plot 💡",
                        value=[],
                        elem_id="value-select-5",
                        interactive=True,
                    )
                with gr.Column():
                    mobile_view = gr.CheckboxGroup(
                        ["Mobile View"],
                        label="View plot on smaller screens 📱",
                        value=[],
                        elem_id="value-select-6",
                        interactive=True,
                    )

            """
            PLOT BLOCK
            """
            # Create a dummy DataFrame as an input to the plotly_plot function.
            # Uses this data to plot the %played v/s quality score
            with gr.Row():
                dummy_plot_df = gr.DataFrame(
                    value=get_plot_df(),
                    visible=False
                )

            with gr.Row():
                with gr.Column():
                    # Output block for the plot
                    plot_output = gr.Plot()

            """
            PLOT CHANGE ACTIONS
            Toggle 'Select All Models' based on the values in Accordion checkbox groups
            """
            open_models_selection.change(
                plotly_plot,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend,
                 mobile_view],
                [plot_output],
                queue=True
            )

            closed_models_selection.change(
                plotly_plot,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend,
                 mobile_view],
                [plot_output],
                queue=True
            )

            show_all.change(
                plotly_plot,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend,
                 mobile_view],
                [plot_output],
                queue=True
            )

            show_names.change(
                plotly_plot,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend,
                 mobile_view],
                [plot_output],
                queue=True
            )

            show_legend.change(
                plotly_plot,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend,
                 mobile_view],
                [plot_output],
                queue=True
            )

            mobile_view.change(
                plotly_plot,
                [dummy_plot_df, open_models_selection, closed_models_selection, show_all, show_names, show_legend,
                 mobile_view],
                [plot_output],
                queue=True
            )
            """
            LEADERBOARD SELECT CHANGE ACTIONS
            Update Checkbox Groups and Dummy DF based on the leaderboard selected
            """
            leaderboard_selection.change(
                update_open_models,
                [leaderboard_selection],
                [open_models_selection],
                queue=True
            )

            leaderboard_selection.change(
                update_closed_models,
                [leaderboard_selection],
                [closed_models_selection],
                queue=True
            )

            leaderboard_selection.change(
                get_plot_df,
                [leaderboard_selection],
                [dummy_plot_df],
                queue=True
            )

            ## Reset Plot when Leaderboard selection changes
            leaderboard_selection.change(
                reset_show_all,
                outputs=[show_all],
                queue=True
            )

            open_models_selection.change(
                reset_show_all,
                outputs=[show_all],
                queue=True
            )

            closed_models_selection.change(
                reset_show_all,
                outputs=[show_all],
                queue=True
            )

            leaderboard_selection.change(
                reset_show_names,
                outputs=[show_names],
                queue=True
            )

            leaderboard_selection.change(
                reset_show_legend,
                outputs=[show_legend],
                queue=True
            )

            leaderboard_selection.change(
                reset_mobile_view,
                outputs=[mobile_view],
                queue=True
            )

        """
        #######################       FOURTH TAB - TRENDS     #######################
        """
        with gr.TabItem("📈Trends", elem_id="trends-tab", id=3):
            with gr.Row():
                mkd_text = gr.Markdown("### Commercial v/s Open-Weight models - clemscore over time.  The size of the circles represents the scaled value of the parameters of the models. Larger circles indicate higher parameter values.")
    
            with gr.Row():
                with gr.Column(scale=3):
                    trend_select = gr.Dropdown(
                        choices=["Text", "Multimodal"],
                        value=None,
                        label="Select Benchmark 🔍",
                        elem_id="value-select-7",
                        interactive=True,
                    )
                with gr.Column(scale=1):
                    mobile_view = gr.CheckboxGroup(
                        choices=["Mobile View"],
                        value=[],
                        label="View plot on smaller screens 📱",
                        elem_id="value-select-8",
                        interactive=True,
                    )

            with gr.Row():
                trend_plot = gr.Plot(show_label=False)

            trend_select.change(
                get_final_trend_plot,
                [trend_select, mobile_view],
                [trend_plot],
                queue=True
            )

            mobile_view.change(
                get_final_trend_plot,
                [trend_select, mobile_view],
                [trend_plot],
                queue=True
            )
 
            
        """
        #######################       FIFTH TAB - VERSIONS AND DETAILS     #######################
        """
        with gr.TabItem("🔄 Versions and Details", elem_id="versions-details-tab", id=4):
            with gr.Row():
                version_select = gr.Dropdown(
                    version_names, label="Select Version 🕹️", value=latest_version
                )
            with gr.Row():
                search_bar_prev = gr.Textbox(
                    placeholder=" 🔍 Search for models - separate multiple queries with `;` and press ENTER...",
                    show_label=False,
                    elem_id="search-bar-3",
                )

            prev_table = gr.Dataframe(
                value=version_df,
                elem_id="version-leaderboard-table",
                interactive=False,
                visible=True
            )

            dummy_prev_table = gr.Dataframe(
                value=version_df,
                elem_id="version-dummy-leaderboard-table",
                interactive=False,
                visible=False
            )

            gr.HTML(CLEMSCORE_TEXT)
            gr.HTML(f"Last updated - {last_updated_date}")

            search_bar_prev.submit(
                query_search,
                [dummy_prev_table, search_bar_prev],
                prev_table,
                queue=True
            )

            version_select.change(
                select_version_df,
                [version_select],
                prev_table,
                queue=True
            )

            # Update Dummy Leaderboard, when changing versions
            version_select.change(
                select_version_df,
                [version_select],
                dummy_prev_table,
                queue=True
            )

    hf_app.load()
hf_app.queue()

# Add scheduler to auto-restart the HF space at every TIME interval and update every component each time
scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, 'interval', seconds=TIME)
scheduler.start()

# Log current start time and scheduled restart time
print(datetime.now())
print(f"Scheduled restart at {datetime.now() + timedelta(seconds=TIME)}")

hf_app.launch()
