import copy

import pandas as pd
from matplotlib import pyplot as plt

from visualization.concept_drift_plot import plot_concept_drift_line
from visualization.plot_feature_importance import plot_double_stacked_line_plots

if __name__ == "__main__":
    data_dir = "../experiment_results/"

    mdi_right_df = pd.read_csv(data_dir+"mdi_df_right.csv")
    pfi_right_df = pd.read_csv(data_dir + "pfi_df_right.csv")
    performance_right_df = pd.read_csv(data_dir + "perf_right.csv")

    mdi_left_df = pd.read_csv(data_dir + "mdi_df_left.csv")
    pfi_left_df = pd.read_csv(data_dir + "pfi_df_left.csv")
    performance_left_df = pd.read_csv(data_dir + "perf_left.csv")


    x_data_left = [i for i in range(1, len(mdi_left_df) + 1)]
    x_data_right = [i for i in range(1, len(mdi_right_df) + 1)]

    top_data = {'line_names': ['performance']}
    top_left_data = copy.deepcopy(top_data)
    top_left_data['y_data'] = {'accuracy': performance_left_df}
    top_left_data['x_data'] = x_data_left
    top_right_data = copy.deepcopy(top_data)
    top_right_data['x_data'] = x_data_right
    top_right_data['y_data'] = {'accuracy': performance_right_df}
    bot_left_data = {'y_data': {'iPFI': pfi_left_df, 'mdi': mdi_left_df},
                     'x_data': x_data_left,
                     'line_names': list(pfi_left_df.columns)}
    bot_right_data = {'y_data': {'iPFI': pfi_right_df, 'mdi': mdi_right_df},
                     'x_data': x_data_right,
                     'line_names': list(pfi_right_df.columns)}

    # plot styles --------------------------------------------------------------------------------------------------
    top_styles = {
        "y_min": 0, "y_max": 1,
        "color_list": ["red"],
        "line_styles": {"accuracy": "solid"},
    }
    top_right_styles = copy.copy(top_styles)
    top_left_styles = copy.copy(top_styles)
    top_left_styles["y_label"] = "Accuracy"
    top_left_styles["title"] = "agrawal data stream"
    top_right_styles["title"] = "elec2 data stream"

    bot_styles = {
        "y_min": -0.05, "y_max": 0.55,
        "names_to_highlight": ['nswprice', 'date'],
        "line_styles": {'iPFI': 'solid', 'mdi': 'dotted'},
        }
    bot_left_styles = copy.copy(bot_styles)
    bot_left_styles["legend_lines"] = {'loc': "upper left", 'ncol': 3, 'columnspacing': 0.5, 'labelspacing': 0.25}
    bot_left_styles["names_to_highlight"] = ['salary', 'age', 'elevel', 'commission']
    bot_left_styles["y_label"] = "Feature Importance"
    bot_left_styles["secondary_legends"] = [{
            'legend_props': {'loc': "upper left", "framealpha": 1., "bbox_to_anchor": (0.935, 1), 'columnspacing': 0.5, 'labelspacing': 0.25},
            'legend_items': [('iPFI', '-', 'black'), ('MDI', 'dotted', 'black')]}]
    bot_right_styles = copy.copy(bot_styles)

    bot_right_styles["legend_lines"] = {'loc': "upper right", 'ncol': 3, 'columnspacing': 0.5, 'labelspacing': 0.25}
    bot_right_styles["names_to_highlight"] = ['nswprice', 'date']


    fig, (top_left_axis, top_right_axis, bot_left_axis, bot_right_axis) = plot_double_stacked_line_plots(
        figsize=(14, 5),
        top_left_data=top_left_data,
        top_left_styles=top_left_styles,
        bot_left_data=bot_left_data,
        bot_left_styles=bot_left_styles,
        top_right_data=top_right_data,
        top_right_styles=top_right_styles,
        bot_right_data=bot_right_data,
        bot_right_styles=bot_right_styles,
        top_portion=0.25, title=r"iPFI and MDI comparison - sampling: geometric, $\alpha$: 0.001",
        show=False
    )
    fig.supxlabel("Samples")
    plot_concept_drift_line(top_left_axis, 50_000, concept_drift_color=None, line_style='--', show=False)
    plot_concept_drift_line(bot_left_axis, 50_000, concept_drift_color=None, line_style='--', show=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.000, wspace=0.000, bottom=0.1, top=0.9)
    plt.savefig("../experiment_results/mdi_ipfi_comparison.png", dpi=300)
    plt.show()
