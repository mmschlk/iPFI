import copy

import pandas as pd
from matplotlib import pyplot as plt

from visualization.concept_drift_plot import plot_concept_drift_line
from visualization.plot_feature_importance import plot_double_stacked_line_plots

if __name__ == "__main__":
    data_dir = "../experiment_results/"

    pfi_means_left = pd.read_csv(data_dir+"elec2_NN_pfi_means.csv")
    pfi_std_left = pd.read_csv(data_dir + "elec2_NN_pfi_std.csv")
    pfi_interval_left = pd.read_csv(data_dir + "elec2_NN_pfi_interval.csv")
    performance_left_df = pd.read_csv(data_dir + "elec2_NN_performance.csv")

    pfi_means_right = pd.read_csv(data_dir + "elec2_pfi_means_right.csv")
    pfi_std_right = pd.read_csv(data_dir + "elec2_pfi_std_right.csv")
    pfi_interval_right = pd.read_csv(data_dir + "elec2_pfi_interval_right.csv")
    performance_right_df = pd.read_csv(data_dir + "elec2_performance_right.csv")

    x_data_left = [i for i in range(1, len(pfi_means_left) + 1)]
    x_data_right = copy.deepcopy(x_data_left)

    top_data = {'line_names': ['performance']}
    top_left_data = copy.deepcopy(top_data)
    top_left_data['y_data'] = {'accuracy': performance_left_df}
    top_left_data['x_data'] = x_data_left
    top_right_data = copy.deepcopy(top_data)
    top_right_data['x_data'] = x_data_right
    top_right_data['y_data'] = {'accuracy': performance_right_df}
    bot_left_data = {'y_data': {'iPFI': pfi_means_left, 'int': pfi_interval_left},
                     'x_data': x_data_left,
                     'line_names': list(pfi_means_left.columns)}
    bot_right_data = {'y_data': {'iPFI': pfi_means_right, 'int': pfi_interval_right},
                     'x_data': x_data_right,
                     'line_names': list(pfi_means_right.columns)}

    # plot styles --------------------------------------------------------------------------------------------------
    top_styles = {
        "y_min": 0, "y_max": 1,
        "color_list": ["red"],
        "line_styles": {"accuracy": "solid"},
    }
    top_right_styles = copy.copy(top_styles)
    top_left_styles = copy.copy(top_styles)
    top_left_styles["y_label"] = "Accuracy"
    top_left_styles["title"] = "Neural Network classifier"
    top_right_styles["title"] = "ARF classifier"

    bot_styles = {
        "y_min": -0.05, "y_max": 0.53,
        "line_styles": {'iPFI': 'solid', 'int': 'dashed'},
        }
    bot_left_styles = copy.copy(bot_styles)
    bot_left_styles["legend_lines"] = {'loc': "upper left", 'ncol': 4, 'columnspacing': 0.5, 'labelspacing': 0.25}
    bot_left_styles["y_label"] = "Permutation Feature Importance"
    bot_left_styles["names_to_highlight"] = ['nswprice', 'date', 'vicprice']#, 'nswdemand', 'day', 'period','vicdemand','transfer']
    bot_left_styles["std"] = {'iPFI': pfi_std_left}
    bot_left_styles["secondary_legends"] = [{
            'legend_props': {'loc': "upper left", "framealpha": 1., "bbox_to_anchor": (0.89, 1), 'columnspacing': 0.5, 'labelspacing': 0.25},
            'legend_items': [('iPFI', '-', 'black'), ('Interval PFI', 'dashed', 'black')]}]
    bot_right_styles = copy.copy(bot_styles)

    bot_right_styles["legend_lines"] = {'loc': "upper right", 'ncol': 4, 'columnspacing': 0.5, 'labelspacing': 0.25}
    bot_right_styles["names_to_highlight"] = ['nswprice', 'date', 'vicprice']
    bot_right_styles["std"] = {'iPFI': pfi_std_right}


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
        top_portion=0.25, title=r"iPFI - data: elec2, sampling: geometric, $\alpha$: 0.001",
        show=False
    )
    fig.supxlabel("Samples")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.000, wspace=0.000, bottom=0.1, top=0.88)
    plt.savefig("../experiment_results/elec2_exp_b.png", dpi=400)
    plt.show()
