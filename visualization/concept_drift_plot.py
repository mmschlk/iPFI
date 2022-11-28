import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_sets import OPEN_ML_ADULT_FEATURE_NAMES
from .plot_feature_importance import plot_multi_line_graph, plot_double_stacked_line_plots
from visualization.color import BASE_COLOR
from visualization.utils import EXPERIMENT_DATA_DIR, EXPERIMENT_RESULT_SUMMARY_PATH, rename_explainers, FEATURE_NAMES, \
    get_mean_and_std, SAVE_DPI, EXPERIMENT_RESULT_DIR


def plot_concept_drift_line(axis, concept_drift_time, concept_drift_color=None, line_style='-', show=False):
    if concept_drift_color is None:
        concept_drift_color = BASE_COLOR

    axis.axvline(x=concept_drift_time, color=concept_drift_color, ls=line_style, linewidth=1)
    if show:
        plt.show()
        return None
    else:
        return axis

def calculate_error(batch_fi, ipfi_fi):
    batch_fi[batch_fi < 0] = 0
    ipfi_fi[ipfi_fi < 0] = 0
    batch_fi = batch_fi / sum(batch_fi)
    ipfi_fi = ipfi_fi / sum(ipfi_fi)
    error = sum(abs(batch_fi - ipfi_fi))
    return error


def calculate_interval_vs_ipfi(data_batch, data_ipfi):
    timesteps = np.unique(np.where(data_batch[:-1] != data_batch[1:])[0]) + 1
    errors = []
    for timestep in timesteps:
        error = calculate_error(data_batch[timestep], data_ipfi[timestep])
        errors.append(error)
    return errors, timesteps


def print_errors(errors, timesteps, concept_drift_time, concept_drift_id, dataset, model_name, explainer_to_show, alpha):
    before_drift = max(np.where(timesteps <= concept_drift_time)[0])
    error_before = errors[0:before_drift + 1]
    errors_after = errors[before_drift + 1:]
    print(f"{dataset} {model_name} {concept_drift_id} {explainer_to_show} {alpha}"
          f" Whole Stream:"
          f" Median {round(np.quantile(errors, q=0.5), 4)}"
          f" IQR {round(np.quantile(errors, q=0.75) - np.quantile(errors, q=0.25), 4)}"
          f" Quantile(0.25) {round(np.quantile(errors, q=0.25), 4)}"
          f" Quantile(0.75) {round(np.quantile(errors, q=0.75), 4)}"
          f" Before Drift:"
          f" Median {round(np.quantile(error_before, q=0.5), 4)}"
          f" IQR {round(np.quantile(error_before, q=0.75) - np.quantile(error_before, q=0.25), 4)}"
          f" Quantile(0.25) {round(np.quantile(error_before, q=0.25), 4)}"
          f" Quantile(0.75) {round(np.quantile(error_before, q=0.75), 4)}"
          f" After Drift:"
          f" Median {round(np.quantile(errors_after, q=0.5), 4)}"
          f" IQR {round(np.quantile(errors_after, q=0.75) - np.quantile(errors_after, q=0.25), 4)}"
          f" Quantile(0.25) {round(np.quantile(errors_after, q=0.25), 4)}"
          f" Quantile(0.75) {round(np.quantile(errors_after, q=0.75), 4)}")


def plot_concept_drift(dataset_left, model_name_left, concept_drift_id_left,
                       dataset_right, model_name_right, concept_drift_id_right,
                       explainer_to_show_left, explainer_to_show_right,
                       y_min, y_max, yticks, plot_titles,
                       concept_drift_time_left, concept_drift_time_right,
                       alpha_left=0.001, alpha_right=0.001,
                       figsize=(10, 5), x_ticks_left=None, x_ticks_right=None,
                       features_highlight_left=None, features_highlight_right=None, show_batch=True, x_axis_label=None,
                       color_list_left=None, color_list_right=None):
    data = pd.read_csv(EXPERIMENT_RESULT_SUMMARY_PATH)
    data = rename_explainers(data)

    left_data = data[(data['dataset'] == dataset_left) &
                     (data['model_name'] == model_name_left) &
                     (data['concept_drift'] == concept_drift_id_left)
                     & ((data['alpha'] == alpha_left) | (data['alpha'] == 0.0))
                     &(data['datastream_seed'] == 1)]

    right_data = data[(data['dataset'] == dataset_right) &
                      (data['model_name'] == model_name_right) &
                      (data['concept_drift'] == concept_drift_id_right)
                      & ((data['alpha'] == alpha_right) | (data['alpha'] == 0.0))
                      & (data['datastream_seed'] == 1)]

    left_pfi_means, left_pfi_stds, left_acc_means, left_acc_stds = get_mean_and_std(left_data, [explainer_to_show_left])
    right_pfi_means, right_pfi_stds, right_acc_means, right_acc_stds = get_mean_and_std(right_data, [explainer_to_show_right])


    left_x_data = list(range(len(left_pfi_means[list(left_pfi_means.keys())[0]])))
    right_x_data = list(range(len(right_pfi_means[list(right_pfi_means.keys())[0]])))

    if show_batch:
        left_pfi_means_batch, left_pfi_stds_batch, _, _ = get_mean_and_std(left_data, ['batch_interval'])
        left_pfi_means['batch_interval'] = left_pfi_means_batch['batch_interval']
        right_pfi_means_batch, right_pfi_stds_batch, _, _ = get_mean_and_std(right_data, ['batch_interval'])
        right_pfi_means['batch_interval'] = right_pfi_means_batch['batch_interval']

        left_x_data = {explainer_to_show_left: left_x_data, 'batch_interval': left_x_data}
        right_x_data = {explainer_to_show_right: right_x_data, 'batch_interval': right_x_data}

        errors, timesteps = calculate_interval_vs_ipfi(left_pfi_means['batch_interval'],
                                                       left_pfi_means[explainer_to_show_left])
        print_errors(errors, timesteps, concept_drift_time_left, concept_drift_id_left, dataset_left,
                     model_name_left, explainer_to_show_left, alpha_left)
        errors, timesteps = calculate_interval_vs_ipfi(right_pfi_means['batch_interval'],
                                                       right_pfi_means[explainer_to_show_right])
        print_errors(errors, timesteps, concept_drift_time_right, concept_drift_id_right, dataset_right,
                     model_name_right, explainer_to_show_right, alpha_right)
    else:
        left_x_data = {explainer_to_show_left: left_x_data}
        right_x_data = {explainer_to_show_right: right_x_data}

    feature_names_left = FEATURE_NAMES[dataset_left]
    feature_names_right = FEATURE_NAMES[dataset_right]

    fi_left_data = {
        'y_data': left_pfi_means,
        'x_data': left_x_data,
        'line_names': feature_names_left
    }

    ncol = 2 if len(features_highlight_left) > 3 else 1

    fi_left_style = {
        'y_min': y_min,
        'y_max': y_max,
        'yticks': yticks,
        'y_label': 'Permutation Feature Importance',
        'std': left_pfi_stds,
        'x_ticks': x_ticks_left,
        'names_to_highlight': features_highlight_left,
        'line_styles': {explainer_to_show_left: '-', 'batch_interval': '--'},
        'legend_lines': {'loc': 2, 'ncol': ncol, 'columnspacing': 0.5, 'labelspacing': 0.25},
        'color_list': color_list_left
    }

    fi_right_data = {
        'y_data': right_pfi_means,
        'x_data': right_x_data,
        'line_names': feature_names_right,
    }

    fi_right_style = {
        'std': right_pfi_stds,
        'x_ticks': x_ticks_right,
        'names_to_highlight': features_highlight_right,
        'line_styles': {explainer_to_show_right: '-', 'batch_interval': '--'},
        'legend': {
            'legend_props': {'loc': 0, 'columnspacing': 0.5, 'labelspacing': 0.25},
            'legend_items': [('iPFI', '-', 'black'), ('Interval PFI', '--', 'black')]},
        'color_list': color_list_right
    }

    ac_left_data = {
        'y_data': left_acc_means,
        'x_data': {explainer_to_show_left: left_x_data[explainer_to_show_left]},
        'line_names': ['accuracy']
    }

    ac_upper = 0.1 if dataset_left == 'stagger' else 0.
    ac_y_max = ac_upper + 1

    ac_left_style = {
        'y_min': 0.0,
        'y_max': ac_y_max,
        'yticks': [0, 0.25, 0.5, 0.75, 1.],
        'y_label': 'Accuracy',
        'title': plot_titles['left_title'],
        'color_list': ['red']
    }

    ac_right_data = {
        'y_data': right_acc_means,
        'x_data': {explainer_to_show_right: right_x_data[explainer_to_show_right]},
        'line_names': ['accuracy']
    }

    ac_right_style = {
        'y_min': 0.,
        'y_max': ac_y_max,
        'yticks': [0, 0.5, 1.],
        'title': plot_titles['right_title'],
        'color_list': ['red']
    }

    fig, (top_left_axis, top_right_axis, bot_left_axis, bot_right_axis) = plot_double_stacked_line_plots(
        top_left_data=ac_left_data, top_right_data=ac_right_data,
        top_left_styles=ac_left_style, top_right_styles=ac_right_style,
        bot_left_data=fi_left_data, bot_right_data=fi_right_data,
        bot_left_styles=fi_left_style, bot_right_styles=fi_right_style,
        figsize=figsize, top_portion=0.25, left_proportion=1.,
        title=plot_titles['top_title'], show=False)

    plot_concept_drift_line(top_left_axis, concept_drift_time_left, concept_drift_color=None, line_style='--', show=False)
    plot_concept_drift_line(bot_left_axis, concept_drift_time_left, concept_drift_color=None, line_style='--', show=False)
    plot_concept_drift_line(top_right_axis, concept_drift_time_right, concept_drift_color=None, line_style='--', show=False)
    plot_concept_drift_line(bot_right_axis, concept_drift_time_right, concept_drift_color=None, line_style='--', show=False)

    if x_axis_label is not None:
        fig.supxlabel(x_axis_label)

    save_path = '_'.join((dataset_left, model_name_left, explainer_to_show_left, concept_drift_id_left, str(alpha_left),
                          dataset_right, model_name_right, explainer_to_show_right, concept_drift_id_right, str(alpha_right)))
    save_path = os.path.join(EXPERIMENT_RESULT_DIR, '_'.join(('cd_plot', save_path + '.png')))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Experiment B
    if False:
        feature_highlight = ['salary', 'commission', 'age', 'elevel', 'car']
        color_list = ['#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#53687e', '#a2e3c4']
        plot_titles = {'top_title': r'iPFI - data: agrawal, sampling: geometric, $\alpha$: 0.001',
                       'left_title': 'function-drift',
                       'right_title': 'feature-drift'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fn',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fe2',
                           explainer_to_show_left='EMA_geometric', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.001, alpha_right=0.001,
                           concept_drift_time_left=10000, concept_drift_time_right=10000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)
    # Experiment C
    if False:
        feature_highlight = ['nswprice', 'nswdemand']
        color_list = ['#ef27a6', '#7d53de']
        plot_titles = {'top_title': r'iPFI - data: elec2 (feature-drift), $\alpha$: 0.001)',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='elec2', model_name_left='ARF', concept_drift_id_left='fe',
                           dataset_right='elec2', model_name_right='ARF', concept_drift_id_right='fe',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           alpha_left=0.001, alpha_right=0.001,
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           concept_drift_time_left=20000, concept_drift_time_right=20000,
                           x_ticks_left=[0, 10000, 20000, 30000, 40000], x_ticks_right=[0, 10000, 20000, 30000, 40000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)

    ####################################################################################################################
    # Appendix
    ####################################################################################################################

    # Agrawal -------------------------

    # agrawal all: feature_highlight = ['salary', 'commission', 'age', 'elevel', 'car', 'zipcode', 'hvalue', 'hyears', 'loan']
    # color_list = ['#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#53687e', '#a2e3c4', '#d7263d', '#f4e76e', '#f7fe72']

    # Agrawal (fu. 1) <- fn
    if False:
        feature_highlight = ['salary', 'commission', 'age', 'elevel']
        color_list = ['#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#53687e', '#a2e3c4']

        plot_titles = {'top_title': r'iPFI - data: agrawal (fu. 1), $\alpha$: 0.001',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fn',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fn',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.001, alpha_right=0.001,
                           concept_drift_time_left=10000, concept_drift_time_right=10000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)
        plot_titles = {'top_title': r'iPFI - data: agrawal (fu. 1), $\alpha$: 0.01',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fn',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fn',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.01, alpha_right=0.01,
                           concept_drift_time_left=10000, concept_drift_time_right=10000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)

    # Agrawal (fu. 2) <- fn3
    if False:
        feature_highlight = ['salary', 'commission', 'age', 'loan']
        color_list = ['#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#53687e', '#a2e3c4']

        plot_titles = {'top_title': r'iPFI - data: agrawal (fu. 2), $\alpha$: 0.001',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fn3',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fn3',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.001, alpha_right=0.001,
                           concept_drift_time_left=10000, concept_drift_time_right=10000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)
        plot_titles = {'top_title': r'iPFI - data: agrawal (fu. 2), $\alpha$: 0.01',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fn3',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fn3',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.01, alpha_right=0.01,
                           concept_drift_time_left=10000, concept_drift_time_right=10000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)

    # Agrawal (fu. 2, early) <- fn31
    if False:
        feature_highlight = ['salary', 'commission', 'age', 'loan']
        color_list = ['#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#53687e', '#a2e3c4']

        plot_titles = {'top_title': r'iPFI - data: agrawal (fu. 2, early), $\alpha$: 0.001',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fn31',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fn31',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.001, alpha_right=0.001,
                           concept_drift_time_left=5000, concept_drift_time_right=5000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)
        plot_titles = {'top_title': r'iPFI - data: agrawal (fu. 2, early), $\alpha$: 0.01',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fn41',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fn41',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.01, alpha_right=0.01,
                           concept_drift_time_left=5000, concept_drift_time_right=5000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)

    # Agrawal (fu. 2, late) <- fn32
    if False:
        feature_highlight = ['salary', 'commission', 'age', 'loan']
        color_list = ['#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#53687e', '#a2e3c4']

        plot_titles = {'top_title': r'iPFI - data: agrawal (fu. 2, late), $\alpha$: 0.001',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fn32',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fn32',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.001, alpha_right=0.001,
                           concept_drift_time_left=15000, concept_drift_time_right=15000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)
        plot_titles = {'top_title': r'iPFI - data: agrawal (fu. 2, late), $\alpha$: 0.01',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fn42',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fn42',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.01, alpha_right=0.01,
                           concept_drift_time_left=15000, concept_drift_time_right=15000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)

    # Agrawal (fu. 3) <- fn4
    if False:
        feature_highlight = ['elevel', 'salary', 'age', 'commission', 'loan']
        color_list = ['#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#53687e', '#a2e3c4']

        plot_titles = {'top_title': r'iPFI - data: agrawal (fu. 3), $\alpha$: 0.001',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fn4',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fn4',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.001, alpha_right=0.001,
                           concept_drift_time_left=10000, concept_drift_time_right=10000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.35, yticks=[0, 0.1, 0.2, 0.3],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)
        plot_titles = {'top_title': r'iPFI - data: agrawal (fu. 3), $\alpha$: 0.01',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fn4',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fn4',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.01, alpha_right=0.01,
                           concept_drift_time_left=10000, concept_drift_time_right=10000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.35, yticks=[0, 0.1, 0.2, 0.3],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)

    # Agrawal (fe. 1) <- fe2
    if False:
        feature_highlight = ['salary', 'age', 'elevel', 'car']
        color_list = ['#ef27a6', '#4ea5d9', '#7d53de', '#44cfcb', '#53687e', '#a2e3c4']

        plot_titles = {'top_title': r'iPFI - data: agrawal (fe. 1), $\alpha$: 0.001',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fe2',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fe2',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.001, alpha_right=0.001,
                           concept_drift_time_left=10000, concept_drift_time_right=10000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.6, yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)
        plot_titles = {'top_title': r'iPFI - data: agrawal (fe. 1), $\alpha$: 0.01',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='agrawal', model_name_left='ARF', concept_drift_id_left='fe2',
                           dataset_right='agrawal', model_name_right='ARF', concept_drift_id_right='fe2',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.01, alpha_right=0.01,
                           concept_drift_time_left=10000, concept_drift_time_right=10000,
                           x_ticks_left=[0, 5000, 10000, 15000, 20000], x_ticks_right=[0, 5000, 10000, 15000, 20000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.6, yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)

    # Stagger ------------------------
    # stagger all: feature_highlight = ['size', 'color','shape']
    # color_list = ['#ef27a6', '#4ea5d9', '#7d53de']

    # Stagger (fu. 1) <- fn
    if False:
        feature_highlight = ['size', 'color','shape']
        color_list = ['#ef27a6', '#4ea5d9', '#7d53de']

        plot_titles = {'top_title': r'iPFI - data: stagger (fu. 1), $\alpha$: 0.001',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='stagger', model_name_left='ARF', concept_drift_id_left='fn',
                           dataset_right='stagger', model_name_right='ARF', concept_drift_id_right='fn',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.001, alpha_right=0.001,
                           concept_drift_time_left=5000, concept_drift_time_right=5000,
                           x_ticks_left=[0, 2500, 5000, 7500, 10000], x_ticks_right=[0, 2500, 5000, 7500, 10000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)

    # Elec 2 ------------------------
    # elec2 all: feature_highlight = ['date','day','period','nswprice','nswdemand','vicprice','vicdemand','transfer']
    # color_list = ['#ef27a6', '#7d53de']

    # Elec2 (fe. 1) <- fe
    if False:
        feature_highlight = ['nswprice', 'nswdemand']
        color_list = ['#ef27a6', '#7d53de']

        plot_titles = {'top_title': r'iPFI - data: elec2 (fe. 1), $\alpha$: 0.001',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='elec2', model_name_left='ARF', concept_drift_id_left='fe',
                           dataset_right='elec2', model_name_right='ARF', concept_drift_id_right='fe',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.001, alpha_right=0.001,
                           concept_drift_time_left=20000, concept_drift_time_right=20000,
                           x_ticks_left=[0, 10000, 20000, 30000, 40000], x_ticks_right=[0, 10000, 20000, 30000, 40000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)

    # Elec2 (fe. 1, gradual) <- fe3
    if False:
        feature_highlight = ['nswprice', 'nswdemand']
        color_list = ['#ef27a6', '#7d53de']

        plot_titles = {'top_title': r'iPFI - data: elec2 (fe. 1, gradual), $\alpha$: 0.001',
                       'left_title': 'uniform sampling',
                       'right_title': 'geometric sampling'}
        plot_concept_drift(dataset_left='elec2', model_name_left='ARF', concept_drift_id_left='fe3',
                           dataset_right='elec2', model_name_right='ARF', concept_drift_id_right='fe3',
                           explainer_to_show_left='EMA_uniform', explainer_to_show_right='EMA_geometric',
                           features_highlight_left=feature_highlight,
                           features_highlight_right=feature_highlight,
                           alpha_left=0.001, alpha_right=0.001,
                           concept_drift_time_left=20000, concept_drift_time_right=20000,
                           x_ticks_left=[0, 10000, 20000, 30000, 40000], x_ticks_right=[0, 10000, 20000, 30000, 40000],
                           x_axis_label='Samples',
                           y_min=-0.02, y_max=0.5, yticks=[0, 0.1, 0.2, 0.3, 0.4],
                           plot_titles=plot_titles,
                           figsize=(14, 4),
                           color_list_left=color_list, color_list_right=color_list,
                           show_batch=True)