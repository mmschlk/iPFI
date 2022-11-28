import os

import matplotlib.pyplot as plt
import pandas as pd

from visualization.color import BACKGROUND_COLOR
from visualization.color import SAVE_DPI


def get_parameter_study_data(parameter_runs_dir='../experiment_data_ps'):
    metadata = {'run_file': [], 'alpha': [], 'reservoir_length': [], 'explainer': [], 'explanation_stream_length': []}
    runs = set(filter(lambda file: '.txt' in file, os.listdir(parameter_runs_dir)))
    for run_name in runs:
        with open(os.path.join(parameter_runs_dir, run_name), 'r') as run_file:
            explainers = run_file.readlines()
            for explainer in explainers:
                explainer_metadata = explainer.split(' ')
                metadata['run_file'].append(explainer_metadata[0] + '.csv')
                metadata['explanation_stream_length'].append(int(explainer_metadata[3]))
                metadata['alpha'].append(float(explainer_metadata[4]))
                metadata['reservoir_length'].append(int(explainer_metadata[5]))
                split_id = explainer_metadata[0].split('_')
                if split_id[-1] == 'dynamic':
                    explainer_name = '_'.join((split_id[-3], split_id[-2]))
                else:
                    explainer_name = '_'.join((split_id[-2], split_id[-1]))
                metadata['explainer'].append(explainer_name)
    data = {run_file: pd.read_csv(os.path.join(parameter_runs_dir, run_file)) for run_file in metadata['run_file']}
    metadata = pd.DataFrame(metadata)
    return metadata, data


def draw_parameter_analysis(ps_kind, ordering, feature_to_plot,
                            explainer='EMA_geometric', parameter_runs_dir='../experiment_data_ps'):
    metadata, data_all = get_parameter_study_data(parameter_runs_dir=parameter_runs_dir)
    metadata = metadata[metadata['explainer'] == explainer]
    data = {run_file: run_data[feature_to_plot] for run_file, run_data in data_all.items()}
    if ps_kind == 'alpha':
        selection_data = metadata[metadata['reservoir_length'] == 1000]
    else:
        selection_data = metadata[metadata['alpha'] == 0.001]
    plot_ps_runs(selection_data, data, ordering, save_name=ps_kind)


def plot_ps_runs(meta_data, data, ordering, save_name='ps', save_path='../experiment_results'):
    fig, ax = plt.subplots(figsize=(5, 2.5), nrows=1, ncols=1)
    for run in ordering:
        ps_kind, value, color, legend_name = run
        run_file_name = list(meta_data[meta_data[ps_kind] == value]['run_file'])[0]
        y_run = data[run_file_name]
        x_run = list(range(len(y_run)))
        ax.plot(x_run, y_run, c=color, label=legend_name)
    ax.set_facecolor(BACKGROUND_COLOR)
    save_path = os.path.join(save_path, save_name + '.png')
    ax.legend(loc=8, edgecolor="0.8", fancybox=False, ncol=2, columnspacing=0.5, labelspacing=0.25)
    ax.axhline(y=0, color=(0.5, 0.5, 0.5, 0.5), ls='--', linewidth=1.)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Permutation Feature Importance")
    if save_name == 'alpha':
        ax.set_title(r"Sensitivity to $\alpha$")
    else:
        ax.set_title(r"Sensitivity to reservoir length")
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    plt.tight_layout()
    plt.savefig(fname=save_path, dpi=SAVE_DPI)
    plt.show()


if __name__ == "__main__":
    color_list = ['#bebdc0', '#00883b', '#FE6100', '#615d69']
    ps_kind = 'alpha'
    ordering = [(ps_kind, 0.2, color_list[0], r"$\alpha$ = 0.2"),
                (ps_kind, 0.01, color_list[1], r"$\alpha$ = 0.01"),
                (ps_kind, 0.001, color_list[2], r"$\alpha$ = 0.001"),
                (ps_kind, 0.0005, color_list[3], r"$\alpha$ = 0.0005")]
    draw_parameter_analysis(ps_kind, ordering, feature_to_plot='nswprice')
    ps_kind = 'reservoir_length'
    ordering = [(ps_kind, 50, color_list[0], r"res. = 50"), (ps_kind, 100, color_list[1], r"res. = 100"),
                (ps_kind, 1000, color_list[2], r"res. = 1000"), (ps_kind, 2000, color_list[3], r"res. = 2000")]
    draw_parameter_analysis(ps_kind, ordering, feature_to_plot='nswprice')