import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from visualization.color import BACKGROUND_COLOR
from visualization.utils import rename_explainers, get_final_pfis, FEATURE_NAMES, EXPERIMENT_RESULT_SUMMARY_PATH, \
    EXPERIMENT_RESULT_DIR, COLOR_LIST, SAVE_DPI


def get_grouped_boxplot(dataset, model_name, explainer_to_show, legend, model_seed=1, boxplot_width=0.5,
                        min_space=0.1, feature_spacing=0.5, y_min=-1., y_max=1., legend_pos=0):
    feature_columns = FEATURE_NAMES[dataset]

    result_df = pd.read_csv(EXPERIMENT_RESULT_SUMMARY_PATH)
    result_df = result_df[
        (result_df.dataset == dataset) & (result_df.model_name == model_name) &
        (result_df.model_seed == model_seed) & (result_df.dynamic == False)
        ].copy()
    result_df.reset_index(inplace=True)
    result_df = rename_explainers(result_df)

    pfi_df = get_final_pfis(result_df.final_pfi, dataset)
    result_df = pd.concat((result_df, pfi_df), axis=1)

    means = result_df.groupby(['datastream_seed', 'explainer'])[feature_columns].mean().reset_index()

    iterations_dict = {}
    for feature in feature_columns:
        feature_dict = {}
        for explainer in explainer_to_show:
            feature_dict[explainer] = means[means.explainer == explainer][feature].values
        iterations_dict[feature] = feature_dict.copy()

    save_path = os.path.join(EXPERIMENT_RESULT_DIR, '_'.join(('multiboxplot',dataset,model_name+'.png')))
    grouped_boxplot(iterations_dict, feature_columns, explainer_to_show, legend=legend, save_path=save_path,
                    width=boxplot_width, min_space=min_space, feature_spacing=feature_spacing, y_min=y_min,
                    y_max=y_max, legend_pos=legend_pos)


def grouped_boxplot(data, feature_names, explainer_names, save_path, legend, width=0.5, min_space=0.1, feature_spacing=0.5, y_min=-1., y_max=1., legend_pos=0):
    n_explainer = len(explainer_names)
    n_features = len(feature_names)

    feature_names_short = [name[0:3]+'.' if len(name) > 3 else name[0:3] for name in feature_names]

    group_length = width * n_explainer + min_space * (n_explainer-1)
    x_range = np.array([x * group_length for x in range(n_features)])
    x_range = np.array([x_range[i] + feature_spacing * i for i in range(n_features)])
    x_locations = sorted(
        [feature_loc + explainer * (width + min_space) for explainer in range(n_explainer) for feature_loc in x_range])
    feature_x_locations = x_range + group_length / 2 - width / 2

    fig = plt.figure(0, figsize=[9, 4.8])
    ax = fig.add_subplot(111)
    x_position_index = 0
    for feature in feature_names:
        color_index = 0
        for explainer in explainer_names:
            if explainer == 'batch_total' or explainer == 'batch_interval':
                color = 'black'
            else:
                color = COLOR_LIST[color_index]
                color_index += 1
            ax.boxplot(data[feature][explainer], positions=[x_locations[x_position_index]],
                       widths=width, patch_artist=True,
                       medianprops={'color': 'red', 'linestyle': '-'},
                       boxprops=dict(facecolor=color, color=color)
            )
            x_position_index += 1
    x_position_index = 0
    color_index = 0
    for explainer in explainer_names:
        if explainer == 'batch_total' or explainer == 'batch_interval':
            color = 'black'
        else:
            color = COLOR_LIST[color_index]
            color_index += 1
        legend_name = legend[x_position_index]
        ax.plot([], c=color, label=legend_name)
        x_position_index += 1

    ax.legend(edgecolor="0.8", fancybox=False, loc=legend_pos)
    ax.set_xticks(feature_x_locations, feature_names_short)
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=0, color=(0.5, 0.5, 0.5, 0.3), ls='--')
    ax.set_ylabel('Permutation Feature Importance')
    ax.set_xlabel('Features')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    plt.rcParams['axes.facecolor'] = BACKGROUND_COLOR
    explainer_names = ['batch_total', 'EMA_geometric', 'EMA_uniform']

    legend_names = ['Batch (left)', 'iPFI geometric sampling (middle)', 'iPFI uniform sampling (right)']

    if False:
        get_grouped_boxplot(dataset='bike',
                            model_name='LGBM',
                            explainer_to_show=explainer_names,
                            legend=legend_names,
                            y_min=-5., y_max=140, legend_pos=0)
    if False:
        get_grouped_boxplot(dataset='bank-marketing',
                            model_name='NN',
                            explainer_to_show=explainer_names,
                            legend=legend_names,
                            y_min=-0.005, y_max=0.06, legend_pos=0)

    if False:
        get_grouped_boxplot(dataset='elec2',
                            model_name='LGBM',
                            explainer_to_show=explainer_names,
                            legend=legend_names,
                            y_min=-0.05, y_max=0.4, legend_pos=0)

    if False:
        get_grouped_boxplot(dataset='adult',
                            model_name='GBT',
                            explainer_to_show=explainer_names,
                            legend=legend_names,
                            y_min=-0.005, y_max=0.065, legend_pos=0)

    if False:
        get_grouped_boxplot(dataset='agrawal',
                            model_name='LGBM',
                            explainer_to_show=explainer_names,
                            legend=legend_names,
                            y_min=-0.05, y_max=0.4, legend_pos=0)
