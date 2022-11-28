import copy
from fractions import Fraction

import pandas as pd
import matplotlib.pyplot as plt
import os


from visualization.color import color_list_generator, get_color_with_generator, STD_ALPHA, DPI, BACKGROUND_COLOR
from visualization.utils import EXPERIMENT_RESULT_SUMMARY_PATH, EXPERIMENT_DATA_DIR, RESULT_SUMMARY_COLUMNS, \
    EXPERIMENT_RESULT_DIR, BASE_COLOR, COLOR_LIST, OPEN_ML_ADULT_FEATURE_NAMES


def validate_y_x_data(y_data, x_data=None, line_names=None):
    if isinstance(y_data, dict):
        y_keys = set(y_data.keys())
    else:
        y_data = {'data': y_data}
        y_keys = {'data'}

    for y_key in y_keys:
        y_data_facet = y_data[y_key]
        if not isinstance(y_data_facet, pd.Series) and not isinstance(y_data_facet, pd.DataFrame):
            if line_names is None:
                line_names = ['X' + str(i) for i in range(len(y_data_facet))]
            y_data[y_key] = pd.DataFrame(y_data_facet, columns=line_names)

    if x_data is not None and isinstance(x_data, dict):
        x_keys = list(x_data.keys())
    else:
        if x_data is None:
            x_data = {y_key: list(range(len(y_data[y_key]))) for y_key in y_keys}
        else:
            x_data = {y_key: x_data for y_key in y_keys}
        x_keys = list(x_data.keys())

    for y_key in y_keys:
        if y_key not in x_keys:
            x_data[y_key] = x_data[x_keys[0]]

    return copy.deepcopy(y_data), copy.deepcopy(x_data), copy.deepcopy(line_names)


def validate_std(y_data, std, line_names):
    if std is None:
        return None
    if not isinstance(std, dict):
        std_dict = {}
        if len(y_data) > 1:
            for n, y_key in enumerate(y_data.keys()):
                std_dict[y_key] = pd.DataFrame(std[n], columns=line_names)
        else:
            std_dict[list(y_data.keys())[0]] = pd.DataFrame(std, columns=line_names)
        std = std_dict
    else:
        for std_key in std.keys():
            if not isinstance(std[std_key], pd.Series) and not isinstance(std[std_key], pd.DataFrame):
                std[std_key] = pd.DataFrame(std[std_key], columns=line_names)
    return std


def plot_multi_line_graph(axis, y_data, x_data, line_names=None, names_to_highlight=None, line_styles=None, std=None,
                          title=None, y_label=None, x_label=None,
                          yticks=None, x_ticks=None, y_min=None, y_max=None, show=False, legend_lines=None, legend=None,
                          base_color=None, color_list=None, secondary_legends=None, plot_others=True):
    if names_to_highlight is None:
        names_to_highlight = line_names if line_names is not None else []

    y_data, x_data, line_names = validate_y_x_data(y_data, x_data, line_names)
    std = validate_std(y_data, std, line_names)

    color_gens = {}
    for y_data_key in y_data.keys():
        color_gens[y_data_key] = color_list_generator(color_list=color_list)

    if line_styles is None:
        line_styles = {facet: '-' for facet in y_data.keys()}

    line_colors = {}

    for data_name, data_y in y_data.items():
        data_x = x_data[data_name]
        data_std = None
        if std is not None:
            if data_name in std:
                data_std = std[data_name]
        for line_name in line_names:
            color_line = get_color_with_generator(color_generator=color_gens[data_name],
                                                  base_color=base_color,
                                                  item=line_name,
                                                  item_selection=names_to_highlight)
            line_colors[line_name] = color_line
            alpha = 1. if line_name in names_to_highlight else 0.6
            axis.plot(data_x, data_y[line_name],
                      ls=line_styles[data_name],
                      c=color_line,
                      alpha=alpha,
                      linewidth=1)
            if data_std is not None:
                axis.fill_between(data_x,
                                  data_y[line_name] - data_std[line_name],
                                  data_y[line_name] + data_std[line_name],
                                  color=color_line,
                                  alpha=STD_ALPHA,
                                  linewidth=0.)

    if legend_lines is not None:
        for line_name in names_to_highlight:
            axis.plot([], label=line_name, color=line_colors[line_name])
        if plot_others:
            axis.plot([], label='others', color=BASE_COLOR)
        axis.legend(edgecolor="0.8", fancybox=False, **legend_lines)

    if legend is not None:
        for legend_item in legend['legend_items']:
            axis.plot([], label=legend_item[0], color=legend_item[2], ls=legend_item[1])
            axis.legend(edgecolor="0.8", fancybox=False, **legend['legend_props'])

    if secondary_legends is not None:
        for seconday_legend in secondary_legends:
            ax2 = axis.twinx()
            legend_lines = []
            legend_labels = []
            for legend_item in seconday_legend['legend_items']:
                legend_labels.append(legend_item[0])
                legend_line, = ax2.plot([], label=legend_item[0], color=legend_item[2], ls=legend_item[1])
                legend_lines.append(legend_line)
            ax2.get_yaxis().set_visible(False)
            ax2.legend(legend_lines, legend_labels, edgecolor="0.8", fancybox=False, **seconday_legend['legend_props'])


    if y_label is not None:
        axis.set_ylabel(y_label)

    if x_label is not None:
        axis.set_xlabel(x_label)

    if title is not None:
        axis.set_title(title)

    if y_min is not None:
        axis.set_ylim(bottom=y_min)

    if y_max is not None:
        axis.set_ylim(top=y_max)

    if yticks is not None:
        axis.set_yticks(yticks)

    if x_ticks is not None:
        axis.set_xticks(x_ticks)

    axis.set_facecolor(BACKGROUND_COLOR)

    # axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if show:
        plt.show()
        return None
    else:
        return axis


def plot_stacked_line_plots(top_data, bot_data, top_styles, bot_styles,
                            figsize=(10, 5), top_portion=0.5, title=None, show=False):
    ratio = Fraction(top_portion).limit_denominator()
    top_ratio, bot_ratio = (ratio.numerator, ratio.denominator)
    top_styles = copy.deepcopy(top_styles)
    bot_styles = copy.deepcopy(bot_styles)
    fig, (top_axis, bot_axis) = plt.subplots(nrows=2, ncols=1, sharex='all',
                                             figsize=figsize,
                                             gridspec_kw={'height_ratios': [top_ratio, bot_ratio]})
    plt.subplots_adjust(hspace=0.000)
    bot_axis = plot_multi_line_graph(axis=bot_axis,
                                     y_data=bot_data['y_data'],
                                     x_data=bot_data['x_data'],
                                     line_names=bot_data['line_names'],
                                     show=False,
                                     **bot_styles)
    top_axis = plot_multi_line_graph(axis=top_axis,
                                     y_data=top_data['y_data'],
                                     x_data=top_data['x_data'],
                                     line_names=top_data['line_names'],
                                     show=False,
                                     **top_styles)
    if title is not None:
        fig.suptitle(title)

    if show:
        plt.show()
        return None
    else:
        return fig, (top_axis, bot_axis)


def plot_double_stacked_line_plots(top_left_data, top_right_data, bot_left_data, bot_right_data,
                                   top_left_styles, top_right_styles, bot_left_styles, bot_right_styles,
                                   figsize=(10, 5), top_portion=1., left_proportion=1., title=None, show=False):
    ratio_vertical = Fraction(top_portion).limit_denominator()
    top_ratio, bot_ratio = (ratio_vertical.numerator, ratio_vertical.denominator)
    ratio_horizontal = Fraction(left_proportion).limit_denominator()
    left_ratio, right_ratio = (ratio_horizontal.numerator, ratio_horizontal.denominator)
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', dpi=DPI,
                             figsize=figsize,
                             gridspec_kw={'height_ratios': [top_ratio, bot_ratio],
                                          'width_ratios': [left_ratio, right_ratio]})
    plt.subplots_adjust(hspace=0.000, wspace=0.000)
    (top_left_axis, top_right_axis), (bot_left_axis, bot_right_axis) = axes
    top_left_axis = plot_multi_line_graph(axis=top_left_axis,
                                          y_data=top_left_data['y_data'],
                                          x_data=top_left_data['x_data'],
                                          line_names=top_left_data['line_names'],
                                          show=False,
                                          **top_left_styles)
    top_right_axis = plot_multi_line_graph(axis=top_right_axis,
                                           y_data=top_right_data['y_data'],
                                           x_data=top_right_data['x_data'],
                                           line_names=top_right_data['line_names'],
                                           show=False,
                                           **top_right_styles)
    bot_left_axis = plot_multi_line_graph(axis=bot_left_axis,
                                          y_data=bot_left_data['y_data'],
                                          x_data=bot_left_data['x_data'],
                                          line_names=bot_left_data['line_names'],
                                          show=False,
                                          **bot_left_styles)
    bot_right_axis = plot_multi_line_graph(axis=bot_right_axis,
                                           y_data=bot_right_data['y_data'],
                                           x_data=bot_right_data['x_data'],
                                           line_names=bot_right_data['line_names'],
                                           show=False,
                                           **bot_right_styles)
    if title is not None:
        fig.suptitle(title)
    if show:
        plt.show()
        return None
    else:
        return fig, (top_left_axis, top_right_axis, bot_left_axis, bot_right_axis)


if __name__ == "__main__":
    filepath = os.path.join(EXPERIMENT_DATA_DIR, 'adult_GBT_st_1_1_1_1659453965.8354123_EMA_uniform.csv')
    data = pd.read_csv(filepath)
    data = data.values
    x_data = list(range(1, len(data) + 1))
    fig, test = plt.subplots(nrows=1, ncols=1)
    test = plot_multi_line_graph(axis=test, y_data=data, x_data=x_data, line_names=OPEN_ML_ADULT_FEATURE_NAMES,
                                 names_to_highlight=[], show=False)
    plt.show()
