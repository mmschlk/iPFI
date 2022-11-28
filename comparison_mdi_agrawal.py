import copy
import matplotlib.pyplot as plt
import pandas as pd
import river.metrics
from river.datasets.synth import Agrawal
from river.tree import HoeffdingAdaptiveTreeClassifier

from explainer import IncrementalPFI
import data_sets

# iPFI Setup
from mdi import MeanDecreaseImpurityExplainer
from visualization.concept_drift_plot import plot_concept_drift_line
from visualization.plot_feature_importance import plot_double_stacked_line_plots

alpha = 0.001  # alpha for exponential smoothing
reservoir_length = 100  # length of the incremental reservoir (k)


if __name__ == "__main__":

    for i in range(1, 2):

        stream_1 = Agrawal(classification_function=1)
        stream_2 = Agrawal(classification_function=2)
        feature_names = data_sets.AGRAWAL_FEATURE_NAMES
        stream_length = 100_000
        stream_1_length = stream_length / 2
        stream_2_length = stream_length / 2

        # model to be used
        model = HoeffdingAdaptiveTreeClassifier(leaf_prediction='mc', seed=1)

        # get the explainers
        explainer = IncrementalPFI(
            model=model,
            classification=True,
            feature_names=feature_names,
            mode_aggr="EMA",
            alpha=alpha,  # alpha for exponential smoothing
            reservoir_length=reservoir_length,  # length of the reservoir
            reservoir_mode='original',
            constant_probability=1,
            samplewise_reservoir=True,
            remove_used_reservoir_sample=False,
            sub_sample_length=5
            )

        mdi_explainer = MeanDecreaseImpurityExplainer(feature_names=feature_names, tree_classifier=model)

        performance_metric = river.metrics.Accuracy()
        metric = river.metrics.Rolling(metric=performance_metric, window_size=200)

        performance = []
        pfis = []
        mdis = []
        accuracy = []
        splits = []

        for (n, (x_i, y_i)) in enumerate(stream_1):
            n += 1
            # prequential evaluation
            y_i_pred_test = model.predict_one(x_i)
            # learning
            model.learn_one(x_i, y_i)
            # update metric
            metric.update(y_true=y_i, y_pred=y_i_pred_test)
            accuracy.append({"performance": metric.get()})
            # explaining
            y_i_pred = model.predict_one(x_i)
            pfi_i = explainer.explain_one(x_orig=x_i, y_true_orig=y_i, y_pred_orig=y_i_pred)
            pfis.append(copy.deepcopy(pfi_i))
            mdi_i, splits_i = mdi_explainer.explain_one()
            mdi_i = list(mdi_i.values())
            splits.append(copy.deepcopy(splits_i))
            mdis.append(mdi_i)
            if n % 1000 == 0:
                print(f"{n} data         {x_i}\n"
                      f"{n} Performance: {metric.get()}\n"
                      f"{n} iPFI:        {pfi_i}\n"
                      f"{n} mdi          {mdi_i}\n")
            if n >= stream_1_length:
                break

        for (n, (x_i, y_i)) in enumerate(stream_2):
            n += 1
            # prequential evaluation
            y_i_pred_test = model.predict_one(x_i)
            # learning
            model.learn_one(x_i, y_i)
            # update metric
            metric.update(y_true=y_i, y_pred=y_i_pred_test)
            accuracy.append({"performance": metric.get()})
            # explaining
            y_i_pred = model.predict_one(x_i)
            pfi_i = explainer.explain_one(x_orig=x_i, y_true_orig=y_i, y_pred_orig=y_i_pred)
            pfis.append(copy.deepcopy(pfi_i))
            mdi_i, splits_i = mdi_explainer.explain_one()
            mdi_i = list(mdi_i.values())
            splits.append(copy.deepcopy(splits_i))
            mdis.append(mdi_i)
            if n % 1000 == 0:
                print(f"{n} data         {x_i}\n"
                      f"{n} Performance: {metric.get()}\n"
                      f"{n} iPFI:        {pfi_i}\n"
                      f"{n} mdi          {mdi_i}\n")
            if n >= stream_2_length:
                break

        # plot data ----------------------------------------------------------------------------------------------------

        splits = pd.DataFrame(splits)
        plt.figure()
        plt.plot(splits)
        plt.ylabel('# Splits')
        plt.xlabel('Time')
        plt.legend(feature_names)
        plt.show()

        # plot data ----------------------------------------------------------------------------------------------------

        performance_df = pd.DataFrame(accuracy)
        pfi_df = pd.DataFrame(pfis, columns=feature_names)
        mdi_df = pd.DataFrame(mdis, columns=feature_names)

        performance_df.to_csv("experiment_results/perf_left.csv", index=False)
        pfi_df.to_csv("experiment_results/pfi_df_left.csv", index=False)
        mdi_df.to_csv("experiment_results/mdi_df_left.csv", index=False)

        x_data = [i for i in range(1, stream_length + 1)]

        top_left_data = {'y_data': {"accuracy": performance_df},
                         'x_data': x_data,
                         'line_names': ['performance']}
        top_right_data = top_left_data
        bot_left_data = {'y_data': {'iPFI': pfi_df},
                         'x_data': x_data,
                         'line_names': list(pfi_df.columns)}
        bot_right_data = {'y_data': {'mdi': mdi_df},
                         'x_data': x_data,
                         'line_names': list(mdi_df.columns)}

        # plot styles --------------------------------------------------------------------------------------------------
        top_styles = {
            "y_min": 0, "y_max": 1.025,
            "color_list": ["red"],
            "line_styles": {"accuracy": "solid"},
            "yticks": [0, 0.5, 1]
        }
        top_right_styles = copy.copy(top_styles)
        top_left_styles = copy.copy(top_styles)
        top_left_styles["y_label"] = "Accuracy"
        top_left_styles["title"] = "iPFI"
        top_right_styles["title"] = "MDI"

        bot_styles = {
            "y_min": -0.05, "y_max": 0.59,
            "names_to_highlight": ['salary', 'age', 'elevel', 'commission'],
            "line_styles": {'iPFI': 'solid', 'mdi': 'solid'}
        }
        bot_left_styles = copy.copy(bot_styles)
        bot_right_styles = copy.copy(bot_styles)

        bot_left_styles["y_label"] = "Feature Importance"
        bot_left_styles['legend_lines'] = {'loc': "upper left", 'ncol': 2, 'columnspacing': 0.5, 'labelspacing': 0.25}

        fig, (top_left_axis, top_right_axis, bot_left_axis, bot_right_axis) = plot_double_stacked_line_plots(
            figsize=(12.5, 5),
            top_left_data=top_left_data,
            top_left_styles=top_left_styles,
            bot_left_data=bot_left_data,
            bot_left_styles=bot_left_styles,
            top_right_data=top_right_data,
            top_right_styles=top_right_styles,
            bot_right_data=bot_right_data,
            bot_right_styles=bot_right_styles,
            top_portion=0.25, title=r"iPFI and MDI comparison - data: agrawal, sampling: geometric: $\alpha$: 0.001",
            show=False
        )
        fig.supxlabel("Samples")
        plot_concept_drift_line(top_left_axis, stream_1_length, concept_drift_color=None, line_style='--', show=False)
        plot_concept_drift_line(bot_left_axis, stream_1_length, concept_drift_color=None, line_style='--', show=False)
        plot_concept_drift_line(top_right_axis, stream_1_length, concept_drift_color=None, line_style='--', show=False)
        plot_concept_drift_line(bot_right_axis, stream_1_length, concept_drift_color=None, line_style='--', show=False)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.000, wspace=0.000, bottom=0.1, top=0.9)
        plt.savefig("experiment_results/mdi_ipfi_comparison_agrawal.png", dpi=300)
        plt.show()
