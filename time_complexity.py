import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import river.metrics
from river.ensemble import AdaptiveRandomForestClassifier, AdaBoostClassifier, AdaptiveRandomForestRegressor
from river.neighbors import KNNClassifier
import time

from data_sets import get_static_data_stream, get_concept_drift_data_stream
from data_sets import AGRAWAL_FEATURE_NAMES, STAGGER_FEATURE_NAMES, BIKE_FEATURE_NAMES, \
    OPEN_ML_BANK_MARKETING_FEATURE_NAMES, OPEN_ML_ADULT_FEATURE_NAMES
from explainer import IncrementalPFI
import data_sets

FEATURE_NAMES = {
    'agrawal': AGRAWAL_FEATURE_NAMES,
    'stagger': STAGGER_FEATURE_NAMES,
    'bike': BIKE_FEATURE_NAMES,
    'bank-marketing': OPEN_ML_BANK_MARKETING_FEATURE_NAMES,
    'adult': OPEN_ML_ADULT_FEATURE_NAMES
}

concept_drift_kind = 'fe'  # {'fn', 'fe'}  if you want to use concept drift: fn denotes function-drift (only viable for synthetic data (i.e. agrawal or stagger)), fe denotes feature-drift
drift_position = 0.5  # when in the stream, the concept drift should occur
drift_width = 0.01  # width of drift (small values like 0.01 -> sudden drift, large values like 0.2 -> gradual drift)

# iPFI Setup
sample_strategy = 'geometric'  # {'geometric', 'uniform'} denotes which sampling strategy to use
alpha = 0.001  # alpha for exponential smoothing
reservoir_length = 100  # length of the incremental reservoir (k)

ITERATIONS = 9

########################################################################################################################
# Begin Setup
########################################################################################################################
if __name__ == "__main__":

    # Data Stream Setup
    explanation_stream_length = 20_000  # how long the incremental learning procedure should be explained
    dataset_name = 'adult'  # {'agrawal', 'stagger', 'bank-marketing', 'bike', 'adult', 'elec2'} name of the dataset to be used

    # Automatically select the dataset and concept drift
    if concept_drift_kind in {'fe'}:
        if dataset_name == 'agrawal':
            feature_switching = {'elevel': 'salary'}  # make changes if you want
        elif dataset_name == 'stagger':
            feature_switching = {'size': 'color'}  # make changes if you want
        elif dataset_name == 'adult':
            feature_switching = {'capital-gain': 'sex'}  # make changes if you want
        elif dataset_name == 'bike':
            feature_switching = {'hr': 'temp'}  # make changes if you want
        elif dataset_name == 'bank-marketing':
            feature_switching = {'duration': 'housing'}  # make changes if you want
        else:  # dataset_name is elec2
            feature_switching = {'nswprice': 'nswdemand'}  # make changes if you want
        synth_classification_functions = {'class_fun_1': 2,
                                          'class_fun_2': 2}  # default classification function for synth data
    elif concept_drift_kind in {'fn'} and dataset_name in {'agrawal', 'stagger'}:
        synth_classification_functions = {'class_fun_1': 1, 'class_fun_2': 2}  # make changes if you want
        feature_switching = None
    else:
        raise ValueError("Specify correct concept drift kind 'fn' or 'fe' for the right dataset.")

    concept_drift_parameters = {'concept_drift_kind': concept_drift_kind,
                                'drift_position': drift_position,
                                'drift_width': drift_width,
                                'synth_classification_functions': synth_classification_functions,
                                'feature_switching': feature_switching}

    time_learning_list = []
    time_explaining_list = []
    time_predicting_list = []

    for i in range(1, ITERATIONS + 1):  # TOOD revert

        stream, description = get_concept_drift_data_stream(data_set=dataset_name,
                                                            encoding_required=True,
                                                            stream_length=explanation_stream_length,
                                                            concept_drift_parameters=concept_drift_parameters)

        print(f"Dataset: {dataset_name}, "
              f"n_features: {len(description['feature_names'])}, "
              f"classification task: {description['classification']}, "
              f"stream length: {explanation_stream_length}")

        print(description['feature_names'])

        # model to be used
        model = AdaptiveRandomForestClassifier(n_models=25) if description['classification'] else AdaptiveRandomForestRegressor(n_models=25)

        # get the explainers
        if sample_strategy == 'geometric':
            explainer = IncrementalPFI(
                    model=model,
                    classification=description['classification'],
                    feature_names=description['feature_names'],
                    mode_aggr="EMA",
                    alpha=alpha,  # alpha for exponential smoothing
                    reservoir_length=reservoir_length,  # length of the reservoir
                    reservoir_mode='original',  # for uniform sampling - set reservoir_mode to 'original'
                    constant_probability=1,
                    samplewise_reservoir=True,
                    remove_used_reservoir_sample=False
                )
        elif sample_strategy == 'uniform':
            explainer = IncrementalPFI(
                model=model,
                classification=description['classification'],
                feature_names=description['feature_names'],
                mode='accuracy',
                mode_aggr="EMA",
                alpha=alpha,
                reservoir_length=reservoir_length,
                sub_sample_length=1,
                reservoir_mode='original',
                samplewise_reservoir=True
            )
        else:
            raise ValueError("Select a valid sampling strategy.")

        performance_metric = river.metrics.Accuracy() if description['classification'] else river.metrics.MAE()
        metric = river.metrics.Rolling(metric=performance_metric, window_size=200)

        time_learning = 0.
        time_explaining = 0.
        time_predicting = 0.

        performance = []
        # Train and Explain
        pfis = []
        accuracy = []
        for (n, (x_i, y_i)) in enumerate(stream):
            n += 1
            # prequential evaluation
            y_i_pred_test = model.predict_one(x_i)
            # learning
            time_learning_start = time.time()
            model.learn_one(x_i, y_i)
            time_learning += time.time() - time_learning_start
            # update metric
            metric.update(y_true=y_i, y_pred=y_i_pred_test)
            accuracy.append(metric.get())
            # explaining
            y_i_pred = model.predict_one(x_i)
            time_explaining_start = time.time()
            pfi_i = explainer.explain_one(x_orig=x_i, y_true_orig=y_i, y_pred_orig=y_i_pred)
            time_explaining += time.time() - time_explaining_start
            pfis.append(copy.deepcopy(pfi_i))
            if n % 100 == 0:
                continue # print(f"{n} Performance: {metric.get()} iPFI: {pfi_i}") # TOOD revert
            if n >= explanation_stream_length:
                break

        timer_list = explainer.timer_list
        time_predicting = sum(timer_list)

        print(f"\n"
              f"Time Report:                                {i}\n"
              f"Time Predicting:                            {round(time_predicting, 2)}\n"
              f"Time Learning:                              {round(time_learning, 2)}\n"
              f"Time Explaining:                            {round(time_explaining, 2)}\n"
              f"Relative Explaining / Predict:              {round(time_explaining / time_learning, 2)}\n"
              f"Relative Predict / Learning:                {round(time_predicting / time_learning, 2)}\n"
              f"Relative Predict / Explaining:              {round(time_predicting / time_explaining, 2)}\n"
              f"Relative Predict / Explaining * n_features: {round(time_predicting / time_explaining * len(description['feature_names']), 2)}\n")

        time_learning_list.append(time_learning)
        time_explaining_list.append(time_explaining)
        time_predicting_list.append(time_predicting)

    df = pd.DataFrame((time_learning_list, time_explaining_list, time_predicting_list))
    df.to_csv("experiment_results/"+dataset_name+".csv")

    # Data Stream Setup
    explanation_stream_length = 20_000  # how long the incremental learning procedure should be explained
    dataset_name = 'bank-marketing'  # {'agrawal', 'stagger', 'bank-marketing', 'bike', 'adult', 'elec2'} name of the dataset to be used

    # Automatically select the dataset and concept drift
    if concept_drift_kind in {'fe'}:
        if dataset_name == 'agrawal':
            feature_switching = {'elevel': 'salary'}  # make changes if you want
        elif dataset_name == 'stagger':
            feature_switching = {'size': 'color'}  # make changes if you want
        elif dataset_name == 'adult':
            feature_switching = {'capitalgain': 'sex'}  # make changes if you want
        elif dataset_name == 'bike':
            feature_switching = {'hr': 'temp'}  # make changes if you want
        elif dataset_name == 'bank-marketing':
            feature_switching = {'duration': 'housing'}  # make changes if you want
        else:  # dataset_name is elec2
            feature_switching = {'nswprice': 'nswdemand'}  # make changes if you want
        synth_classification_functions = {'class_fun_1': 2,
                                          'class_fun_2': 2}  # default classification function for synth data
    elif concept_drift_kind in {'fn'} and dataset_name in {'agrawal', 'stagger'}:
        synth_classification_functions = {'class_fun_1': 1, 'class_fun_2': 2}  # make changes if you want
        feature_switching = None
    else:
        raise ValueError("Specify correct concept drift kind 'fn' or 'fe' for the right dataset.")

    concept_drift_parameters = {'concept_drift_kind': concept_drift_kind,
                                'drift_position': drift_position,
                                'drift_width': drift_width,
                                'synth_classification_functions': synth_classification_functions,
                                'feature_switching': feature_switching}

    time_learning_list = []
    time_explaining_list = []
    time_predicting_list = []

    for i in range(1, ITERATIONS + 1):  # TOOD revert

        stream, description = get_concept_drift_data_stream(data_set=dataset_name,
                                                            encoding_required=True,
                                                            stream_length=explanation_stream_length,
                                                            concept_drift_parameters=concept_drift_parameters)

        print(f"Dataset: {dataset_name}, "
              f"n_features: {len(description['feature_names'])}, "
              f"classification task: {description['classification']}, "
              f"stream length: {explanation_stream_length}")

        # model to be used
        model = AdaptiveRandomForestClassifier(n_models=25) if description[
            'classification'] else AdaptiveRandomForestRegressor(n_models=25)

        # get the explainers
        if sample_strategy == 'geometric':
            explainer = IncrementalPFI(
                model=model,
                classification=description['classification'],
                feature_names=description['feature_names'],
                mode_aggr="EMA",
                alpha=alpha,  # alpha for exponential smoothing
                reservoir_length=reservoir_length,  # length of the reservoir
                reservoir_mode='original',  # for uniform sampling - set reservoir_mode to 'original'
                constant_probability=1,
                samplewise_reservoir=True,
                remove_used_reservoir_sample=False
            )
        elif sample_strategy == 'uniform':
            explainer = IncrementalPFI(
                model=model,
                classification=description['classification'],
                feature_names=description['feature_names'],
                mode='accuracy',
                mode_aggr="EMA",
                alpha=alpha,
                reservoir_length=reservoir_length,
                sub_sample_length=1,
                reservoir_mode='original',
                samplewise_reservoir=True
            )
        else:
            raise ValueError("Select a valid sampling strategy.")

        performance_metric = river.metrics.Accuracy() if description['classification'] else river.metrics.MAE()
        metric = river.metrics.Rolling(metric=performance_metric, window_size=200)

        time_learning = 0.
        time_explaining = 0.
        time_predicting = 0.

        performance = []
        # Train and Explain
        pfis = []
        accuracy = []
        for (n, (x_i, y_i)) in enumerate(stream):
            n += 1
            # prequential evaluation
            y_i_pred_test = model.predict_one(x_i)
            # learning
            time_learning_start = time.time()
            model.learn_one(x_i, y_i)
            time_learning += time.time() - time_learning_start
            # update metric
            metric.update(y_true=y_i, y_pred=y_i_pred_test)
            accuracy.append(metric.get())
            # explaining
            y_i_pred = model.predict_one(x_i)
            time_explaining_start = time.time()
            pfi_i = explainer.explain_one(x_orig=x_i, y_true_orig=y_i, y_pred_orig=y_i_pred)
            time_explaining += time.time() - time_explaining_start
            pfis.append(copy.deepcopy(pfi_i))
            if n % 100 == 0:
                continue  # print(f"{n} Performance: {metric.get()} iPFI: {pfi_i}") # TOOD revert
            if n >= explanation_stream_length:
                break

        timer_list = explainer.timer_list
        time_predicting = sum(timer_list)

        print(f"\n"
              f"Time Report:                                {i}\n"
              f"Time Predicting:                            {round(time_predicting, 2)}\n"
              f"Time Learning:                              {round(time_learning, 2)}\n"
              f"Time Explaining:                            {round(time_explaining, 2)}\n"
              f"Relative Explaining / Predict:              {round(time_explaining / time_learning, 2)}\n"
              f"Relative Predict / Learning:                {round(time_predicting / time_learning, 2)}\n"
              f"Relative Predict / Explaining:              {round(time_predicting / time_explaining, 2)}\n"
              f"Relative Predict / Explaining * n_features: {round(time_predicting / time_explaining * len(description['feature_names']), 2)}\n")

        time_learning_list.append(time_learning)
        time_explaining_list.append(time_explaining)
        time_predicting_list.append(time_predicting)

    df = pd.DataFrame((time_learning_list, time_explaining_list, time_predicting_list))
    df.to_csv("experiment_results/" + dataset_name + ".csv")

    explanation_stream_length = 15_000  # how long the incremental learning procedure should be explained
    dataset_name = 'bike'  # {'agrawal', 'stagger', 'bank-marketing', 'bike', 'adult', 'elec2'} name of the dataset to be used

    # Automatically select the dataset and concept drift
    if concept_drift_kind in {'fe'}:
        if dataset_name == 'agrawal':
            feature_switching = {'elevel': 'salary'}  # make changes if you want
        elif dataset_name == 'stagger':
            feature_switching = {'size': 'color'}  # make changes if you want
        elif dataset_name == 'adult':
            feature_switching = {'capitalgain': 'sex'}  # make changes if you want
        elif dataset_name == 'bike':
            feature_switching = {'hr': 'temp'}  # make changes if you want
        elif dataset_name == 'bank-marketing':
            feature_switching = {'duration': 'housing'}  # make changes if you want
        else:  # dataset_name is elec2
            feature_switching = {'nswprice': 'nswdemand'}  # make changes if you want
        synth_classification_functions = {'class_fun_1': 2,
                                          'class_fun_2': 2}  # default classification function for synth data
    elif concept_drift_kind in {'fn'} and dataset_name in {'agrawal', 'stagger'}:
        synth_classification_functions = {'class_fun_1': 1, 'class_fun_2': 2}  # make changes if you want
        feature_switching = None
    else:
        raise ValueError("Specify correct concept drift kind 'fn' or 'fe' for the right dataset.")

    concept_drift_parameters = {'concept_drift_kind': concept_drift_kind,
                                'drift_position': drift_position,
                                'drift_width': drift_width,
                                'synth_classification_functions': synth_classification_functions,
                                'feature_switching': feature_switching}

    time_learning_list = []
    time_explaining_list = []
    time_predicting_list = []

    for i in range(1, ITERATIONS + 1):  # TOOD revert

        stream, description = get_concept_drift_data_stream(data_set=dataset_name,
                                                            encoding_required=True,
                                                            stream_length=explanation_stream_length,
                                                            concept_drift_parameters=concept_drift_parameters)

        print(f"Dataset: {dataset_name}, "
              f"n_features: {len(description['feature_names'])}, "
              f"classification task: {description['classification']}, "
              f"stream length: {explanation_stream_length}")

        # model to be used
        model = AdaptiveRandomForestClassifier(n_models=25) if description[
            'classification'] else AdaptiveRandomForestRegressor(n_models=25)

        # get the explainers
        if sample_strategy == 'geometric':
            explainer = IncrementalPFI(
                model=model,
                classification=description['classification'],
                feature_names=description['feature_names'],
                mode_aggr="EMA",
                alpha=alpha,  # alpha for exponential smoothing
                reservoir_length=reservoir_length,  # length of the reservoir
                reservoir_mode='original',  # for uniform sampling - set reservoir_mode to 'original'
                constant_probability=1,
                samplewise_reservoir=True,
                remove_used_reservoir_sample=False
            )
        elif sample_strategy == 'uniform':
            explainer = IncrementalPFI(
                model=model,
                classification=description['classification'],
                feature_names=description['feature_names'],
                mode='accuracy',
                mode_aggr="EMA",
                alpha=alpha,
                reservoir_length=reservoir_length,
                sub_sample_length=1,
                reservoir_mode='original',
                samplewise_reservoir=True
            )
        else:
            raise ValueError("Select a valid sampling strategy.")

        performance_metric = river.metrics.Accuracy() if description['classification'] else river.metrics.MAE()
        metric = river.metrics.Rolling(metric=performance_metric, window_size=200)

        time_learning = 0.
        time_explaining = 0.
        time_predicting = 0.

        performance = []
        # Train and Explain
        pfis = []
        accuracy = []
        for (n, (x_i, y_i)) in enumerate(stream):
            n += 1
            # prequential evaluation
            y_i_pred_test = model.predict_one(x_i)
            # learning
            time_learning_start = time.time()
            model.learn_one(x_i, y_i)
            time_learning += time.time() - time_learning_start
            # update metric
            metric.update(y_true=y_i, y_pred=y_i_pred_test)
            accuracy.append(metric.get())
            # explaining
            y_i_pred = model.predict_one(x_i)
            time_explaining_start = time.time()
            pfi_i = explainer.explain_one(x_orig=x_i, y_true_orig=y_i, y_pred_orig=y_i_pred)
            time_explaining += time.time() - time_explaining_start
            pfis.append(copy.deepcopy(pfi_i))
            if n % 100 == 0:
                continue  # print(f"{n} Performance: {metric.get()} iPFI: {pfi_i}") # TOOD revert
            if n >= explanation_stream_length:
                break

        timer_list = explainer.timer_list
        time_predicting = sum(timer_list)

        print(f"\n"
              f"Time Report:                                {i}\n"
              f"Time Predicting:                            {round(time_predicting, 2)}\n"
              f"Time Learning:                              {round(time_learning, 2)}\n"
              f"Time Explaining:                            {round(time_explaining, 2)}\n"
              f"Relative Explaining / Predict:              {round(time_explaining / time_learning, 2)}\n"
              f"Relative Predict / Learning:                {round(time_predicting / time_learning, 2)}\n"
              f"Relative Predict / Explaining:              {round(time_predicting / time_explaining, 2)}\n"
              f"Relative Predict / Explaining * n_features: {round(time_predicting / time_explaining * len(description['feature_names']), 2)}\n")

        time_learning_list.append(time_learning)
        time_explaining_list.append(time_explaining)
        time_predicting_list.append(time_predicting)

    df = pd.DataFrame((time_learning_list, time_explaining_list, time_predicting_list))
    df.to_csv("experiment_results/" + dataset_name + ".csv")
