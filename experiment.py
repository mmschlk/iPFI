import sys
import os
import copy
import random
import time
from pathlib import Path

from joblib import Parallel, delayed

import numpy as np
import river
import pandas as pd
import sklearn.model_selection
from river.metrics import Rolling

from explainer import IncrementalPFI, BatchPFI
import data_sets
from plot import plot_pfis

########################################################################################################################
# Models
########################################################################################################################


class BatchModel:
    def __init__(self, model_name, classification, model_iteration):
        if model_name == 'BRF':
            if classification:
                self.model = sklearn.ensemble.RandomForestClassifier(random_state=model_iteration)
            else:
                self.model = sklearn.ensemble.RandomForestRegressor(random_state=model_iteration)
        elif model_name == 'GBT':
            if classification:
                self.model = sklearn.ensemble.GradientBoostingClassifier(
                    n_estimators=200,
                    random_state=model_iteration
                )
            else:
                self.model = sklearn.ensemble.GradientBoostingRegressor(
                    n_estimators=200,
                    random_state=model_iteration
                )
        elif model_name == 'LGBM':
            if classification:
                self.model = sklearn.ensemble.HistGradientBoostingClassifier(
                    random_state=model_iteration
                )
            else:
                self.model = sklearn.ensemble.HistGradientBoostingRegressor(
                    random_state=model_iteration
                )
        elif model_name == 'NN':
            if classification:
                self.model = sklearn.neural_network.MLPClassifier(
                    hidden_layer_sizes=(128, 64,),
                    random_state=model_iteration,
                    max_iter=50
                )
            else:
                self.model = sklearn.neural_network.MLPRegressor(
                    hidden_layer_sizes=(128, 128, 64, 32,),
                    random_state=model_iteration,
                    max_iter=250
                )
        else:
            raise NotImplementedError("The specified model ('model_name') is not implemented.")

    def predict(self, x_predict):
        return self.model.predict(x_predict)

    def predict_one(self, x_predict):
        x_predict_array = pd.DataFrame([x_predict])
        return self.model.predict(x_predict_array)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)


def get_model(model_name, classification, model_iteration):
    if model_name == 'ARF':
        if classification:
            model = river.ensemble.AdaptiveRandomForestClassifier(seed=model_iteration, binary_split=True, n_models=50)
        else:
            model = river.ensemble.AdaptiveRandomForestRegressor(seed=model_iteration, binary_split=True, n_models=50)
    else:
        model = BatchModel(model_name, classification=classification, model_iteration=model_iteration)
    return model


########################################################################################################################
# Train Models
########################################################################################################################


def train_model(model_name, model, stream, classification, model_iteration, training_steps, print_interval=1000):
    if model_name == 'ARF':
        model_performance = train_incremental_model(model, stream, classification, training_steps,
                                                    print_interval=print_interval)
    else:
        model_performance = train_batch_model(model, stream, classification, model_iteration, training_steps)
    return model_performance


def train_incremental_model(model, stream, classification, training_steps, print_interval=1000):
    if classification:
        performance_metric = river.metrics.Rolling(river.metrics.Accuracy(), window_size=200)
    else:
        performance_metric = river.metrics.Rolling(river.metrics.MAE(), window_size=200)
    for (n, (x_i, y_i)) in enumerate(stream):
        n += 1
        y_i_pred = model.predict_one(x_i)
        performance_metric.update(y_i, y_i_pred)
        model.learn_one(x_i, y_i)
        if n % print_interval == 0:
            print(n, performance_metric.get())
        if n >= training_steps:
            break
    model_performance = performance_metric.get()
    return model_performance


def train_batch_model(model, stream, classification, model_iteration, training_steps):
    if classification:
        performance_metric = sklearn.metrics.accuracy_score
    else:
        performance_metric = sklearn.metrics.mean_absolute_error
    x_data = []
    y_data = []
    for (n, (x_i, y_i)) in enumerate(stream):
        n += 1
        x_data.append(x_i)
        y_data.append(y_i)
        if n >= training_steps:
            break
    x_data = pd.DataFrame(x_data)
    y_data = pd.Series(y_data)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x_data, y_data, test_size=0.33, random_state=model_iteration
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    model_performance = performance_metric(y_test, predictions)
    print('Batch model trained with ', model_performance, 'performance.')
    return model_performance


########################################################################################################################
# Experiment Class
########################################################################################################################


class Experiment:

    def __init__(self, jobs, dataset_parameters, model_name, explainers_parameter, explainers_to_use,
                 constant_model=True, parameter_study=False,
                 random_seed=1, model_iterations=None, shuffled_stream_iterations=None, constant_stream_iterations=None,
                 training_steps=None, ps_iteration=None,
                 print_interval=2000, plotting=False):
        self.jobs = jobs

        dataset_name = dataset_parameters['dataset_name']
        assert dataset_name in [
            'stagger',
            'agrawal',
            'bank-marketing',
            'adult',
            'bike',
            'elec2'
        ], "Dataset ('dataset_name') must be one of 'stagger', 'agrawal', 'bank-marketing', 'adult', or 'bike', or 'elec2'."
        self.dataset_name = dataset_name

        assert model_name in [
            'ARF',
            'BRF',
            'GBT',
            'LGBM',
            'NN'
        ], "Model ('model_name') must be one of 'ARF', 'BRF', 'GBT', 'LGBM', or 'NN'."
        self.model_name = model_name

        if not constant_model and model_name != 'ARF':
            raise ValueError("To do an experiment with an dynamic model requires incremental learner 'ARF'.")
        self.constant_model = constant_model
        self.parameter_study = parameter_study

        self.ps_iteration = ps_iteration
        if parameter_study and self.ps_iteration is None:
            raise ValueError("Parameter Experiment must be set when conducting ps.")

        self.encoding_required = False if self.model_name == 'ARF' else True

        self.explainers_parameter = explainers_parameter
        self.explainers_to_use = explainers_to_use

        np.random.seed(random_seed)
        random.seed(random_seed)

        self.model_iterations = model_iterations if model_iterations is not None else [1]
        self.shuffled_stream_iterations = shuffled_stream_iterations if shuffled_stream_iterations is not None else [1]
        self.constant_stream_iterations = constant_stream_iterations if constant_stream_iterations is not None else [1]
        try:
            iter(self.model_iterations)
            iter(self.shuffled_stream_iterations)
            iter(self.constant_stream_iterations)
        except TypeError:
            raise AssertionError("The Iterations seeds must be provided as a list.")

        self.training_steps = training_steps
        self.explanation_stream_length = dataset_parameters['stream_length']

        self.concept_drift_kind = 'no'
        self.concept_drift_parameters = None
        self.is_concept_drift = dataset_parameters['concept_drift']
        if self.is_concept_drift:
            self.concept_drift_flag = 'cd'
            self.concept_drift_parameters = dataset_parameters['concept_drift_parameters']
            self.concept_drift_kind = self.concept_drift_parameters['concept_drift_kind']

        self.concept_drift_flag = self.concept_drift_kind

        self.experiment_id = '_'.join((self.dataset_name, self.model_name, self.concept_drift_flag))

        self.print_interval = print_interval
        self.plotting = plotting

    def setup_explainers(self, model, classification, feature_names, categorical_feature_names,
                         explanation_stream_length):
        EMA_uniform = IncrementalPFI(
            model=model,
            classification=classification,
            feature_names=feature_names,
            mode='accuracy',
            mode_aggr="EMA",
            alpha=self.explainers_parameter['EMA_uniform']['alpha'],
            reservoir_length=self.explainers_parameter['EMA_uniform']['reservoir_length'],
            sub_sample_length=1,
            reservoir_mode='original',
            samplewise_reservoir=True
        )
        EMA_geometric = IncrementalPFI(
            model=model,
            classification=classification,
            feature_names=feature_names,
            mode='accuracy',
            mode_aggr="EMA",
            alpha=self.explainers_parameter['EMA_geometric']['alpha'],
            reservoir_length=self.explainers_parameter['EMA_geometric']['reservoir_length'],
            sub_sample_length=1,
            reservoir_mode='constant',
            constant_probability=1,
            samplewise_reservoir=True,
            remove_used_reservoir_sample=False
        )
        batch_interval = BatchPFI(
            model=model,
            classification=classification,
            feature_names=feature_names,
            explanation_interval=self.explainers_parameter['batch_interval']['explanation_interval'],
            sub_sample_length=self.explainers_parameter['batch_interval']['explanation_interval']
        )
        exp_int_bt = self.explainers_parameter['batch_total']['explanation_interval']
        batch_total_interval = exp_int_bt if exp_int_bt is not None else explanation_stream_length
        batch_total = BatchPFI(
            model=model,
            classification=classification,
            feature_names=feature_names,
            explanation_interval=batch_total_interval,
            sub_sample_length=self.explainers_parameter['batch_total']['sub_sample_length']
        )

        explainers = {
            'EMA_uniform': EMA_uniform,
            'EMA_geometric': EMA_geometric,
            'batch_interval': batch_interval,
            'batch_total': batch_total
        }
        times = {
            'EMA_uniform': 0, 'EMA_geometric': 0,
            'batch_interval': 0, 'batch_total': 0
        }
        pfis = {
            'EMA_uniform': [], 'EMA_geometric': [],
            'batch_interval': [], 'batch_total': []
        }
        explainers_to_removed = explainers.keys() - self.explainers_to_use
        remove_explainer_from_iteration(explainers, times, pfis, explainers_to_removed)

        return explainers, times, pfis

    def run_experiment(self):
        for model_iteration in self.model_iterations:
            model_iteration_id = '_'.join((self.experiment_id, str(model_iteration)))

            if self.is_concept_drift:
                stream, description = data_sets.get_concept_drift_data_stream(
                    self.dataset_name, random_seed=model_iteration, encoding_required=self.encoding_required,
                    stream_length=self.explanation_stream_length,
                    concept_drift_parameters=self.concept_drift_parameters
                )
            else:
                stream, description = data_sets.get_static_data_stream(
                    self.dataset_name, random_seed=model_iteration, encoding_required=self.encoding_required, stream_length=self.training_steps
                )
            stream_length = description['length']
            classification = description['classification']
            feature_names = description['feature_names']
            categorical_feature_names = description['categorical_features']

            training_steps = stream_length if self.training_steps is None else self.training_steps

            if self.constant_model:
                model = get_model(self.model_name, classification, model_iteration)
                model_performance = train_model(
                    model_name=self.model_name, model=model, stream=stream, classification=classification,
                    model_iteration=model_iteration,
                    training_steps=training_steps, print_interval=self.print_interval
                )
            else:
                model = None
                model_performance = 0

            combinations = [(shuffled_iteration, constant_iteration)
                            for shuffled_iteration in self.shuffled_stream_iterations
                            for constant_iteration in self.constant_stream_iterations]

            Parallel(n_jobs=self.jobs)(delayed(run_shuffle_iteration)(
                self, shuffled_iteration, constant_iteration, classification, feature_names,
                categorical_feature_names,
                stream_length, model_iteration_id, model_performance, model=model, model_iteration=model_iteration)
                                       for shuffled_iteration, constant_iteration in combinations)


def remove_explainer_from_iteration(explainers, times, pfis, explainers_to_remove):
    for explainer_to_remove in explainers_to_remove:
        if explainer_to_remove in explainers:
            del explainers[explainer_to_remove]
            del times[explainer_to_remove]
            del pfis[explainer_to_remove]


########################################################################################################################
# Run experiment method with experiment setup
########################################################################################################################


def run_shuffle_iteration(experiment_setup,
                          shuffled_iteration, constant_iteration, classification, feature_names,
                          categorical_feature_names,
                          stream_length, model_iteration_id, model_performance, model=None, model_iteration=None):
    constant_iteration_id = '_'.join((
        model_iteration_id, str(shuffled_iteration), str(constant_iteration), str(time.time()))
    )
    print('Starting run ' + constant_iteration_id)

    if not experiment_setup.constant_model:
        model = get_model(experiment_setup.model_name, classification=classification, model_iteration=model_iteration)

    if experiment_setup.is_concept_drift:
        stream, description = data_sets.get_concept_drift_data_stream(
            experiment_setup.dataset_name, random_seed=shuffled_iteration, encoding_required=experiment_setup.encoding_required,
            stream_length=experiment_setup.explanation_stream_length,
            concept_drift_parameters=experiment_setup.concept_drift_parameters
        )
    else:
        stream, description = data_sets.get_static_data_stream(
            experiment_setup.dataset_name, random_seed=shuffled_iteration, encoding_required=experiment_setup.encoding_required, stream_length=experiment_setup.explanation_stream_length
        )

    explanation_stream_length = stream_length if experiment_setup.explanation_stream_length is None else experiment_setup.explanation_stream_length

    explainers, times, pfis = experiment_setup.setup_explainers(
        model, classification, feature_names, categorical_feature_names, explanation_stream_length
    )

    if classification:
        performance_metric = river.metrics.Accuracy()
    else:
        performance_metric = river.metrics.MAE()
    performance = []
    rolling_performance = river.metrics.Rolling(metric=performance_metric, window_size=200)

    for (n, (x_i, y_i)) in enumerate(stream):
        n += 1
        if not experiment_setup.constant_model:
            y_i_pred = model.predict_one(x_i)
            rolling_performance.update(y_i, y_i_pred)
            performance.append(rolling_performance.get())
            model.learn_one(x_i, y_i)
        y_i_pred = model.predict_one(x_i)
        for explainer_name, explainer in explainers.items():
            start_time = time.time()
            pfi_i = explainer.explain_one(x_orig=x_i, y_true_orig=y_i, y_pred_orig=y_i_pred)
            end_time = time.time()
            times[explainer_name] += end_time - start_time
            pfis[explainer_name].append(copy.deepcopy(pfi_i))
            if n % experiment_setup.print_interval == 0:
                print_pfi_i = [round(pfi_i_feature, 3) for pfi_i_feature in pfi_i]
                print(n, explainer_name, print_pfi_i)
        if n >= explanation_stream_length:
            break

    # clean up if batch_interval did not finish
    if 'batch_interval' in explainers.keys():
        start_time = time.time()
        pfi_i = explainers['batch_interval'].explain_one(x_orig=x_i, y_true_orig=y_i, y_pred_orig=y_i_pred, force_explain=True)
        end_time = time.time()
        times['batch_interval'] += end_time - start_time
        pfis['batch_interval'][-1] = copy.deepcopy(pfi_i)

    # save run results
    for explainer_name, pfi in pfis.items():
        explainer_id = '_'.join((constant_iteration_id, explainer_name))
        txt_filename = str(constant_iteration_id)
        if not experiment_setup.constant_model:
            explainer_id = '_'.join((explainer_id, 'dynamic'))
            txt_filename = '_'.join((txt_filename, 'dynamic'))
        if experiment_setup.parameter_study:
            txt_filename = '_'.join((txt_filename, 'ps', experiment_setup.ps_iteration))
        txt_filename = txt_filename + '.txt'
        txt_filename = os.path.join(EXPERIMENT_DATA_DIR, txt_filename)
        save_path = os.path.join(EXPERIMENT_DATA_DIR, explainer_id + '.csv')
        experiment_data = pd.DataFrame(pfi, columns=feature_names)
        if not experiment_setup.constant_model:
            experiment_data['performance'] = performance
        experiment_data.to_csv(save_path, index=False, float_format='%.4f')
        with open(txt_filename, 'a+') as f:
            f.write(explainer_id + ' ')
            f.write(str(times[explainer_name]) + ' ')
            f.write(str(model_performance) + ' ')
            f.write(str(explanation_stream_length) + ' ')
            if hasattr(explainers[explainer_name], 'alpha'):
                f.write(str(explainers[explainer_name].alpha) + ' ')
                f.write(str(explainers[explainer_name].reservoir_length) + ' ')
            else:
                f.write(str(0.) + ' ')
                f.write(str(0) + ' ')
            if experiment_setup.is_concept_drift:
                drift_position = str(experiment_setup.concept_drift_parameters['drift_position'])
                drift_width = str(experiment_setup.concept_drift_parameters['drift_width'])
            else:
                drift_position = 'nodrift'
                drift_width = 'nodrift'
            f.write(drift_position + ' ')
            f.write(drift_width + ' ')
            f.write('_'.join(str(pfi_feature) for pfi_feature in pfi[-1]))
            f.write('\n')

    # plot pfi
    if experiment_setup.plotting:
        for explainer_name, pfi in pfis.items():
            plot_title = ' '.join([explainer_name, experiment_setup.dataset_name])
            concept_drift_time = None
            if experiment_setup.is_concept_drift:
                concept_drift_time = int(experiment_setup.concept_drift_parameters['drift_position'] * explanation_stream_length)
            plot_pfis(pfis=pfi, feature_names=feature_names, accuracy=performance, plot_title=plot_title,
                      feature_selection=feature_names, show=True, concept_drift_time=concept_drift_time)


########################################################################################################################
# Main method
########################################################################################################################


if __name__ == "__main__":

    JOBS = 5

    # stagger

    save_dir = 'experiment_data_drifts_time'  # str or None
    if save_dir.lower() == 'none':
        EXPERIMENT_DATA_DIR = Path('experiment_data_test')
    else:
        EXPERIMENT_DATA_DIR = Path(str(save_dir))

    model_iteration = [1]
    shuffled_stream_iteration = [1]  # [1, 2, 3]
    constant_stream_iterations = 10
    constant_stream_iterations = list(range(1, constant_stream_iterations + 1))

    dataset_name = 'stagger'
    model_name = 'ARF'
    constant_model = False
    concept_drift = True
    stream_length = 10000

    explainers_to_use = {
        'EMA_geometric',
        'EMA_uniform',
        'batch_interval'
    }

    # Run 1

    alpha = 0.001

    concept_drift_time = 0.5
    concept_drift_kind = 'fn'  # 'both', 'fn', 'fe', 'no'
    synth_classification_functions = {'class_fun_1': 1, 'class_fun_2': 2}
    feature_switching = None

    explainers_parameter = {
        'EMA_uniform': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'EMA_geometric': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'batch_interval': {
            'explanation_interval': 2000
        },
        'batch_total': {
            'explanation_interval': None,
            'sub_sample_length': None
        }
    }

    dataset_parameters = {
        'dataset_name': dataset_name,
        'stream_length': stream_length,
        'concept_drift': concept_drift,
        'concept_drift_parameters': {
            'concept_drift_kind': concept_drift_kind,
            'drift_position': concept_drift_time,
            'drift_width': 0.01,
            'synth_classification_functions': synth_classification_functions,
            'feature_switching': feature_switching
        }
    }

    experiment = Experiment(
        constant_model=constant_model,
        jobs=JOBS,
        dataset_parameters=dataset_parameters,
        model_name=model_name,
        explainers_to_use=explainers_to_use,
        explainers_parameter=explainers_parameter,
        model_iterations=model_iteration,
        shuffled_stream_iterations=shuffled_stream_iteration,
        constant_stream_iterations=constant_stream_iterations,
        training_steps=None,
        print_interval=1000,
        parameter_study=False,
        ps_iteration=None,
        plotting=False,
    )

    try:
        experiment.run_experiment()
    except Exception as e:
        print("An error occurred.")
        print(e)

    alpha = 0.01

    concept_drift_time = 0.5
    concept_drift_kind = 'fn'  # 'both', 'fn', 'fe', 'no'
    synth_classification_functions = None
    feature_switching = {'class_fun_1': 1, 'class_fun_2': 2}

    explainers_parameter = {
        'EMA_uniform': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'EMA_geometric': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'batch_interval': {
            'explanation_interval': 2000
        },
        'batch_total': {
            'explanation_interval': None,
            'sub_sample_length': None
        }
    }

    dataset_parameters = {
        'dataset_name': dataset_name,
        'stream_length': stream_length,
        'concept_drift': concept_drift,
        'concept_drift_parameters': {
            'concept_drift_kind': concept_drift_kind,
            'drift_position': concept_drift_time,
            'drift_width': 0.01,
            'synth_classification_functions': synth_classification_functions,
            'feature_switching': feature_switching
        }
    }

    experiment = Experiment(
        constant_model=constant_model,
        jobs=JOBS,
        dataset_parameters=dataset_parameters,
        model_name=model_name,
        explainers_to_use=explainers_to_use,
        explainers_parameter=explainers_parameter,
        model_iterations=model_iteration,
        shuffled_stream_iterations=shuffled_stream_iteration,
        constant_stream_iterations=constant_stream_iterations,
        training_steps=None,
        print_interval=1000,
        parameter_study=False,
        ps_iteration=None,
        plotting=False,
    )

    try:
        experiment.run_experiment()
    except Exception as e:
        print("An error occurred.")
        print(e)

    # Adult

    save_dir = 'experiment_data_drifts_time'  # str or None
    if save_dir.lower() == 'none':
        EXPERIMENT_DATA_DIR = Path('experiment_data_test')
    else:
        EXPERIMENT_DATA_DIR = Path(str(save_dir))

    model_iteration = [1]
    shuffled_stream_iteration = [1]  # [1, 2, 3]
    constant_stream_iterations = 10
    constant_stream_iterations = list(range(1, constant_stream_iterations + 1))

    dataset_name = 'adult'
    model_name = 'ARF'
    constant_model = False
    concept_drift = True
    stream_length = 40000

    explainers_to_use = {
        'EMA_geometric',
        'EMA_uniform',
        'batch_interval'
    }

    # Run 1

    alpha = 0.001

    concept_drift_time = 0.5
    concept_drift_kind = 'fe'  # 'both', 'fn', 'fe', 'no'
    synth_classification_functions = None
    feature_switching = {'capitalgain': 'sex'}

    explainers_parameter = {
        'EMA_uniform': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'EMA_geometric': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'batch_interval': {
            'explanation_interval': 2000
        },
        'batch_total': {
            'explanation_interval': None,
            'sub_sample_length': None
        }
    }

    dataset_parameters = {
        'dataset_name': dataset_name,
        'stream_length': stream_length,
        'concept_drift': concept_drift,
        'concept_drift_parameters': {
            'concept_drift_kind': concept_drift_kind,
            'drift_position': concept_drift_time,
            'drift_width': 0.01,
            'synth_classification_functions': synth_classification_functions,
            'feature_switching': feature_switching
        }
    }

    experiment = Experiment(
        constant_model=constant_model,
        jobs=JOBS,
        dataset_parameters=dataset_parameters,
        model_name=model_name,
        explainers_to_use=explainers_to_use,
        explainers_parameter=explainers_parameter,
        model_iterations=model_iteration,
        shuffled_stream_iterations=shuffled_stream_iteration,
        constant_stream_iterations=constant_stream_iterations,
        training_steps=None,
        print_interval=1000,
        parameter_study=False,
        ps_iteration=None,
        plotting=False,
    )

    try:
        experiment.run_experiment()
    except Exception as e:
        print("An error occurred.")
        print(e)

    alpha = 0.01

    concept_drift_time = 0.5
    concept_drift_kind = 'fe'  # 'both', 'fn', 'fe', 'no'
    synth_classification_functions = None
    feature_switching = {'capitalgain': 'sex'}

    explainers_parameter = {
        'EMA_uniform': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'EMA_geometric': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'batch_interval': {
            'explanation_interval': 2000
        },
        'batch_total': {
            'explanation_interval': None,
            'sub_sample_length': None
        }
    }

    dataset_parameters = {
        'dataset_name': dataset_name,
        'stream_length': stream_length,
        'concept_drift': concept_drift,
        'concept_drift_parameters': {
            'concept_drift_kind': concept_drift_kind,
            'drift_position': concept_drift_time,
            'drift_width': 0.01,
            'synth_classification_functions': synth_classification_functions,
            'feature_switching': feature_switching
        }
    }

    experiment = Experiment(
        constant_model=constant_model,
        jobs=JOBS,
        dataset_parameters=dataset_parameters,
        model_name=model_name,
        explainers_to_use=explainers_to_use,
        explainers_parameter=explainers_parameter,
        model_iterations=model_iteration,
        shuffled_stream_iterations=shuffled_stream_iteration,
        constant_stream_iterations=constant_stream_iterations,
        training_steps=None,
        print_interval=1000,
        parameter_study=False,
        ps_iteration=None,
        plotting=False,
    )

    try:
        experiment.run_experiment()
    except Exception as e:
        print("An error occurred.")
        print(e)

    # bike

    save_dir = 'experiment_data_drifts_time'  # str or None
    if save_dir.lower() == 'none':
        EXPERIMENT_DATA_DIR = Path('experiment_data_test')
    else:
        EXPERIMENT_DATA_DIR = Path(str(save_dir))

    model_iteration = [1]
    shuffled_stream_iteration = [1]  # [1, 2, 3]
    constant_stream_iterations = 10
    constant_stream_iterations = list(range(1, constant_stream_iterations + 1))

    dataset_name = 'adult'
    model_name = 'ARF'
    constant_model = False
    concept_drift = True
    stream_length = 40000

    explainers_to_use = {
        'EMA_geometric',
        'EMA_uniform',
        'batch_interval'
    }

    # Run 1

    alpha = 0.001

    concept_drift_time = 0.5
    concept_drift_kind = 'fe'  # 'both', 'fn', 'fe', 'no'
    synth_classification_functions = None
    feature_switching = {'hr': 'temp'}

    explainers_parameter = {
        'EMA_uniform': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'EMA_geometric': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'batch_interval': {
            'explanation_interval': 2000
        },
        'batch_total': {
            'explanation_interval': None,
            'sub_sample_length': None
        }
    }

    dataset_parameters = {
        'dataset_name': dataset_name,
        'stream_length': stream_length,
        'concept_drift': concept_drift,
        'concept_drift_parameters': {
            'concept_drift_kind': concept_drift_kind,
            'drift_position': concept_drift_time,
            'drift_width': 0.01,
            'synth_classification_functions': synth_classification_functions,
            'feature_switching': feature_switching
        }
    }

    experiment = Experiment(
        constant_model=constant_model,
        jobs=JOBS,
        dataset_parameters=dataset_parameters,
        model_name=model_name,
        explainers_to_use=explainers_to_use,
        explainers_parameter=explainers_parameter,
        model_iterations=model_iteration,
        shuffled_stream_iterations=shuffled_stream_iteration,
        constant_stream_iterations=constant_stream_iterations,
        training_steps=None,
        print_interval=1000,
        parameter_study=False,
        ps_iteration=None,
        plotting=False,
    )

    try:
        experiment.run_experiment()
    except Exception as e:
        print("An error occurred.")
        print(e)

    alpha = 0.01

    concept_drift_time = 0.5
    concept_drift_kind = 'fe'  # 'both', 'fn', 'fe', 'no'
    synth_classification_functions = None
    feature_switching = {'hr': 'temp'}

    explainers_parameter = {
        'EMA_uniform': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'EMA_geometric': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'batch_interval': {
            'explanation_interval': 2000
        },
        'batch_total': {
            'explanation_interval': None,
            'sub_sample_length': None
        }
    }

    dataset_parameters = {
        'dataset_name': dataset_name,
        'stream_length': stream_length,
        'concept_drift': concept_drift,
        'concept_drift_parameters': {
            'concept_drift_kind': concept_drift_kind,
            'drift_position': concept_drift_time,
            'drift_width': 0.01,
            'synth_classification_functions': synth_classification_functions,
            'feature_switching': feature_switching
        }
    }

    experiment = Experiment(
        constant_model=constant_model,
        jobs=JOBS,
        dataset_parameters=dataset_parameters,
        model_name=model_name,
        explainers_to_use=explainers_to_use,
        explainers_parameter=explainers_parameter,
        model_iterations=model_iteration,
        shuffled_stream_iterations=shuffled_stream_iteration,
        constant_stream_iterations=constant_stream_iterations,
        training_steps=None,
        print_interval=1000,
        parameter_study=False,
        ps_iteration=None,
        plotting=False,
    )

    try:
        experiment.run_experiment()
    except Exception as e:
        print("An error occurred.")
        print(e)

    # bank

    save_dir = 'experiment_data_drifts_time'  # str or None
    if save_dir.lower() == 'none':
        EXPERIMENT_DATA_DIR = Path('experiment_data_test')
    else:
        EXPERIMENT_DATA_DIR = Path(str(save_dir))

    model_iteration = [1]
    shuffled_stream_iteration = [1]  # [1, 2, 3]
    constant_stream_iterations = 10
    constant_stream_iterations = list(range(1, constant_stream_iterations + 1))

    dataset_name = 'bank-marketing'
    model_name = 'ARF'
    constant_model = False
    concept_drift = True
    stream_length = 40000

    explainers_to_use = {
        'EMA_geometric',
        'EMA_uniform',
        'batch_interval'
    }

    # Run 1

    alpha = 0.001

    concept_drift_time = 0.5
    concept_drift_kind = 'fe'  # 'both', 'fn', 'fe', 'no'
    synth_classification_functions = None
    feature_switching = {'duration': 'housing'}

    explainers_parameter = {
        'EMA_uniform': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'EMA_geometric': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'batch_interval': {
            'explanation_interval': 2000
        },
        'batch_total': {
            'explanation_interval': None,
            'sub_sample_length': None
        }
    }

    dataset_parameters = {
        'dataset_name': dataset_name,
        'stream_length': stream_length,
        'concept_drift': concept_drift,
        'concept_drift_parameters': {
            'concept_drift_kind': concept_drift_kind,
            'drift_position': concept_drift_time,
            'drift_width': 0.01,
            'synth_classification_functions': synth_classification_functions,
            'feature_switching': feature_switching
        }
    }

    experiment = Experiment(
        constant_model=constant_model,
        jobs=JOBS,
        dataset_parameters=dataset_parameters,
        model_name=model_name,
        explainers_to_use=explainers_to_use,
        explainers_parameter=explainers_parameter,
        model_iterations=model_iteration,
        shuffled_stream_iterations=shuffled_stream_iteration,
        constant_stream_iterations=constant_stream_iterations,
        training_steps=None,
        print_interval=1000,
        parameter_study=False,
        ps_iteration=None,
        plotting=False,
    )

    try:
        experiment.run_experiment()
    except Exception as e:
        print("An error occurred.")
        print(e)

    alpha = 0.01

    concept_drift_time = 0.5
    concept_drift_kind = 'fe'  # 'both', 'fn', 'fe', 'no'
    synth_classification_functions = None
    feature_switching = {'duration': 'housing'}

    explainers_parameter = {
        'EMA_uniform': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'EMA_geometric': {
            'alpha': alpha,
            'reservoir_length': 1000
        },
        'batch_interval': {
            'explanation_interval': 2000
        },
        'batch_total': {
            'explanation_interval': None,
            'sub_sample_length': None
        }
    }

    dataset_parameters = {
        'dataset_name': dataset_name,
        'stream_length': stream_length,
        'concept_drift': concept_drift,
        'concept_drift_parameters': {
            'concept_drift_kind': concept_drift_kind,
            'drift_position': concept_drift_time,
            'drift_width': 0.01,
            'synth_classification_functions': synth_classification_functions,
            'feature_switching': feature_switching
        }
    }

    experiment = Experiment(
        constant_model=constant_model,
        jobs=JOBS,
        dataset_parameters=dataset_parameters,
        model_name=model_name,
        explainers_to_use=explainers_to_use,
        explainers_parameter=explainers_parameter,
        model_iterations=model_iteration,
        shuffled_stream_iterations=shuffled_stream_iteration,
        constant_stream_iterations=constant_stream_iterations,
        training_steps=None,
        print_interval=1000,
        parameter_study=False,
        ps_iteration=None,
        plotting=False,
    )

    try:
        experiment.run_experiment()
    except Exception as e:
        print("An error occurred.")
        print(e)


