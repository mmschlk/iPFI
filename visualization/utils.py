import copy
import os

import pandas as pd
import numpy as np
from pathlib import Path

from data_sets import AGRAWAL_FEATURE_NAMES, STAGGER_FEATURE_NAMES, BIKE_FEATURE_NAMES, \
    OPEN_ML_BANK_MARKETING_FEATURE_NAMES, OPEN_ML_ADULT_FEATURE_NAMES, ELEC2_FEATURE_NAMES

FEATURE_NAMES = {
    'agrawal': AGRAWAL_FEATURE_NAMES,
    'stagger': STAGGER_FEATURE_NAMES,
    'bike': BIKE_FEATURE_NAMES,
    'bank-marketing': OPEN_ML_BANK_MARKETING_FEATURE_NAMES,
    'adult': OPEN_ML_ADULT_FEATURE_NAMES,
    'elec2': ELEC2_FEATURE_NAMES
}

RESULT_PS_SUMMARY_COLUMNS = [
    'run_id', 'dataset', 'model_name', 'model_seed', 'datastream_seed', 'replication', 'timestamp', 'is_ps', 'ps_iteration',
    'explainer', 'time', 'model_performance', 'explanation_stream_length', 'final_pfi', 'run_file', 'concept_drift'
]

RESULT_SUMMARY_COLUMNS = [
    'run_id', 'dataset', 'model_name', 'model_seed', 'datastream_seed', 'replication', 'timestamp', 'dynamic',
    'explainer', 'time', 'model_performance', 'explanation_stream_length', 'final_pfi', 'run_file', 'version',
    'concept_drift', 'alpha', 'reservoir_length', 'concept_drif_time', 'concept_drift_width'
]

EXPERIMENT_DATA_DIR = Path('../experiment_data')
EXPERIMENT_DATA_PS_DIR = Path('../experiment_data/ps')

EXPERIMENT_RESULT_DIR = Path('../experiment_results')
DYNAMIC_EXPERIMENT_RESULT_DIR = Path('../experiment_results/dynamic')
EXPERIMENT_RESULT_SUMMARY_PATH = Path('../experiment_results/result_summary.csv')

EXPERIMENT_RESULT_PS_SUMMARY_PATH = Path('../experiment_results/ps/result_summary.csv')

RENAME_EXPLAINER_NAMES = {
    'noReplacement': 'withReplacement', 'withReplacement': 'noReplacement'
}

COLOR_LIST = ['#1d4289', '#FFB000', '#FE6100', '#FCFF6C']
BASE_COLOR = '#a6a7a9'
SAVE_DPI = 300


NAME_REMAPPING = {
    'EMA_geometric_noReplacement': 'EMA_geometric',
    'EMA_uniform_noReplacement': 'EMA_uniform',
}


def rename_explainers(data):
    for old_name, new_name in NAME_REMAPPING.items():
        data['explainer'] = data['explainer'].str.replace(old_name, new_name)
    return data


def get_final_pfis(series, dataset):
    final_pfis = np.array([final_pfi.split('_') for final_pfi in series], dtype=float)
    final_pfis = pd.DataFrame(final_pfis, columns=FEATURE_NAMES[dataset])
    return final_pfis


def get_index_of_feature(feature_selection, feature_names):
    return [list(feature_names).index(feature) for feature in feature_selection]


def get_ps_result_data():
    return pd.read_csv(EXPERIMENT_RESULT_PS_SUMMARY_PATH)


def get_mean_and_std(data, explainers):
    means_pfi = {}
    stds_pfi = {}
    means_perf = {}
    stds_perf = {}
    for explainer in explainers:
        runs = []
        run_files = data[data.explainer == explainer]['run_file']
        for run_file in run_files:
            run_file = run_file.replace('_dynamic_', '_')
            run_path = os.path.join(EXPERIMENT_DATA_DIR, run_file)
            run_df = pd.read_csv(run_path)
            runs.append(run_df.values)
        runs = np.asarray(runs)
        means = np.mean(runs, axis=0)
        stds = np.std(runs, axis=0)
        means_pfi[explainer] = means[:, :-1]
        means_perf[explainer] = means[:, -1]
        stds_pfi[explainer] = stds[:, :-1]
        stds_perf[explainer] = stds[:, -1]
    return copy.deepcopy(means_pfi), copy.deepcopy(stds_pfi), copy.deepcopy(means_perf), copy.deepcopy(stds_perf)