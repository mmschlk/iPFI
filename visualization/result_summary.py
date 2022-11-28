import pandas as pd
import os

from visualization.utils import EXPERIMENT_RESULT_SUMMARY_PATH, EXPERIMENT_DATA_DIR, RESULT_SUMMARY_COLUMNS


def add_runs_to_summary():
    result_summary_df = pd.read_csv(EXPERIMENT_RESULT_SUMMARY_PATH)
    indexed_runs = list(result_summary_df['run_id'])
    indexed_runs = set([str(run) + '.txt' for run in indexed_runs])
    runs = set(filter(lambda file: '.txt' in file, os.listdir(EXPERIMENT_DATA_DIR)))
    not_indexed_runs = runs - indexed_runs

    runs_to_append = []
    for run_name in not_indexed_runs:
        run_values = dict.fromkeys(RESULT_SUMMARY_COLUMNS)
        run_id = run_name.split('.txt')[0]
        run_path = os.path.join(EXPERIMENT_DATA_DIR, run_name)
        run_id_features = run_id.split('_')
        run_values['run_id'] = run_id
        run_values['dataset'] = run_id_features[0]
        run_values['model_name'] = run_id_features[1]
        run_values['concept_drift'] = run_id_features[2]
        run_values['model_seed'] = run_id_features[3]
        run_values['datastream_seed'] = run_id_features[4]
        run_values['replication'] = run_id_features[5]
        run_values['timestamp'] = run_id_features[6]
        is_dynamic = len(run_id_features) >= 8 and run_id_features[7] == 'dynamic'
        run_values['dynamic'] = is_dynamic

        with open(run_path, 'r') as run_file:
            explainers = run_file.readlines()

        for explainer in explainers:
            explainer_metadata = explainer.split(' ')
            print(len(explainer_metadata), explainer)
            split_id = explainer_metadata[0].split('_')
            if split_id[-1] == 'dynamic':
                explainer_name = '_'.join((split_id[-3], split_id[-2]))
            else:
                explainer_name = '_'.join((split_id[-2], split_id[-1]))
            run_values['explainer'] = explainer_name
            run_values['time'] = float(explainer_metadata[1])
            run_values['model_performance'] = float(explainer_metadata[2])
            run_values['explanation_stream_length'] = int(explainer_metadata[3])
            run_values['alpha'] = float(explainer_metadata[4])
            run_values['reservoir_length'] = int(explainer_metadata[5])
            run_values['concept_drif_time'] = str(explainer_metadata[6])
            run_values['concept_drift_width'] = str(explainer_metadata[7])
            run_values['final_pfi'] = explainer_metadata[-1].split('\n')[0]
            run_values['run_file'] = explainer_metadata[0] + '.csv'
            run_values['version'] = 2
            runs_to_append.append(run_values.copy())

    result_summary_df = pd.concat((result_summary_df, pd.DataFrame(runs_to_append)))
    result_summary_df.to_csv(EXPERIMENT_RESULT_SUMMARY_PATH, index=False)

if __name__ == "__main__":
    add_runs_to_summary()
