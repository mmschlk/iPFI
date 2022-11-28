import numpy as np
import pandas as pd


from visualization.utils import rename_explainers, get_final_pfis, EXPERIMENT_RESULT_SUMMARY_PATH


def create_table(model_dataset_combinations, explainers_to_include):
    data = pd.read_csv(EXPERIMENT_RESULT_SUMMARY_PATH)
    data = rename_explainers(data)
    for dataset_name, model_name in model_dataset_combinations.items():
        batch_data = data[(data['dataset'] == dataset_name) &
                          (data['model_name'] == model_name) &
                          (data['explainer'] == 'batch_total')]
        batch_pfis = get_final_pfis(series=batch_data['final_pfi'], dataset=dataset_name)
        batch_pfis['datastream_seed'] = batch_data['datastream_seed'].values
        datastream_means_batch = batch_pfis.groupby(by=['datastream_seed']).mean()
        for explainer in explainers_to_include:

            sub_data = data[(data['dataset'] == dataset_name) &
                            (data['model_name'] == model_name) &
                            (data['explainer'] == explainer)]
            inc_pfis = get_final_pfis(series=sub_data['final_pfi'], dataset=dataset_name)
            inc_pfis['datastream_seed'] = sub_data['datastream_seed'].values
            datastream_means_inc = inc_pfis.groupby(by=['datastream_seed']).mean()

            datastream_means_inc[datastream_means_inc < 0] = 0
            datastream_means_batch[datastream_means_batch < 0] = 0

            datastream_means_inc = datastream_means_inc.div(datastream_means_inc.sum(axis=1), axis=0)
            datastream_means_batch = datastream_means_batch.div(datastream_means_batch.sum(axis=1), axis=0)

            errors = abs(datastream_means_inc - datastream_means_batch).sum(axis=1)
            print(f"{dataset_name} {model_name} {explainer}"
                  f" Median {round(np.quantile(errors, q=0.5), 3)}"
                  f" IQR {round(np.quantile(errors, q=0.75) - np.quantile(errors, q=0.25), 3)}"
                  f" Quantile(0.25) {round(np.quantile(errors, q=0.25), 3)}"
                  f" Quantile(0.75) {round(np.quantile(errors, q=0.75), 3)}")


if __name__ == "__main__":
    model_dataset_combinations = {
        'agrawal': 'LGBM',
        'elec2': 'LGBM',
        'adult': 'GBT',
        'bank-marketing': 'NN',
        'bike': 'LGBM'
    }

    explainer_names = ['EMA_uniform', 'EMA_geometric']

    create_table(
        model_dataset_combinations=model_dataset_combinations,
        explainers_to_include=explainer_names
    )


