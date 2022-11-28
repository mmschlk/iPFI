import openml
import pandas as pd
from river.datasets.base import Dataset
from river.datasets.synth import STAGGER, Agrawal
from sklearn.utils import shuffle
import sklearn
from river.datasets import base
import river
import numpy as np

from river.datasets import Elec2



# stagger data stream
STAGGER_FEATURE_NAMES = np.array(
    [
        'size',
        'color',
        'shape'
    ]
)
STAGGER_CATEGORICAL_FEATURE_NAMES = np.array(
    [
        'size',
        'color',
        'shape'
    ]
)
STAGGER_LENGTH = 10000

# agrawal data stream
AGRAWAL_FEATURE_NAMES = np.array(
    [
        'salary',
        'commission',
        'age',
        'elevel',
        'car',
        'zipcode',
        'hvalue',
        'hyears',
        'loan'
    ]
)
AGRAWAL_CATEGORICAL_FEATURE_NAMES = np.array(
    [
        'elevel',
        'car',
        'zipcode'
    ]
)
AGRAWAL_LENGTH = 20000

# adult data stream (census)
OPEN_ML_ADULT_FEATURE_NAMES = np.array(
    [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capitalloss',
        'hoursperweek',
        'native-country'
    ]
)
OPEN_ML_ADULT_CATEGORICAL_FEATURE_NAMES = np.array(
    [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country'
    ]
)
OPEN_ML_ADULT_NUM_FEATURES = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
OPEN_ML_ADULT_NAN_FEATURES = ['workclass', 'occupation', 'native-country']
OPEN_ML_ADULT_LENGTH = 45222

# bank-marketing data stream (bank)
OPEN_ML_BANK_MARKETING_FEATURE_NAMES = np.array(
    [
        'age',
        'job',
        'marital',
        'education',
        'default',
        'balance',
        'housing',
        'loan',
        'contact',
        'day',
        'month',
        'duration',
        'campaign',
        'pdays',
        'previous',
        'poutcome'
    ]
)
OPEN_ML_BANK_MARKETING_CATEGORICAL_FEATURE_NAMES = np.array(
    [
        'job',
        'marital',
        'education',
        'default',
        'housing',
        'loan',
        'contact',
        'day',
        'month',
        'campaign',
        'poutcome'
    ]
)
OPEN_ML_BANK_MARKETING_RENAME_MAPPER = {
    'V1': 'age',
    'V2': 'job',
    'V3': 'marital',
    'V4': 'education',
    'V5': 'default',
    'V6': 'balance',
    'V7': 'housing',
    'V8': 'loan',
    'V9': 'contact',
    'V10': 'day',
    'V11': 'month',
    'V12': 'duration',
    'V13': 'campaign',
    'V14': 'pdays',
    'V15': 'previous',
    'V16': 'poutcome'
}
OPEN_ML_BANK_MARKETING_NUM_FEATURES = ['age', 'balance', 'duration', 'pdays', 'previous']
OPEN_ML_BANK_MARKETING_LENGTH = 45211

# bike data stream (bike rental regression)
BIKE_FEATURE_NAMES = np.array(
    [
        'season',
        'yr',
        'mnth',
        'hr',
        'holiday',
        'weekday',
        'workingday',
        'weathersit',
        'temp',
        'atemp',
        'hum',
        'windspeed'
    ]
)
BIKE_CATEGORICAL_FEATURE_NAMES = np.array(
    [
        'season',
        'yr',
        'mnth',
        'hr',
        'holiday',
        'weekday',
        'workingday',
        'weathersit'
    ]
)
BIKE_FEATURE_NUM_FEATURES = ['temp', 'atemp', 'hum', 'windspeed']

BIKE_LENGTH = 17379

# ELEC 2
ELEC2_FEATURE_NAMES = np.asarray(
    [
        'date',
        'day',
        'period',
        'nswprice',
        'nswdemand',
        'vicprice',
        'vicdemand',
        'transfer'
    ]
)

ELEC2_CATEGORICAL_FEATURE_NAMES = np.asarray([])

ELEC2_LENGTH = 45312


class FeatureSwitchSTAGGER(STAGGER):

    def __init__(self, feature_switch, *args, **kwargs):
        super(FeatureSwitchSTAGGER, self).__init__(*args, **kwargs)
        self.feature_switch = feature_switch

    def __iter__(self):
        for x, y in super(FeatureSwitchSTAGGER, self).__iter__():
            for feature_1, feature_2 in self.feature_switch.items():
                x[feature_1], x[feature_2] = x[feature_2], x[feature_1]
            yield x, y


class FeatureSwitchAgrawal(Agrawal):

    def __init__(self, feature_switch, *args, **kwargs):
        super(FeatureSwitchAgrawal, self).__init__(*args, **kwargs)
        self.feature_switch = feature_switch

    def __iter__(self):
        for x, y in super(FeatureSwitchAgrawal, self).__iter__():
            for feature_1, feature_2 in self.feature_switch.items():
                x[feature_1], x[feature_2] = x[feature_2], x[feature_1]
            yield x, y


class FeatureSwitchElec2(Elec2):

    def __init__(self, feature_switch):
        super(FeatureSwitchElec2, self).__init__()
        self.feature_switch = feature_switch

    def __iter__(self):
        for x, y in super(FeatureSwitchElec2, self).__iter__():
            for feature_1, feature_2 in self.feature_switch.items():
                x[feature_1], x[feature_2] = x[feature_2], x[feature_1]
            yield x, y


class BatchStream(Dataset):

    def __init__(self, stream_gen, task, n_features, n_classes=None, n_outputs=None):
        super().__init__(task, n_features, n_classes=n_classes, n_outputs=n_outputs)
        self.task = task
        self.n_features = n_features
        self.stream_gen = stream_gen

    def __iter__(self):
        for x_i, y_i in self.stream_gen:
            yield x_i, y_i


def resize_dataset(data, size, random_seed):
    replace = True
    if size <= len(data):
        replace = False
    return data.sample(n=int(size), replace=replace, random_state=random_seed)


def scale_columns(data, columns_to_scale):
    for feature in columns_to_scale:
        scaler = sklearn.preprocessing.StandardScaler()  # RoboustScaler()
        data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))


def encode_columns(data, columns_to_encode):
    for feature in columns_to_encode:
        encoder = sklearn.preprocessing.OrdinalEncoder()
        data[feature] = encoder.fit_transform(data[feature].values.reshape(-1, 1))


def get_open_ml_dataset(open_ml_id, version=1):
    dataset = openml.datasets.get_dataset(open_ml_id, version=version, download_data=True)
    class_label = dataset.default_target_attribute
    print(f"Loaded openML dataset '{dataset.name}', the target feature is '{class_label}'.")
    x_data = dataset.get_data()[0]
    return x_data, class_label


def get_open_ml_datastream(open_ml_id, version=1, random_seed=None):
    x_data, class_label = get_open_ml_dataset(open_ml_id, version=version)
    x_data = shuffle(x_data, random_state=random_seed)
    y_data = x_data.pop(class_label)
    stream_1 = river.stream.iter_pandas(x_data, y_data)
    return stream_1


def get_bike(random_seed=None, remap_features=None, stream_length=None):
    x_data = pd.read_csv('datasets/hour.csv').drop(columns=['instant', 'dteday', 'casual', 'registered'])
    x_data = shuffle(x_data, random_state=random_seed)
    if stream_length is not None:
        x_data = resize_dataset(x_data, stream_length, random_seed)
    y_data = x_data.pop('cnt')
    scale_columns(x_data, BIKE_FEATURE_NUM_FEATURES)
    if remap_features is not None:
        x_data.rename(columns=remap_features, inplace=True)
    stream_1 = river.stream.iter_pandas(x_data, y_data)
    feature_names = x_data.columns
    stream_1 = BatchStream(stream_gen=stream_1, task=base.REG, n_features=len(feature_names), n_outputs=1)
    stream_description = {'length': BIKE_LENGTH, 'feature_names': feature_names, 'n_features': len(feature_names),
                          'categorical_features': BIKE_CATEGORICAL_FEATURE_NAMES, 'classification': False}
    return stream_1, stream_description


def get_adult(random_seed=None, encode=False, remap_features=None, stream_length=None):
    x_data, class_label = get_open_ml_dataset("adult", version=2)
    x_data[OPEN_ML_ADULT_NUM_FEATURES] = x_data[OPEN_ML_ADULT_NUM_FEATURES].apply(pd.to_numeric)
    x_data.dropna(inplace=True)
    if encode:
        encode_columns(x_data, OPEN_ML_ADULT_CATEGORICAL_FEATURE_NAMES)
    scale_columns(x_data, OPEN_ML_ADULT_NUM_FEATURES)
    if stream_length is not None:
        x_data = resize_dataset(x_data, stream_length, random_seed)
    x_data = shuffle(x_data, random_state=random_seed)
    y_data = x_data.pop(class_label)
    if remap_features is not None:
        x_data.rename(columns=remap_features, inplace=True)
    stream_1 = river.stream.iter_pandas(x_data, y_data)
    feature_names = x_data.columns
    stream_1 = BatchStream(stream_gen=stream_1, task=base.BINARY_CLF, n_features=len(feature_names), n_outputs=1)
    stream_description = {'length': OPEN_ML_ADULT_LENGTH, 'feature_names': feature_names, 'n_features': len(feature_names),
                          'categorical_features': OPEN_ML_ADULT_CATEGORICAL_FEATURE_NAMES, 'classification': True}
    return stream_1, stream_description


def get_bank(random_seed=None, encode=False, remap_features=None, stream_length=None):
    x_data, class_label = get_open_ml_dataset("bank-marketing", version=1)
    x_data = x_data.rename(columns=OPEN_ML_BANK_MARKETING_RENAME_MAPPER)
    x_data[OPEN_ML_BANK_MARKETING_NUM_FEATURES] = x_data[OPEN_ML_BANK_MARKETING_NUM_FEATURES].apply(pd.to_numeric)
    if encode:
        encode_columns(x_data, OPEN_ML_BANK_MARKETING_CATEGORICAL_FEATURE_NAMES)
    scale_columns(x_data, OPEN_ML_BANK_MARKETING_NUM_FEATURES)
    if stream_length is not None:
        x_data = resize_dataset(x_data, stream_length, random_seed)
    x_data = shuffle(x_data, random_state=random_seed)
    y_data = x_data.pop(class_label)
    if remap_features is not None:
        x_data.rename(columns=remap_features, inplace=True)
    stream_1 = river.stream.iter_pandas(x_data, y_data)
    feature_names = x_data.columns
    stream_1 = BatchStream(stream_gen=stream_1, task=base.BINARY_CLF, n_features=len(feature_names), n_outputs=1)
    stream_description = {'length': OPEN_ML_BANK_MARKETING_LENGTH, 'feature_names': feature_names,
                          'n_features': len(feature_names),
                          'categorical_features': OPEN_ML_BANK_MARKETING_CATEGORICAL_FEATURE_NAMES,
                          'classification': True}
    return stream_1, stream_description


def get_stagger(classification_function=0, balance_classes=False, random_seed=None, feature_switch=None):
    if feature_switch is None:
        stream_1 = river.synth.STAGGER(classification_function=classification_function, balance_classes=balance_classes,
                                       seed=random_seed)
    else:
        stream_1 = FeatureSwitchSTAGGER(feature_switch=feature_switch, classification_function=classification_function,
                                        balance_classes=balance_classes, seed=random_seed)
    stream_description = {'length': STAGGER_LENGTH, 'feature_names': STAGGER_FEATURE_NAMES,
                          'n_features': len(AGRAWAL_FEATURE_NAMES),
                          'categorical_features': STAGGER_CATEGORICAL_FEATURE_NAMES, 'classification': True}
    return stream_1, stream_description


def get_agrawal(classification_function=1, balance_classes=False, random_seed=None, feature_switch=None):
    if feature_switch is None:
        stream_1 = river.synth.Agrawal(classification_function, balance_classes=balance_classes, seed=random_seed)
    else:
        stream_1 = FeatureSwitchAgrawal(feature_switch=feature_switch, classification_function=classification_function,
                                        balance_classes=balance_classes, seed=random_seed)
    stream_description = {'length': AGRAWAL_LENGTH, 'feature_names': AGRAWAL_FEATURE_NAMES,
                          'n_features': len(AGRAWAL_FEATURE_NAMES),
                          'categorical_features': AGRAWAL_CATEGORICAL_FEATURE_NAMES, 'classification': True}
    return stream_1, stream_description


def get_elec2(random_seed=None, stream_length=None, remap_features=None):
    feature_names = ELEC2_FEATURE_NAMES
    batch_data = []
    batch_y = []
    stream = Elec2()
    for n, (x_i, y_i) in enumerate(stream):
        batch_data.append(x_i)
        batch_y.append(y_i)
    x_data = pd.DataFrame(batch_data, columns=feature_names)
    x_data['label'] = batch_y
    if stream_length is not None:
        x_data = resize_dataset(x_data, stream_length*2, random_seed)
    x_data = shuffle(x_data, random_state=random_seed)
    y_data = x_data.pop('label')
    if remap_features is not None:
        x_data.rename(columns=remap_features, inplace=True)
    stream_1 = river.stream.iter_pandas(x_data, y_data)
    stream_1 = BatchStream(stream_gen=stream_1, task=base.BINARY_CLF, n_features=len(feature_names), n_outputs=1)
    if stream_length is None:
        stream_length = ELEC2_LENGTH
    stream_description = {'length': stream_length, 'feature_names': ELEC2_FEATURE_NAMES,
                          'n_features': len(ELEC2_FEATURE_NAMES),
                          'categorical_features': ELEC2_CATEGORICAL_FEATURE_NAMES, 'classification': True}
    return stream_1, stream_description


def get_static_data_stream(dataset_name, random_seed=None, stream_length=None,
                           stagger_classification_function=0, agrawal_classification_function=1, encoding_required=False):
    if dataset_name == 'stagger':
        stream, stream_description = get_stagger(stagger_classification_function, random_seed=random_seed)
    elif dataset_name == 'agrawal':
        stream, stream_description = get_agrawal(agrawal_classification_function, random_seed=random_seed)
    elif dataset_name == 'elec2':
        stream, stream_description = get_elec2(random_seed=random_seed, stream_length=stream_length)
    elif dataset_name == 'bank-marketing':
        stream, stream_description = get_bank(random_seed=random_seed, encode=encoding_required, stream_length=stream_length)
    elif dataset_name == 'adult':
        stream, stream_description = get_adult(random_seed=random_seed, encode=encoding_required, stream_length=stream_length)
    elif dataset_name == 'bike':
        stream, stream_description = get_bike(random_seed=random_seed, stream_length=stream_length)
    else:
        raise NameError("The dataset name is wrong. Implemented datasets are 'agrawal', 'bank-marketing', 'adult', "
                        "and 'bike'.")
    return stream, stream_description


def get_concept_drift_data_stream(data_set, concept_drift_parameters,
                                  stream_length=20000, balance_classes=False, random_seed=None, encoding_required=False):

    drift_position = concept_drift_parameters['drift_position']
    if drift_position <= 1:
        drift_position = int(stream_length * drift_position)
    drift_width = concept_drift_parameters['drift_width']
    if drift_width <= 1:
        drift_width = int(stream_length * drift_width)

    remap_features = concept_drift_parameters['feature_switching']

    if data_set in {'stagger', 'agrawal'}:
        class_fun_1 = concept_drift_parameters['synth_classification_functions']['class_fun_1']
        class_fun_2 = concept_drift_parameters['synth_classification_functions']['class_fun_2']
        if data_set == 'stagger':
            stream_1, stream_description = get_stagger(classification_function=class_fun_1, balance_classes=balance_classes, random_seed=random_seed)
            stream_2, _ = get_stagger(classification_function=class_fun_2, balance_classes=balance_classes, random_seed=random_seed, feature_switch=remap_features)
        else:
            stream_1, stream_description = get_agrawal(classification_function=class_fun_1, balance_classes=balance_classes, random_seed=random_seed)
            stream_2, _ = get_agrawal(classification_function=class_fun_2, balance_classes=balance_classes, random_seed=random_seed, feature_switch=remap_features)
    elif data_set in {'bank-marketing', 'adult', 'bike', 'elec2'}:
        remap_features.update({v: k for k, v in remap_features.items()})
        stream_length_1 = np.floor(stream_length / 2)
        stream_length_2 = np.ceil(stream_length / 2)
        if data_set == 'bank-marketing':
            stream_1, stream_description = get_bank(random_seed=random_seed, encode=encoding_required, stream_length=stream_length_1)
            stream_2, _ = get_bank(random_seed=random_seed, encode=encoding_required, remap_features=remap_features, stream_length=stream_length_2)
        elif data_set == 'adult':
            stream_1, stream_description = get_adult(random_seed=random_seed, encode=encoding_required, stream_length=stream_length_1)
            stream_2, _ = get_adult(random_seed=random_seed, encode=encoding_required, remap_features=remap_features, stream_length=stream_length_2)
        elif data_set == 'elec2':
            stream_1, stream_description = get_elec2(random_seed=random_seed, stream_length=stream_length_1)
            stream_2, _ = get_elec2(random_seed=random_seed, remap_features=remap_features, stream_length=stream_length_2)
        else:
            stream_1, stream_description = get_bike(random_seed=random_seed, stream_length=stream_length_1)
            stream_2, _ = get_bike(random_seed=random_seed, remap_features=remap_features, stream_length=stream_length_2)
    else:
        raise NotImplementedError(f"The dataset {data_set} and concept drift kind is not implemented yet.")

    stream_description['length'] = stream_length

    concept_drift_stream = river.synth.ConceptDriftStream(
        stream=stream_1,
        drift_stream=stream_2,
        position=drift_position,
        width=drift_width
    )
    return concept_drift_stream, stream_description


if __name__ == "__main__":
    get_elec2()
