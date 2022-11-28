import copy

import numpy as np
import river
import pandas as pd
import time
import scipy

from sampling import IncrementalSampler, BatchSampler


class IncrementalPFI:

    def __init__(self, model, mode='accuracy', classification=True, mode_aggr="SMA", reservoir_mode='original',
                 init_instance=None, feature_names=None, categorical_feature_names=None,
                 reservoir_length=100, sub_sample_length=1, alpha=0.001, p_value=0.01,
                 constant_probability=None, samplewise_reservoir=False, remove_used_reservoir_sample=False,
                 timer_list=None):
        self.model = model

        assert mode in [
            'accuracy',
            'accuracy-coupling',
            'proba',
            'proba-loss'
        ]
        self.mode = mode

        assert mode_aggr in [
            'EMA',
            'SMA'
        ]
        self.mode_aggr = mode_aggr

        assert reservoir_mode in [
            'skip',
            'original',
            'constant'
        ]
        self.reservoir_mode = reservoir_mode

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = np.array(list(init_instance.keys()))
        self.n_features = len(self.feature_names)

        if categorical_feature_names is not None:
            self.categorical_feature_names = categorical_feature_names
            self.numerical_feature_names = np.asarray(list(set(feature_names) - set(self.categorical_feature_names)))
        else:
            self.numerical_feature_names = self.feature_names
            self.categorical_feature_names = np.asarray([])
        self.n_numeric_features = len(self.numerical_feature_names)
        self.n_categorical_features = len(self.categorical_feature_names)

        self.reservoir_length = reservoir_length
        self.sub_sample_length = sub_sample_length
        self.remove_used_reservoir_sample = remove_used_reservoir_sample
        self.warm_up_time = 1 if not self.remove_used_reservoir_sample else self.reservoir_length

        if self.remove_used_reservoir_sample and (constant_probability < 1 or sub_sample_length > 1):
            raise ValueError("If a used reservoir sample is removed directly after usage, the adding probability "
                             "'constant_probability must be set to 1 and the 'sub_sample_length' must be set to 1.")

        self.n = 0
        self.pfi = np.zeros(self.n_features)
        self.pfi_var = np.ones(self.n_features)
        self.pfi_store = []
        self.pfi_store_idx = 0
        self.sma_size = int(2 / alpha)
        self.p_value = p_value
        self.t_threshold = scipy.stats.t.ppf(1 - p_value, self.sma_size - 1)
        # self.sma_size = np.log(0.05)/np.log(1-alpha)
        self.alpha = alpha
        self.constant_probability = constant_probability
        self.classification = classification

        self.sampler = IncrementalSampler(
            histogram_feature_names=self.categorical_feature_names,
            reservoir_feature_names=self.numerical_feature_names,
            reservoir_length=self.reservoir_length,
            reservoir_mode=self.reservoir_mode,
            constant_probability=self.constant_probability,
            samplewise_reservoir=samplewise_reservoir,
            remove_used_reservoir_sample=self.remove_used_reservoir_sample
        )

        self.init_explainer = copy.deepcopy(self)

        self.timer_list = timer_list
        if timer_list is None:
            self.timer_list = []

    def test_hypothesis(self, phi_0):
        t = (self.pfi - phi_0) / np.sqrt(self.pfi_var / self.sma_size)
        return t > self.t_threshold

    def explain_one(self, x_orig, y_true_orig, y_pred_orig):
        self.n += 1
        if self.n > self.warm_up_time:
            x_permuted_samples = self.sampler.sample(self.sub_sample_length)
            self._learn_one_pfi_accuracy(x_orig, y_true_orig, y_pred_orig, x_permuted_samples)
        self.sampler.update(x_orig)
        return self.pfi

    def _learn_one_pfi_accuracy(self, x_orig, y_true_orig, y_pred_orig, x_permuted_samples):
        pfi = np.zeros(self.n_features)
        for j, x_permuted_sample in enumerate(x_permuted_samples):
            time_start = time.time()
            for i, feature in enumerate(self.feature_names):
                if self.classification:
                    pfi[i] += self.model.predict_one({**x_orig, feature: x_permuted_sample[feature]}) != y_true_orig
                else:
                    pfi[i] += np.abs(self.model.predict_one({**x_orig, feature: x_permuted_sample[feature]}) - y_true_orig)
            elapsed_time = time.time() - time_start
            self.timer_list.append(elapsed_time)
            if self.classification:
                pfi -= y_pred_orig != y_true_orig
            else:
                pfi -= np.abs(y_pred_orig - y_true_orig)
        pfi = pfi / len(x_permuted_samples)

        if self.mode_aggr == "SMA":
            self.pfi_store.append(pfi)
            if self.pfi_store_idx < self.sma_size:
                self.pfi_store_idx += 1
            else:
                self.pfi_store.pop(0)
            self.pfi = np.mean(self.pfi_store, axis=0)
            self.pfi_var = np.var(self.pfi_store, axis=0)
        if self.mode_aggr == "EMA":
            diff = pfi - self.pfi
            self.pfi += self.alpha * diff
            self.pfi_var = (1 - self.alpha) * (self.pfi_var + self.alpha * diff ** 2)


class BatchPFI:

    def __init__(self, model, feature_names, classification=True, explanation_interval=1,
                 sub_sample_length=None, exhaustive=False, pfi_samples=1):
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(self.feature_names)
        self.sub_sample_length = sub_sample_length
        self.sampler = BatchSampler(feature_names=self.feature_names)
        self.n = 0
        self.pfi = np.zeros(self.n_features)
        self.explanation_interval = explanation_interval
        self.exhaustive = exhaustive
        self.pfi_samples = pfi_samples
        self.classification = classification

    def explain_one(self, x_orig, y_true_orig, y_pred_orig,
                    force_explain=False, sub_sample_length=None, exhaustive=None, pfi_samples=None):
        sub_sample_length = sub_sample_length if sub_sample_length is not None else self.sub_sample_length
        exhaustive = exhaustive if exhaustive is not None else self.exhaustive
        pfi_samples = pfi_samples if pfi_samples is not None else self.pfi_samples
        self.n += 1
        self.sampler.update(x_orig, y_true_orig)
        if force_explain or (self.n > 1 and self.n % self.explanation_interval == 0):
            x_orig, y_orig, x_orig_df = self.sampler.sample(sub_sample_length)
            if exhaustive:
                self._calculate_pfi_exhaustive(x_orig, y_orig, x_orig_df)
            else:
                self._calculate_pfi(x_orig, y_orig, x_orig_df, pfi_samples)
        return self.pfi

    def _calculate_pfi(self, x_orig, y_orig, x_orig_df, pfi_samples=1):
        pfi = np.zeros(self.n_features)
        n_samples = len(x_orig)
        for pfi_sample_iteration in range(pfi_samples):
            for j, x_j in enumerate(x_orig):
                if self.classification:
                    pfi -= self.model.predict_one(x_j) != y_orig[j]
                else:
                    pfi -= np.abs(self.model.predict_one(x_j) - y_orig[j])
            for i, feature in enumerate(self.feature_names):
                x_permutation_df = x_orig_df.copy(deep=True)
                permutation = np.random.permutation(x_permutation_df[feature])
                x_permutation_df[feature] = permutation
                for j, x_j_permuted in enumerate(x_permutation_df.T.to_dict().values()):
                    if self.classification:
                        pfi[i] += self.model.predict_one(x_j_permuted) != y_orig[j]
                    else:
                        pfi[i] += np.abs(self.model.predict_one(x_j_permuted) - y_orig[j])
            pfi = pfi / len(x_orig)
        self.pfi = pfi / pfi_samples * (n_samples / (n_samples - 1))

    def _calculate_pfi_exhaustive(self, x_orig, y_orig, x_orig_df):
        pfi = np.zeros(self.n_features)
        n_samples = len(x_orig)
        for i, feature in enumerate(self.feature_names):
            for n, x_n in enumerate(x_orig):
                y_n_prediction = self.model.predict_one(x_n)
                orig_error_n = np.abs(y_n_prediction - y_orig[n])
                pfi[i] -= orig_error_n * n_samples
                for m, x_m in enumerate(x_orig):
                    x_permuted = {**x_n, feature: x_m[feature]}  # feature is replaced in x_n by value of x_m
                    y_n_m_prediction = self.model.predict_one(x_permuted)
                    permuted_error_n = np.abs(y_n_m_prediction - y_n_prediction)
                    pfi[i] += permuted_error_n
        self.pfi = pfi * (n_samples / (n_samples - 1))
