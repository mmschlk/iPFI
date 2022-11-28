import random
from collections import Counter
import numpy as np
import pandas as pd

EPS = 1e-10


class Reservoir:
    def __init__(self, save_targets=True, reservoir_length=100, mode='original', constant_probability=None, remove_used=False):
        assert mode in ['original', 'constant']
        self.mode = mode
        self.reservoir_length = reservoir_length
        self.n = 0
        self.reservoir_x = []
        self.reservoir_y = []
        self.save_targets = save_targets
        if constant_probability is not None:
            self.constant_probability = constant_probability
        else:
            self.constant_probability = 1 / reservoir_length
        self.removed_used = remove_used
        if self.removed_used and self.constant_probability < 1:
            raise ValueError("The probability of adding a new sample must be set to 1, if every used sample is removed.")

    def _add_sample_to_reservoir(self, x, y=None):
        if self.removed_used:
            self.reservoir_x.append(x.copy())
            if self.save_targets:
                self.reservoir_y.append(y)
        else:
            random_insertion_position = random.randint(0, self.reservoir_length - 1)
            self.reservoir_x[random_insertion_position] = x.copy()
            if self.save_targets:
                self.reservoir_y[random_insertion_position] = y

    def _update_constant(self, x, y=None):
        random_float = random.random()
        if random_float <= self.constant_probability:
            self._add_sample_to_reservoir(x, y)

    def _update_original(self, x, y=None):
        random_integer = random.randint(1, self.n)
        if random_integer <= self.reservoir_length:
            self._add_sample_to_reservoir(x, y)

    def update(self, x, y=None):
        self.n += 1
        if self.n <= self.reservoir_length:
            self.reservoir_x.append(x)
            if self.save_targets:
                self.reservoir_y.append(y)
        else:
            if self.mode == 'constant':
                self._update_constant(x, y)
            else:
                self._update_original(x, y)

    def sample(self, sub_sample_length):
        sub_sample_length = min(len(self.reservoir_x), sub_sample_length)
        sub_sample_indices = random.sample(list(range(0, len(self.reservoir_x))), sub_sample_length)
        if self.removed_used:
            x_sample = [self.reservoir_x.pop(sub_sample_index) for sub_sample_index in sub_sample_indices]
        else:
            x_sample = [self.reservoir_x[sub_sample_index] for sub_sample_index in sub_sample_indices]
        if self.save_targets:
            if self.removed_used:
                y_sample = [self.reservoir_y.pop(sub_sample_index) for sub_sample_index in sub_sample_indices]
            else:
                y_sample = [self.reservoir_y[sub_sample_index] for sub_sample_index in sub_sample_indices]
            return x_sample, y_sample
        return x_sample


class IncrementalReservoir:
    def __init__(self, feature_names, reservoir_length=100, mode='skip', constant_probability=None):
        assert mode in ['skip', 'original', 'constant']
        self.mode = mode
        self.reservoir_length = reservoir_length
        self.feature_names = feature_names
        self.n_features = len(self.feature_names)
        self.n = 0
        self.reservoir_permutations = []
        self.reservoir_skips = np.zeros(self.n_features)
        self.W = np.exp(np.log(np.random.rand(self.n_features)) / self.reservoir_length)
        if constant_probability is not None:
            self.constant_probability = constant_probability
        else:
            self.constant_probability = 1 / reservoir_length

    def _update_skip(self, x):
        reservoir_skip_candidates = (np.log(np.random.rand(self.n_features)) / np.log(
            1 - EPS - self.W)).astype(int) + 1
        self.reservoir_skips[self.reservoir_skips < 0] = reservoir_skip_candidates[
            self.reservoir_skips < 0]
        collect = self.reservoir_skips == 0
        for i, feature in enumerate(self.feature_names[collect]):
            random_insertion_position = random.randint(0, self.reservoir_length - 1)
            self.reservoir_permutations[random_insertion_position][feature] = x[feature]
            self.W[i] *= np.exp(np.log(np.random.rand()) / self.reservoir_length)
        self.reservoir_skips -= 1

    def _update_original(self, x):
        for i, feature in enumerate(self.feature_names):
            random_integer = random.randint(1, self.n)
            if random_integer <= self.reservoir_length:
                random_insertion_position = random.randint(0, self.reservoir_length - 1)
                self.reservoir_permutations[random_insertion_position][feature] = x[feature]

    def _update_constant(self, x):
        for i, feature in enumerate(self.feature_names):
            random_float = random.random()
            if random_float <= self.constant_probability:
                random_insertion_position = random.randint(0, self.reservoir_length - 1)
                self.reservoir_permutations[random_insertion_position][feature] = x[feature]

    def update(self, x):
        self.n += 1
        if self.n <= self.reservoir_length:
            permuted_sample = dict((feature, x[feature]) for feature in self.feature_names)
            self.reservoir_permutations.append(permuted_sample)
        else:
            if self.mode == 'skip':
                self._update_skip(x)
            elif self.mode == 'constant':
                self._update_constant(x)
            else:
                self._update_original(x)

    def sample(self, sub_sample_length):
        sub_sample = random.sample(
            self.reservoir_permutations,
            min(len(self.reservoir_permutations), sub_sample_length)
        )
        return sub_sample


class Histogram:

    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.histograms = {}
        for feature in feature_names:
            self.histograms[feature] = Counter()

    def update(self, x):
        for feature in self.feature_names:
            self.histograms[feature].update([x[feature]])

    def sample(self, sub_sample_length):
        sub_sample = []
        for _ in range(sub_sample_length):
            sampled_features = {}
            for feature in self.feature_names:
                values_to_sample, sample_weights = zip(*self.histograms[feature].items())
                random_value = random.sample(values_to_sample, k=1, counts=sample_weights)[0]
                sampled_features[feature] = random_value
            sub_sample.append(sampled_features)
        sub_sample = np.asarray(sub_sample)
        return sub_sample


class IncrementalSampler:

    def __init__(self, histogram_feature_names=None, reservoir_feature_names=None,
                 reservoir_length=100, reservoir_mode='original', constant_probability=None,
                 samplewise_reservoir=False, remove_used_reservoir_sample=False):
        self.reservoir_length = reservoir_length
        self.reservoir_feature_names = reservoir_feature_names
        self.histogram_feature_names = histogram_feature_names
        self.reservoir_mode = reservoir_mode

        if samplewise_reservoir:
            self.numeric_sampler = Reservoir(
                reservoir_length=self.reservoir_length,
                mode=self.reservoir_mode,
                constant_probability=constant_probability,
                save_targets=False,
                remove_used=remove_used_reservoir_sample
            )
        else:
            self.numeric_sampler = IncrementalReservoir(
                feature_names=self.reservoir_feature_names,
                reservoir_length=self.reservoir_length,
                mode=self.reservoir_mode,
                constant_probability=constant_probability
            )

        self.categorical_sampler = Histogram(
            feature_names=self.histogram_feature_names
        )

    def update(self, x):
        self.numeric_sampler.update(x)
        self.categorical_sampler.update(x)

    def sample(self, sub_sample_length):
        permuted_numeric_features = self.numeric_sampler.sample(sub_sample_length)
        permuted_categorical_features = self.categorical_sampler.sample(sub_sample_length)

        permuted_samples = [{**permuted_numeric_features[i], **permuted_categorical_features[i]} for i in
                            range(len(permuted_numeric_features))]
        return permuted_samples


class BatchSampler:

    def __init__(self, feature_names):
        self.x_orig = []
        self.y_orig = []
        self.feature_names = feature_names
        self.n_features = len(self.feature_names)
        self.x_orig_df = pd.DataFrame(columns=self.feature_names)
        self.n = 0

    def update(self, x, y):
        self.x_orig.append(x.copy())
        self.y_orig.append(y)
        self.x_orig_df.loc[self.n] = x
        self.n += 1

    def sample(self, sub_sample_length=None):
        if sub_sample_length is None:
            return self.x_orig, self.y_orig, self.x_orig_df
        else:
            return self.x_orig[-sub_sample_length:], self.y_orig[-sub_sample_length:], self.x_orig_df.tail(sub_sample_length)

