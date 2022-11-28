import copy
import numpy as np
import pandas as pd
import river.metrics
from river.datasets import Elec2
from explainer import IncrementalPFI, BatchPFI
import data_sets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from river import preprocessing, compose


# iPFI Setup
alpha = 0.001  # alpha for exponential smoothing
reservoir_length = 100  # length of the incremental reservoir (k)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(N_INPUT, 100)
        self.layer_2 = nn.Linear(100, 100)
        self.layer_3 = nn.Linear(100, 10)
        self.layer_4 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x


class TorchWrapper:

    def __init__(self, model, optimizer, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self._supervised = True

    def predict_one(self, x_i):
        x_tensor = torch.tensor([list(x_i.values())], dtype=torch.float32)
        return int(torch.argmax(self.model(x_tensor)))

    def learn_one(self, x, y):
        x_tensor = torch.tensor([list(x.values())], dtype=torch.float32)
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(x_tensor)
        y_tensor = torch.zeros(1, 2, dtype=torch.float32)
        y_tensor[0, int(y)] = 1.

        loss = self.loss_function(y_pred, y_tensor)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    feature_names = data_sets.ELEC2_FEATURE_NAMES
    N_INPUT = len(feature_names)

    NETWORK = Net()

    LOSS_FUNCTION = nn.CrossEntropyLoss()
    OPTIMIZER = optim.SGD(NETWORK.parameters(), lr=0.001, momentum=0.9)

    stream = Elec2()
    feature_names = data_sets.ELEC2_FEATURE_NAMES
    stream_length = data_sets.ELEC2_LENGTH

    model = TorchWrapper(model=NETWORK, loss_function=LOSS_FUNCTION, optimizer=OPTIMIZER)
    model = compose.Pipeline(preprocessing.StandardScaler(), model)

    explainers = [
        IncrementalPFI(
            model=model,
            classification=True,
            feature_names=feature_names,
            mode_aggr="EMA",
            alpha=alpha,
            reservoir_length=reservoir_length,
            reservoir_mode='original',
            constant_probability=1,
            samplewise_reservoir=True,
            remove_used_reservoir_sample=False,
            sub_sample_length=1
        )
        for _ in range(10)
    ]

    interval_explainers = [
        BatchPFI(
            model=model,
            classification=True,
            feature_names=feature_names,
            explanation_interval=5000,
            sub_sample_length=5000
        )
        for _ in range(1)
    ]

    performance_metric = river.metrics.Accuracy()
    metric = river.metrics.Rolling(metric=performance_metric, window_size=200)

    performance = []
    pfis = np.zeros(shape=(stream_length, len(feature_names)))
    pfis_std = np.zeros(shape=(stream_length, len(feature_names)))
    pfis_interval = np.zeros(shape=(stream_length, len(feature_names)))

    # Train and Explain
    for (n, (x_i, y_i)) in enumerate(stream):
        # prequential evaluation
        y_i_pred_test = model.predict_one(x_i)
        # learning
        model.learn_one(x_i, y_i)
        # update metric
        metric.update(y_true=y_i, y_pred=y_i_pred_test)
        performance.append({'performance': metric.get()})
        # explaining
        y_i_pred = model.predict_one(x_i)
        pfi_i = []
        for explainer in explainers:
            pfi_i.append(copy.copy(explainer.explain_one(x_orig=x_i, y_true_orig=y_i, y_pred_orig=y_i_pred)))
        pfis[n] = np.mean(np.asarray(pfi_i), axis=0)
        pfis_std[n] = np.std(np.asarray(pfi_i), axis=0)

        pfi_i_interval = []
        for interval_explainer in interval_explainers:
            pfi_i_interval.append(copy.copy(interval_explainer.explain_one(x_orig=x_i, y_true_orig=y_i, y_pred_orig=y_i_pred)))
        pfis_interval[n] = np.mean(np.asarray(pfi_i_interval), axis=0)

        if n % 1000 == 0:
            print(f"{n} Performance: {metric.get()}\n"
                  f"{n} iPFI:        {pfis[n]}\n"
                  f"{n} iPFI std:    {pfis_std[n]}\n"
                  f"{n} interval:    {pfis_interval[n]}")
        if n >= stream_length:
            pfi_i_interval = []
            for interval_explainer in interval_explainers:
                pfi_i_interval.append(copy.copy(interval_explainer.explain_one(x_orig=x_i, y_true_orig=y_i, y_pred_orig=y_i_pred)))
            pfis_interval[-1] = np.mean(np.asarray(pfi_i_interval), axis=0)
            break

    pd.DataFrame(pfis, columns=feature_names).to_csv("experiment_results/elec2_NN_pfi_means.csv", index=False)
    pd.DataFrame(pfis_std, columns=feature_names).to_csv("experiment_results/elec2_NN_pfi_std.csv", index=False)

    pd.DataFrame(pfis_interval, columns=feature_names).to_csv("experiment_results/elec2_NN_pfi_interval.csv", index=False)
    pd.DataFrame(performance).to_csv("experiment_results/elec2_NN_performance.csv", index=False)
