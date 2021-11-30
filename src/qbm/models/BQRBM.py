import numpy as np
import pandas as pd

from qbm.utils import get_rng, log_logistic


class BQRBM:
    def __init__(
        self,
        X_train,
        n_hidden,
        s_freeze,
        n_epochs=1000,
        mini_batch_size=16,
        learning_rate=1e-3,
        learning_rate_schedule=None,
        seed=42,
    ):
        self.X_train = X_train
        self.n_visible = X_train.shape[1]
        self.n_hidden = n_hidden
        self.s_freeze = s_freeze
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.learning_rate_schedule = learning_rate_schedule
        self.rng = get_rng(seed)
        self.Gamma = 1
        self.grads = {}

        self._initialize_weights()

    @property
    def Q(self):
        Q = np.diag(np.concatenate((self.a, self.b)))
        Q[: self.n_visible, self.n_visible :] = self.W

        return Q

    def _initialize_weights(self):
        self.W = self.rng.normal(0, 0.01, (self.n_visible, self.n_hidden))
        self.a = self.rng.normal(0, 0.01, self.n_visible)
        self.b = self.rng.normal(0, 0.01, self.n_hidden)

    def _random_mini_batch_indices(self):
        return np.split(
            self.rng.permutation(np.arange(self.X_train.shape[0])),
            np.arange(self.mini_batch_size, self.X_train.shape[0], self.mini_batch_size),
        )

    def _compute_positive_grads(self, V):
        # for einsums: "k" denotes sample index, "i" denotes visible index, and "j" denotes hidden index
        b_eff = self.b + V @ self.W
        D = np.sqrt(self.Gamma ** 2 + b_eff ** 2)
        # TODO: incorporate β/S in tanh?
        b_tanh = (b_eff / D) * np.tanh(D)

        self.grads["a_pos"] = np.mean(V, axis=0)
        self.grads["b_pos"] = np.mean(b_tanh, axis=0)
        self.grads["W_pos"] = np.mean(np.einsum("ki,kj->kij", V, b_tanh), axis=0)

    def _compute_negative_grads(self):
        self.grads["a_neg"] = 0
        self.grads["b_neg"] = 0
        self.grads["W_neg"] = 0

    def _apply_grads(self, learning_rate):
        self.a += learning_rate * (self.grads["a_pos"] - self.grads["a_neg"])
        self.b += learning_rate * (self.grads["b_pos"] - self.grads["b_neg"])
        self.W += learning_rate * (self.grads["W_pos"] - self.grads["W_neg"])

    def _free_energy(self, V):
        # see https://stats.stackexchange.com/a/132872
        return -V @ self.a - np.sum(np.log(1 + np.exp(self.b + V @ self.W)), axis=1)

    def pseudolikelihood(self, V):
        # see https://stats.stackexchange.com/a/306055
        corrupt_indices = (
            np.arange(V.shape[0]),
            self.rng.randint(0, V.shape[1], V.shape[0]),
        )
        V_corrupt = V.copy()
        V_corrupt[corrupt_indices] = 1 - V_corrupt[corrupt_indices]

        free_energy = self._free_energy(V)
        free_energy_corrupt = self._free_energy(V_corrupt)

        return V.shape[1] * log_logistic(free_energy_corrupt - free_energy)

    def _train_step(self, V, learning_rate):
        self._compute_positive_grads(V)
        self._compute_negative_grads()
        self._apply_grads(learning_rate)

    def train(self):
        pseudolikelihoods = []
        for epoch in range(self.n_epochs):
            learning_rate = self.learning_rate
            if self.learning_rate_schedule is not None:
                learning_rate *= learning_rate_schedule[epoch]

            for mini_batch_indices in self._random_mini_batch_indices():
                V = self.X_train[mini_batch_indices]
                self._train_step(V, learning_rate / V.shape[0])

            pseudolikelihoods.append(self.pseudolikelihood(self.X_train))
            print(
                f"Epoch {epoch}: pseudolikelihood = {pseudolikelihoods[-1]}, learning_rate = {learning_rate}"
            )
