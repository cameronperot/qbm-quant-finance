from abc import ABC, abstractmethod
from time import time

import numpy as np

from qbm.utils import get_rng, log_logistic


class QBMBase(ABC):
    """
    Abstract base class for Quantum Boltzmann Machines

    Inspiration for implementing certain numerical methods taken from:
        - https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
        - sklearn.neural_network.BernoulliRBM
    """

    def __init__(self, X, n_hidden, seed):
        """
        :param X: Training data.
        :param n_hidden: Number of hidden units.
        :param seed: Seed for the random number generator.
        """
        self.X = X
        self.n_visible = X.shape[1]
        self.n_hidden = n_hidden
        self.n_qubits = self.n_visible + self.n_hidden
        self.seed = seed
        self.rng = get_rng(self.seed)
        self.grads = {}

        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self):
        """
        Initializes the weights and biases.
        """
        self.W = self.rng.normal(0, 0.01, (self.n_visible, self.n_hidden))
        # compute the proportion of training vectors in which the units are on
        p = self.X.mean(axis=0)
        # avoid any division by zero errors
        p[p == 1] -= 1e-15
        self.a = np.log(p / (1 - p))
        self.b = np.zeros(self.n_hidden)

    def _random_mini_batch_indices(self, mini_batch_size):
        """
        Generates random, non-intersecting sets of indices for creating mini-batches of the
        training data.

        :param mini_batch_size: Size of the mini-batches.

        :returns: List of numpy arrays, each array containing the indices corresponding to
            a mini-batch.
        """
        return np.split(
            self.rng.permutation(np.arange(self.X.shape[0])),
            np.arange(mini_batch_size, self.X.shape[0], mini_batch_size),
        )

    @abstractmethod
    def _compute_positive_grads(self):
        pass

    @abstractmethod
    def _compute_negative_grads(self):
        pass

    def _apply_grads(self, learning_rate):
        """
        Applies the gradients from the positive and negative phases using the provided
        learning rate.

        :param learning_rate: Learning rate to scale the gradients with.
        """
        self.a += learning_rate * (self.grads["a_pos"] - self.grads["a_neg"])
        self.b += learning_rate * (self.grads["b_pos"] - self.grads["b_neg"])
        self.W += learning_rate * (self.grads["W_pos"] - self.grads["W_neg"])

    def _free_energy(self, V):
        """
        Free energy, F(v) = -sum_i a_i * v_i - sum_j log(1 + exp(b_j + sum_i v_i * w_ij))
        See section 16.1 of https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf for more
        details.

        :param V: Numpy array where the rows are data vectors.

        :returns: Array of shape (V.shape[0],) where the ith entry is F(V[i]).
        """
        return -V @ self.a - np.sum(np.log(1 + np.exp(self.b + V @ self.W)), axis=1)

    def pseudolikelihood(self, V):
        """
        Pseudolikelihood. See section 18.3 of https://www.deeplearningbook.org/contents/partition.html
        for more details.

        :param V: Numpy array where the rows are data vectors.

        :returns: Array of shape (V.shape[0],) where the ith entry is the pseudolikelihood
            of V[i].
        """
        corrupt_indices = (
            np.arange(V.shape[0]),
            self.rng.randint(0, V.shape[1], V.shape[0]),
        )
        V_corrupt = V.copy()
        V_corrupt[corrupt_indices] = 1 - V_corrupt[corrupt_indices]

        free_energy = self._free_energy(V)
        free_energy_corrupt = self._free_energy(V_corrupt)

        return V.shape[1] * log_logistic(free_energy_corrupt - free_energy)

    def train(
        self,
        n_epochs=1000,
        learning_rate=1e-3,
        learning_rate_schedule=None,
        mini_batch_size=16,
        print_interval=None,
        store_pseudolikelihoods=True,
    ):
        """
        Fits the model to the training data.

        :param n_epochs: Number of epochs to train for.
        :param learning_rate: Initial learning rate.
        :param learning_rate_schedule: Array of length n_epochs, where the ith entry is
            used to scale the initial learning rate during epoch i.
        :param mini_batch_size: Size of the mini-batches.
        :param print_interval: How many epochs between printing the pseudolikelihood and
            learning rate. If None, then nothing is printed.
        :param store_pseudolikelihoods: If True will compute and store the pseudolikelihood
            every epoch in the pseudolikelihoods attribute. Note that this can impact
            performance as the pseudolikelihood calculation can be computationally expensive
            depending on the data.
        """
        if learning_rate_schedule is not None:
            assert len(learning_rate_schedule) == n_epochs

        if store_pseudolikelihoods and not hasattr(self, "pseudolikelihoods"):
            self.pseudolikelihoods = []

        for epoch in range(1, n_epochs + 1):
            start_time = time()

            # set the effective learning rate
            learning_rate_effective = learning_rate
            if learning_rate_schedule is not None:
                learning_rate_effective *= learning_rate_schedule[epoch - 1]

            # loop over the mini-batches and compute/apply the gradients for each
            for mini_batch_indices in self._random_mini_batch_indices(mini_batch_size):
                V = self.X[mini_batch_indices]
                self._compute_positive_grads(V)
                self._compute_negative_grads()
                self._apply_grads(learning_rate_effective / V.shape[0])

            # compute and store the pseudolikelihood
            if store_pseudolikelihoods:
                pseudolikelihood = self.pseudolikelihood(self.X).mean()
                self.pseudolikelihoods.append(pseudolikelihood)

            end_time = time()

            # print information if verbose
            if print_interval is not None:
                if epoch % print_interval == 0:
                    if not store_pseudolikelihoods:
                        pseudolikelihood = self.pseudolikelihood(self.X).mean()

                    print(
                        f"[{type(self).__name__}] Epoch {epoch}:",
                        f"pseudolikelihood = {pseudolikelihood:.2f},",
                        f"learning rate = {learning_rate},",
                        f"epoch time = {(end_time - start_time) * 1e3:.2f}ms",
                    )
