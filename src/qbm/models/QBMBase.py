from abc import ABC, abstractmethod
from time import time

import numpy as np

from qbm.utils import get_rng, log_logistic


class QBMBase(ABC):
    """
    Abstract base class for Quantum Boltzmann Machines

    Based on the work laid out in:
        - https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021050
    """

    def __init__(self, X_train, n_hidden, seed):
        """
        :param X_train: Training data.
        :param n_hidden: Number of hidden units.
        :param seed: Seed for the random number generator.
        """
        self.X_train = X_train
        self.n_visible = X_train.shape[1]
        self.n_hidden = n_hidden
        self.n_qubits = self.n_visible + self.n_hidden
        self.seed = seed
        self.rng = get_rng(self.seed)
        self.grads = {}

        self._initialize_weights_and_biases()

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

    def _apply_grads(self, learning_rate):
        """
        Applies the gradients from the positive and negative phases using the provided
        learning rate.

        :param learning_rate: Learning rate to scale the gradients with.
        """
        self.a += learning_rate * (self.grads["a_pos"] - self.grads["a_neg"])
        self.b += learning_rate * (self.grads["b_pos"] - self.grads["b_neg"])
        self.W += learning_rate * (self.grads["W_pos"] - self.grads["W_neg"])

    def _binary_to_eigen(self, x):
        """
        Convert bit values {0, 1} to corresponding spin values {+1, -1}.

        :param x: Input array of values {0, 1}.

        :returns: Output array of values {+1, -1}.
        """
        return (1 - 2 * x).astype(np.int8)

    def _eigen_to_binary(self, x):
        """
        Convert spin values {+1, -1} to corresponding bit values {0, 1}.

        :param x: Input array of values {+1, -1}.

        :returns: Output array of values {0, 1}.
        """
        return ((1 - x) / 2).astype(np.int8)

    def _free_energy(self, V):
        """
        Free energy, F(v) = -sum_i a_i * v_i - sum_j log(1 + exp(b_j + sum_i v_i * w_ij))
        See section 16.1 of https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf for more
        details.

        :param V: Numpy array where the rows are data vectors.

        :returns: Array of shape (V.shape[0],) where the ith entry is F(V[i]).
        """
        return -V @ self.a - np.sum(np.log(1 + np.exp(self.b + V @ self.W)), axis=1)

    def _initialize_weights_and_biases(self):
        """
        Initializes the weights and biases.
        """
        self.a = np.zeros(self.n_visible)
        self.b = np.zeros(self.n_hidden)
        self.W = self.rng.normal(0, 0.1, (self.n_visible, self.n_hidden))

    def _mean_energy(self, V, H, VW):
        """
        Computes the mean classical energy w.r.t. the weights and biases over the provided
        visible and hidden unit state vectors.

        :param V: Numpy array where the rows are visible units.
        :param H: Numpy array where the rows are hidden units.
        :param VW: V @ W (used to avoid double computation).

        :returns: Mean energy.
        """
        return (
            -(V @ self.a).sum() - (H @ self.b).sum() - np.einsum("kj,kj", VW, H)
        ) / V.shape[0]

    def _random_mini_batch_indices(self, mini_batch_size):
        """
        Generates random, non-intersecting sets of indices for creating mini-batches of the
        training data.

        :param mini_batch_size: Size of the mini-batches.

        :returns: List of numpy arrays, each array containing the indices corresponding to
            a mini-batch.
        """
        return np.split(
            self.rng.permutation(np.arange(self.X_train.shape[0])),
            np.arange(mini_batch_size, self.X_train.shape[0], mini_batch_size),
        )

    @abstractmethod
    def _compute_positive_grads(self):
        pass

    @abstractmethod
    def _compute_negative_grads(self):
        pass
