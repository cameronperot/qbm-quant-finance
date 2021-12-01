import numpy as np

from qbm.models import QBMBase


class BQRBM(QBMBase):
    """
    Bound-based Quantum Restricted Boltzmann Machine

    Based on the work laid out in:
        - https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021050
    """

    def __init__(self, X, n_hidden, s_freeze, seed=42):
        """
        :param X: Training data.
        :param n_hidden: Number of hidden units.
        :param s_freeze: The annealing freeze-out time.
        :param seed: Seed for the random number generator.
        """
        super().__init__(X=X, n_hidden=n_hidden, seed=seed)
        self.s_freeze = s_freeze

        self._initialize_weights_and_biases()
        self._compute_Gamma()

    @property
    def Q(self):
        """
        The QUBO coefficient matrix Q.
        """
        Q = np.diag(np.concatenate((self.a, self.b)))
        Q[: self.n_visible, self.n_visible :] = self.W

        return Q

    def _compute_Gamma(self):
        """
        Computes the Γ coefficient in the QBM Hamiltonian based off the annealing schedule
        at the freeze-out time.
        """
        # TODO: implement
        self.Gamma = 1

    def _compute_positive_grads(self, V):
        """
        Computes the gradients for the positive phase, i.e., the expectation values w.r.t.
        the clamped Hamiltonian.

        :param V: Numpy array where the rows are data vectors which to clamp the Hamiltonian
            with.
        """
        # for einsums: "k" denotes sample index, "i" denotes visible index, and "j" denotes hidden index
        b_eff = self.b + V @ self.W
        D = np.sqrt(self.Gamma ** 2 + b_eff ** 2)
        # TODO: incorporate β/S in tanh?
        b_tanh = (b_eff / D) * np.tanh(D)

        self.grads["a_pos"] = np.mean(V, axis=0)
        self.grads["b_pos"] = np.mean(b_tanh, axis=0)
        self.grads["W_pos"] = np.mean(np.einsum("ki,kj->kij", V, b_tanh), axis=0)

    def _compute_negative_grads(self):
        """
        Computes the gradients for the negative phase, i.e., the expectation values w.r.t.
        the model distribution.
        """
        # TODO: implement
        self.grads["a_neg"] = 0
        self.grads["b_neg"] = 0
        self.grads["W_neg"] = 0
