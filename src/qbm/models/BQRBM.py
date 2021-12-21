import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from minorminer import find_embedding
from scipy.special import expit

from qbm.models import QBMBase


class BQRBM(QBMBase):
    """
    Bound-based Quantum Restricted Boltzmann Machine

    Based on the work laid out in:
        - https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021050
    """

    def __init__(
        self,
        X,
        n_hidden,
        embedding,
        annealing_params,
        β_initial=1.5,
        qpu_params={"region": "eu-central-1", "solver": "Advantage_system5.1"},
        seed=42,
    ):
        """
        :param X: Training data.
        :param n_hidden: Number of hidden units.
        :param embedding: Embedding of the problem onto the QPU.
        :param annealing_params: Dictionary with the keys
            ["schedule", "s_freeze", "A_freeze", "B_freeze", "relative_chain_strength"]
            representing the annealing schedule (list of tuples), relative freeze time,
            A(s_freeze), B(s_freeze), and relative chain strength, respectively.
        :param β: Initial effective β of the D-Wave. Used for scaling the coefficients.
        :param qpu_params: Parameters dict to unpack for the qpu.
        :param seed: Seed for the random number generator (used for randomizing minibatches).
        """
        # convert from binary to ±1 if necessary
        if set(np.unique(X)) == set([0, 1]):
            X = 2 * X - 1
        assert set(np.unique(X)) == set([-1, 1])
        self.X_binary = (self.X + 1) / 2

        super().__init__(X=X, n_hidden=n_hidden, seed=seed)
        self.embedding = embedding
        self.annealing_params = annealing_params
        self.β = β
        self.βs = [β]

        self._initialize_sampler()

    def _initialize_sampler(self):
        """
        Initializes the D-Wave sampler using the fixed embedding provided to the object
        instantiation.
        """
        self.qpu = DWaveSampler(**self.qpu_params)
        self.sampler = FixedEmbeddingComposite(self.qpu, self.embedding)
        self.h_range = np.array(qpu.properties["h_range"])
        self.J_range = np.array(qpu.properties["j_range"])

    def _update_β(self):
        """
        Updates the effective β = 1 / kT. Used for scaling the coefficients sent to the
        annealer.
        """
        # set the visible and hidden values to use for the training set calculations
        V_train = self.X
        VW_train = V_train @ self.W
        H_train = expit(VW_train + self.b)

        # compute the classical energy w.r.t. the training set
        E_classical_train = (
            -np.sum(V_train @ self.a) - np.sum(H_train @ self.b) - np.sum(VW_train, H_train)
        ) / V.shape[0]

        # TODO: compute the quantum energy w.r.t. the training set

        # TODO: get samples

        # set the visible and hidden values to use for the model calculations
        V_model = samples.record.sample[:, : self.n_visible]
        VW_model = V_model @ self.W
        H_model = samples.record.sample[:, self.n_visible :]

        # TODO: compute the classical energy w.r.t. to the model
        E_classical_model = (
            -np.sum(V_model @ self.a) - np.sum(H_model @ self.b) - np.sum(VW_model, H_model)
        ) / V.shape[0]

        # TODO: compute the quantum energy w.r.t. to the model

        # combine the energies
        E_train = E_classical_train + E_quantum_train
        E_model = E_classical_model + E_quantum_model

        self.β += self.learning_rate / self.β ** 2 * (E_train - E_model)
        self.βs.append(self.β)
        self.scaling_factor = self.β * self.annealing_params["B_freeze"]

    def _compute_positive_grads(self, V_pos):
        """
        Computes the gradients for the positive phase, i.e., the expectation values w.r.t.
        the clamped Hamiltonian.

        :param V: Numpy array where the rows are data vectors which to clamp the Hamiltonian
            with.
        """
        Γ = self.β * self.annealing_params["A_freeze"]
        b_eff = self.b + V_pos @ self.W
        D = np.sqrt(Γ ** 2 + b_eff ** 2)
        H_pos = (b_eff / D) * np.tanh(D)

        self.grads["a_pos"] = np.mean(V_pos, axis=0)
        self.grads["b_pos"] = np.mean(H_pos, axis=0)
        self.grads["W_pos"] = V_pos.T @ H_pos / V.shape[0]

    def _compute_negative_grads(self, n_samples):
        """
        Computes the gradients for the negative phase, i.e., the expectation values w.r.t.
        the model distribution.

        :param n_samples: Number of samples to obtain from the annealer.
        """
        samples = self.sample(n_samples)
        V_neg = samples.record.samples[:, :n_visible]
        H_neg = samples.record.samples[:, n_visible:]

        self.grads["a_neg"] = V_neg.mean(axis=0)
        self.grads["b_neg"] = H_neg.mean(axis=0)
        self.grads["W_neg"] = V_neg.T @ H_neg / V_neg.shape[0]

    def _clip_weights_and_biases(self):
        """
        Clip the weights and biases so that the corresponding h's and J's passed to the
        annealer are within the allowed ranges.
        """
        # here we multiply by (1 - 1e-15) to avoid possible floating point errors that might
        # lead to the computed h's and J's being outside of their allowed ranges
        bias_range = self.h_range * self.scaling_factor * (1 - 1e-15)
        weight_range = self.J_range * self.scaling_factor * (1 - 1e-15)

        self.a = np.clip(self.a, *bias_range)
        self.b = np.clip(self.b, *bias_range)
        self.W = np.clip(self.W, *weight_range)

    def sample(self, n_samples):
        # compute the h's and J's
        h = np.concatenate((self.a, self.b)) / self.scaling_factor
        J = np.zeros((self.n_qubits, self.n_qubits)) / self.scaling_factor
        J[: self.n_visible, self.n_visible :] = self.W / self.scaling_factor

        # compute the chain strength
        r = max(
            (
                max((h.max() / self.h_range.max(), 0)),
                max((h.min() / self.h_range.min(), 0)),
                max((J[: self.n_visible, self.n_visible :].max() / self.J_range.max(), 0)),
                max((J[: self.n_visible, self.n_visible :].min() / self.J_range.min(), 0)),
            )
        )
        chain_strength = self.relative_chain_strength * r

        # get samples from the annealer
        samples = self.sampler.sample_ising(
            h,
            J,
            num_reads=n_samples,
            anneal_schedule=self.anneal_params["schedule"],
            chain_strength=chain_strength,
            answer_mode="raw",
        )

        return samples

    def train(
        self,
        n_epochs=1000,
        learning_rate=1e-3,
        learning_rate_schedule=None,
        mini_batch_size=16,
        verbose=True,
    ):
        """
        Fits the model to the training data.

        :param n_epochs: Number of epochs to train for.
        :param learning_rate: Initial learning rate.
        :param learning_rate_schedule: Array of length n_epochs, where the ith entry is
            used to scale the initial learning rate during epoch i.
        :param mini_batch_size: Size of the mini-batches.
        :param verbose: If True then will print the pseudolikelihood, learning rate,
            effective β, and epoch duration.
        """
        self.learning_rate = learning_rate

        if learning_rate_schedule is not None:
            assert len(learning_rate_schedule) == n_epochs

        if not hasattr(self, "pseudolikelihoods"):
            self.pseudolikelihoods = []

        for epoch in range(1, n_epochs + 1):
            start_time = time()

            # set the effective learning rate
            if learning_rate_schedule is not None:
                self.learning_rate = learning_rate * learning_rate_schedule[epoch - 1]

            # loop over the mini-batches and compute/apply the gradients for each
            for mini_batch_indices in self._random_mini_batch_indices(mini_batch_size):
                V = self.X[mini_batch_indices]
                self._compute_positive_grads(V)
                self._compute_negative_grads(V.shape[0])
                self._apply_grads(self.learning_rate / V.shape[0])
                self._clip_weights_and_biases()

            # update the effective temperature
            self._update_β()
            self._clip_weights_and_biases()

            # compute and store the pseudolikelihood
            self.pseudolikelihoods.append(self.pseudolikelihood(self.X_binary).mean())

            end_time = time()

            # print information to stdout
            if verbose:
                print(
                    f"[{type(self).__name__}] epoch {epoch}:",
                    f"pseudolikelihood = {self.pseudolikelihoods[-1]:.2f},",
                    f"learning rate = {self.learning_rate:.2e},",
                    f"effective β = {self.β:.2f},",
                    f"epoch duration = {(end_time - start_time) * 1e3:.2f}ms",
                )


if __name__ == "__main__":
    from time import time
    from qbm.utils import get_rng

    rng = get_rng(42)
    X = rng.choice([0, 1], (5000, 64)).astype(np.int8)
    n_hidden = 30
    mini_batch_size = 16

    model = BQRBM(X, n_hidden, s_freeze=1)
    t0 = time()
    model.train(n_epochs=300, mini_batch_size=mini_batch_size, print_interval=10)
    print(time() - t0)
