from datetime import timedelta
from time import time

import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from qbm.models import QBMBase
from qbm.utils import convert_bin_str_to_list, load_artifact, save_artifact
from qbm.utils.exact_qbm import compute_H, compute_rho, get_pauli_kron


class BQRBM(QBMBase):
    """
    Bound-based Quantum Restricted Boltzmann Machine

    Based on the work laid out in:
        - https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.021050
    """

    def __init__(
        self,
        X_train,
        n_hidden,
        embedding,
        anneal_params,
        beta_initial=1.5,
        beta_range=(0.1, 10),
        qpu_params={"region": "eu-central-1", "solver": "Advantage_system5.1"},
        exact_params=None,
        seed=0,
    ):
        """
        :param X_train: Training data.
        :param n_hidden: Number of hidden units.
        :param embedding: Embedding of the problem onto the QPU.
        :param anneal_params: Dictionary with the required keys ["schedule", "s", "A", "B",
            "relative_chain_strength"]. A(s) and B(s) values must be prodivded in GHz.
        :param beta_initial: Initial effective β value.
        :param β_range: Range of allowed β values, used for making sure β is not updated
            to an infeasible value (e.g. negative).
        :param qpu_params: Parameters dict to pass (unpacked) to DWaveSampler().
        :param seed: Seed for the random number generator. Used for random minibatches, as
            well as the exact sampler.
        :param exact_params: Dictionary with the optional parameters to use the exact
            sampler. Required keys are ["beta"]
        """
        # convert from binary to ±1 if necessary
        if set(np.unique(X_train)) == set([0, 1]):
            X_train = self._binary_to_eigen(X_train)
        assert set(np.unique(X_train)) == set([-1, 1])

        super().__init__(X_train=X_train, n_hidden=n_hidden, seed=seed)

        self.embedding = embedding
        self.anneal_params = anneal_params
        self.qpu_params = qpu_params
        self.beta = beta_initial
        self.beta_range = beta_range
        self.beta_history = [beta_initial]
        self.A = anneal_params["A"]
        self.B = anneal_params["B"]
        self.exact_params = exact_params

        if self.exact_params is None:
            self._initialize_qpu_sampler()
        else:
            # initialize Pauli Kronecker matrices for exact H computation
            self._pauli_kron = get_pauli_kron(self.n_visible, self.n_hidden)

            # initialize states and state vectors
            self._states = np.arange(2 ** self.n_qubits)
            self._state_vectors = self._binary_to_eigen(
                np.vstack(
                    [
                        convert_bin_str_to_list(f"{x:{f'0{self.n_qubits}b'}}")
                        for x in range(2 ** self.n_qubits)
                    ]
                )
            )

            # set h and J ranges (these values are for Advantage_system5.1)
            self.h_range = np.array([-4, 4])
            self.J_range = np.array([-1, 1])

    def sample(self, n_samples, answer_mode="raw", use_gauge=True, binary=False):
        """
        Generate samples using the model, either exact or from the annealer.

        :param n_samples: Number of samples to generate (num_reads param in sample_ising).
        :param answer_mode: "raw" or "histogram".
        :param use_gauge: If True will use a random gauge transformation (recommended for
            more robust sample generation).
        :param binary: If true will convert the state vector values from {+1, -1} to {0, 1}.

        :returns: Dictionary (exact) or Ocean SDK SampleSet object (annealer).
        """
        if self.exact_params is not None:
            return self._sample_exact(n_samples, binary=binary)
        else:
            return self._sample_annealer(
                n_samples, answer_mode=answer_mode, use_gauge=use_gauge, binary=binary
            )

    def train(
        self,
        n_epochs=1000,
        learning_rate=1e-1,
        learning_rate_beta=1e-1,
        mini_batch_size=16,
        n_samples=10_000,
        callback=None,
    ):
        """
        Fits the model to the training data.

        :param n_epochs: Number of epochs to train for.
        :param learning_rate: Learning rate. If a list or array, then it will represent the
            learning rate over the epochs, must be of length n_epochs.
        :param learning_rate_beta: Learning rate for the effective temperature. If a list or
            array, then it will represent the learning rate over the epochs, must be of
            length n_epochs.
            Note: It might be useful to use a larger learning_rate_beta in the beginning to
            help the model find a good temperature, then drop it after a number of epochs.
        :param mini_batch_size: Size of the mini-batches.
        :param n_samples: Number of samples to generate after every epoch. Used for
            computing β gradient, as well as the callback.
        :param callback: A function called at the end of each epoch. It takes the arguments
            (model, samples), and returns a dictionary with required keys ["value",
            "print"], where the "print" value is a string to be printed at the end of each
            epoch.
        """
        if isinstance(learning_rate, float):
            learning_rate = [learning_rate] * n_epochs
        assert len(learning_rate) == n_epochs

        if isinstance(learning_rate_beta, float):
            learning_rate_beta = [learning_rate_beta] * n_epochs
        assert len(learning_rate_beta) == n_epochs

        if not hasattr(self, "callback_outputs"):
            self.callback_outputs = []

        for epoch in range(1, n_epochs + 1):
            start_time = time()

            # set the effective learning rates
            self.learning_rate = learning_rate[epoch - 1]
            self.learning_rate_beta = learning_rate_beta[epoch - 1]

            # compute and apply gradient updates for each mini batch
            for mini_batch_indices in self._random_mini_batch_indices(mini_batch_size):
                V_pos = self.X_train[mini_batch_indices]
                self._compute_positive_grads(V_pos)
                self._compute_negative_grads(V_pos.shape[0])
                self._apply_grads(self.learning_rate / V_pos.shape[0])
                self._clip_weights_and_biases()

            # update β and clip the weights and biases into proper range
            samples = self.sample(n_samples)
            self._update_beta(samples)

            # callback function
            if callback is not None:
                callback_output = callback(self, self._get_state_vectors(samples))
                self.callback_outputs.append(callback_output)

            # print diagnostics
            end_time = time()
            print(
                f"[{type(self).__name__}] epoch {epoch}:",
                f"β = {self.beta:.3f},",
                f"learning rate = {learning_rate[epoch - 1]:.2e},",
                f"β learning rate = {learning_rate_beta[epoch - 1]:.2e},",
                f"epoch duration = {timedelta(seconds=end_time - start_time)}",
            )
            if callback is not None:
                print("\t" + callback_output["print"])

    def save(self, file_path):
        """
        Saves the BQRBM model at file_path. Necessary because of pickling issues with the
        qpu and sampler objects.

        :param file_path: Path to save the model to.
        """
        if hasattr(self, "qpu"):
            self.qpu = None
        if hasattr(self, "sampler"):
            self.sampler = None

        save_artifact(self, file_path)

    @staticmethod
    def load(file_path):
        """
        Loads the BQRBM model at file_path. Necessary because of pickling issues with the
        qpu and sampler objects.

        :param file_path: Path to the model to load.

        :returns: BQRBM instance.
        """
        model = load_artifact(file_path)
        if model.exact_params is None:
            model._initialize_qpu_sampler()

        return model

    @property
    def h(self):
        return -np.concatenate((self.a, self.b)) / (self.beta * self.B)

    @property
    def J(self):
        J = np.zeros((self.n_qubits, self.n_qubits))
        J[: self.n_visible, self.n_visible :] = -self.W / (self.beta * self.B)
        return J

    def _clip_weights_and_biases(self):
        """
        Clip the weights and biases so that the corresponding h's and J's passed to the
        annealer are within the allowed ranges.
        """
        # here we multiply by (1 - 1e-15) to avoid possible floating point errors that might
        # lead to the computed h's and J's being outside of their allowed ranges
        bias_range = self.h_range * self.beta * self.B * (1 - 1e-15)
        weight_range = self.J_range * self.beta * self.B * (1 - 1e-15)

        self.a = np.clip(self.a, *bias_range)
        self.b = np.clip(self.b, *bias_range)
        self.W = np.clip(self.W, *weight_range)

    def _compute_positive_grads(self, V_pos):
        """
        Computes the gradients for the positive phase, i.e., the expectation values w.r.t.
        the clamped Hamiltonian.

        :param V_pos: Numpy array where the rows are data vectors which to clamp the Hamiltonian
            with.
        """
        b_eff = self.b + V_pos @ self.W
        D = np.sqrt((self.beta * self.A) ** 2 + b_eff ** 2)
        H_pos = (b_eff / D) * np.tanh(D)

        self.grads["a_pos"] = V_pos.mean(axis=0)
        self.grads["b_pos"] = H_pos.mean(axis=0)
        self.grads["W_pos"] = V_pos.T @ H_pos / V_pos.shape[0]

    def _compute_negative_grads(self, mini_batch_size):
        """
        Computes the gradients for the negative phase, i.e., the expectation values w.r.t.
        the model distribution.

        :param mini_batch_size: Number of samples to obtain from the annealer.
        """
        samples = self.sample(mini_batch_size)
        state_vectors = self._get_state_vectors(samples)

        V_neg = state_vectors[:, : self.n_visible]
        b_eff = self.b + V_neg @ self.W
        D = np.sqrt((self.beta * self.A) ** 2 + b_eff ** 2)
        H_neg = (b_eff / D) * np.tanh(D)

        self.grads["a_neg"] = V_neg.mean(axis=0)
        self.grads["b_neg"] = H_neg.mean(axis=0)
        self.grads["W_neg"] = V_neg.T @ H_neg / V_neg.shape[0]

    def _get_state_vectors(self, samples):
        """
        Get the state vectors from the samples (depending on exact or annealer generated).

        :param samples: Return value out of BQRBM.sample().

        :returns: Array of state vectors, shape (n_samples, n_qubits).
        """
        if self.exact_params is not None:
            return samples["state_vectors"]
        else:
            return samples.record.sample

    def _initialize_qpu_sampler(self):
        """
        Initializes the D-Wave sampler using the fixed embedding provided to the object
        instantiation.
        """
        self.qpu = DWaveSampler(**self.qpu_params)
        self.sampler = FixedEmbeddingComposite(self.qpu, self.embedding)
        self.h_range = np.array(self.qpu.properties["h_range"])
        self.J_range = np.array(self.qpu.properties["j_range"])

    def _sample_annealer(self, n_samples, answer_mode="raw", use_gauge=True, binary=False):
        """
        Obtain a sample set using the annealer.

        :param n_samples: Number of samples to generate (num_reads param in sample_ising).
        :param answer_mode: "raw" or "histogram".
        :param use_gauge: If True will use a random gauge transformation (recommended for
            more robust sample generation).
        :param binary: If true will convert the state vector values from {-1, +1} to {0, 1}.

        :returns: Ocean SDK SampleSet object.
        """
        # compute the h's and J's
        h = self.h
        J = self.J

        # apply a random gauge
        if use_gauge:
            gauge = self.rng.choice([-1, 1], self.n_qubits)
            h *= gauge
            J *= np.outer(gauge, gauge)

        # compute the chain strength
        chain_strength = None
        if self.anneal_params.get("relative_chain_strength") is not None:
            J_nonzero = J[: self.n_visible, self.n_visible :]
            chain_strength = self.anneal_params["relative_chain_strength"] * max(
                (
                    max((h.max() / self.h_range.max(), 0)),
                    max((h.min() / self.h_range.min(), 0)),
                    max((J_nonzero.max() / self.J_range.max(), 0)),
                    max((J_nonzero.min() / self.J_range.min(), 0)),
                )
            )

        # get samples from the annealer
        samples = self.sampler.sample_ising(
            h,
            J,
            num_reads=n_samples,
            anneal_schedule=self.anneal_params["schedule"],
            chain_strength=chain_strength,
            answer_mode=answer_mode,
            auto_scale=False,
        )

        # undo the gauge
        if use_gauge:
            samples.record.sample *= gauge

        # convert to binary if specified
        if binary:
            samples.record.sample = self._eigen_to_binary(samples.record.sample)

        return samples

    def _sample_exact(self, n_samples, binary=False):
        """
        Sample using the exact computed probabilities.

        :param n_samples: Number of samples to generate.
        :param binary: If true will convert the state vector values from {-1, +1} to {0, 1}.

        :returns: Samples array of shape (n_samples, n_qubits).
        """
        # compute the h's and J's
        h = self.h
        J = self.J

        # compute the Hamiltonian and density matrix
        H = compute_H(h, J, self.A, self.B, self.n_qubits, self._pauli_kron)
        rho = compute_rho(H, self.exact_params["beta"], diagonal=(self.A == 0))

        # sample using the probabilities on the diagonal of rho
        samples = {}
        samples["E"] = np.diag(H).copy()
        samples["p"] = np.diag(rho).copy()
        samples["states"] = self.rng.choice(self._states, size=n_samples, p=samples["p"])
        samples["state_vectors"] = self._state_vectors[samples["states"]]

        # convert to binary if specified
        if binary:
            samples["state_vectors"] = self._eigen_to_binary(samples["state_vectors"])

        return samples

    def _update_beta(self, samples):
        """
        Updates the effective β = 1 / kT. Used for scaling the coefficients sent to the
        annealer.

        :param samples: Samples to use for computing the mean energies w.r.t. the model.
        """
        # compute the train energy
        VW_train = self.X_train @ self.W
        b_eff = self.b + VW_train
        D = np.sqrt((self.beta * self.A) ** 2 + b_eff ** 2)
        H_train = (b_eff / D) * np.tanh(D)
        E_train = self._mean_energy(self.X_train, H_train, VW_train)

        # compute the model energy
        state_vectors = self._get_state_vectors(samples)
        V_model = state_vectors[:, : self.n_visible]
        VW_model = V_model @ self.W
        H_model = state_vectors[:, self.n_visible :]
        E_model = self._mean_energy(V_model, H_model, VW_model)

        # update the params
        self.beta = np.clip(
            self.beta + self.learning_rate_beta * (E_train - E_model), *self.beta_range
        )
        self.beta_history.append(self.beta)
        self._clip_weights_and_biases()
