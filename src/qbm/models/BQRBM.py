from time import time

import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite

from qbm.models import QBMBase
from qbm.utils import convert_bin_str_to_list, load_artifact, save_artifact
from qbm.utils.exact_qbm import compute_H, compute_ρ, sparse_X, sparse_Z, sparse_kron


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
        anneal_params,
        anneal_schedule_data,
        β_initial=1.5,
        β_range=(0.1, 10),
        qpu_params={"region": "eu-central-1", "solver": "Advantage_system5.1"},
        seed=42,
        exact=False,
        β_exact=2,
    ):
        """
        :param X: Training data.
        :param n_hidden: Number of hidden units.
        :param embedding: Embedding of the problem onto the QPU.
        :param anneal_params: Dictionary with the required keys ["schedule", "s",
            "relative_chain_strength"].
        :param anneal_schedule_data: Dataframe which can be used to determine the A(s) and
            B(s) values in the anneal schedule. Should be able to lookup using e.g.
            anneal_schedule_data.loc[s, "A(s) (GHz)"].
        :param β_initial: Initial effective β of the D-Wave. Used for scaling the
            coefficients.
        :param β_range: Range of allowed β values, used for making sure β is not updated
            to an infeasible value (e.g. negative).
        :param qpu_params: Parameters dict to pass (unpacked) to DWaveSampler().
        :param seed: Seed for the random number generator. Used for random minibatches, as
            well as the exact sampler.
        :param exact: If True will use an exact QBM computation rather than the annealer.
        :param β_exact: β value to use for exact sampling.
        """
        # convert from binary to ±1 if necessary
        if set(np.unique(X)) == set([0, 1]):
            X = (1 - 2 * X).astype(np.int8)
        assert set(np.unique(X)) == set([-1, 1])

        super().__init__(X=X, n_hidden=n_hidden, seed=seed)

        self.embedding = embedding
        self.anneal_params = anneal_params
        self.anneal_schedule_data = anneal_schedule_data
        self.qpu_params = qpu_params
        self.β = β_initial
        self.βs = [β_initial]
        self.β_range = β_range
        self.A = anneal_schedule_data.loc[anneal_params["s"], "A(s) (GHz)"]
        self.B = anneal_schedule_data.loc[anneal_params["s"], "B(s) (GHz)"]
        self.exact = exact
        self.β_exact = β_exact
        self.X_binary = ((1 - self.X) / 2).astype(np.int8)

        self._initialize_sampler()

        if self.exact:
            # set Kronecker product Pauli matrices
            self.σ = {}
            for i in range(self.n_qubits):
                self.σ["x", i] = sparse_kron(i, self.n_qubits, sparse_X)
                self.σ["z", i] = sparse_kron(i, self.n_qubits, sparse_Z)

            # initialize states and state vectors
            self.states = np.arange(2 ** self.n_qubits)
            self.state_vectors = np.vstack(
                [
                    1 - 2 * convert_bin_str_to_list(f"{x:{f'0{self.n_qubits}b'}}")
                    for x in range(2 ** self.n_qubits)
                ]
            ).astype(np.int8)

    def save(self, file_path):
        """
        Saves the BQRBM model at file_path. Necessary because of pickling issues with the
        qpu and sampler objects.

        :param file_path: Path to save the model to.
        """
        self.qpu = None
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
        model._initialize_sampler()
        return model

    @property
    def Γ(self):
        return self.β * self.A

    def _initialize_sampler(self):
        """
        Initializes the D-Wave sampler using the fixed embedding provided to the object
        instantiation.
        """
        self.qpu = DWaveSampler(**self.qpu_params)
        self.sampler = FixedEmbeddingComposite(self.qpu, self.embedding)
        self.h_range = np.array(self.qpu.properties["h_range"])
        self.J_range = np.array(self.qpu.properties["j_range"])

    def _compute_mean_energy(self, V, H, VW):
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

    def _update_β(self, samples, learning_rate):
        """
        Updates the effective β = 1 / kT. Used for scaling the coefficients sent to the
        annealer.
        """
        # compute the train energy
        V_train = self.X
        VW_train = V_train @ self.W
        b_eff = self.b + VW_train
        D = np.sqrt(self.Γ ** 2 + b_eff ** 2)
        H_train = (b_eff / D) * np.tanh(D)
        E_train = self._compute_mean_energy(V_train, H_train, VW_train)

        # compute the model energy
        state_vectors = self._get_state_vectors(samples)
        V_model = state_vectors[:, : self.n_visible]
        VW_model = V_model @ self.W
        H_model = state_vectors[:, self.n_visible :]
        E_model = self._compute_mean_energy(V_model, H_model, VW_model)

        # update the params
        Δβ = learning_rate * (E_train - E_model)
        self.β = np.clip(self.β + Δβ, *self.β_range)
        self.βs.append(self.β)
        self._clip_weights_and_biases()

    def _compute_positive_grads(self, V_pos):
        """
        Computes the gradients for the positive phase, i.e., the expectation values w.r.t.
        the clamped Hamiltonian.

        :param V_pos: Numpy array where the rows are data vectors which to clamp the Hamiltonian
            with.
        """
        b_eff = self.b + V_pos @ self.W
        D = np.sqrt(self.Γ ** 2 + b_eff ** 2)
        H_pos = (b_eff / D) * np.tanh(D)

        self.grads["a_pos"] = V_pos.mean(axis=0)
        self.grads["b_pos"] = H_pos.mean(axis=0)
        self.grads["W_pos"] = V_pos.T @ H_pos / V_pos.shape[0]

    def _compute_negative_grads(self, n_samples):
        """
        Computes the gradients for the negative phase, i.e., the expectation values w.r.t.
        the model distribution.

        :param n_samples: Number of samples to obtain from the annealer.
        """
        samples = self.sample(n_samples)
        state_vectors = self._get_state_vectors(samples)

        V_neg = state_vectors[:, : self.n_visible]
        H_neg = state_vectors[:, self.n_visible :]

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
        bias_range = self.h_range * self.β * self.B * (1 - 1e-15)
        weight_range = self.J_range * self.β * self.B * (1 - 1e-15)

        self.a = np.clip(self.a, *bias_range)
        self.b = np.clip(self.b, *bias_range)
        self.W = np.clip(self.W, *weight_range)

    def sample(self, n_samples, answer_mode="raw", use_gauge=True, binary=False):
        """
        Generate samples using the model, either exact or from the annealer.

        :param n_samples: Number of samples to generate (num_reads param in sample_ising).
        :param answer_mode: "raw" or "histogram".
        :param use_gauge: If True will use a random gauge transformation (recommended for
            more robust sample generation).
        :param binary: If true will convert the state vector values from {-1, +1} to {0, 1}.

        :returns: Dictionary (exact) or Ocean SDK SampleSet object (annealer).
        """
        if self.exact:
            return self._sample_exact(n_samples, binary=binary)
        else:
            return self._sample_annealer(
                n_samples, answer_mode=answer_mode, use_gauge=use_gauge, binary=binary
            )

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
        h = -np.concatenate((self.a, self.b)) / (self.β * self.B)
        J = np.zeros((self.n_qubits, self.n_qubits))
        J[: self.n_visible, self.n_visible :] = -self.W / (self.β * self.B)

        # apply a random gauge
        if use_gauge:
            gauge = self.rng.choice([-1, 1], self.n_qubits)
            h *= gauge
            J *= np.outer(gauge, gauge)

        # compute the chain strength
        chain_strength_factor = max(
            (
                max((h.max() / self.h_range.max(), 0)),
                max((h.min() / self.h_range.min(), 0)),
                max((J[: self.n_visible, self.n_visible :].max() / self.J_range.max(), 0)),
                max((J[: self.n_visible, self.n_visible :].min() / self.J_range.min(), 0)),
            )
        )
        chain_strength = self.relative_chain_strength * chain_strength_factor

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
            samples.record.samples *= gauge

        # convert to binary if specified
        if binary:
            samples.record.samples = ((samples.record.samples + 1) / 2).astype(np.int8)

        return samples

    def _sample_exact(self, n_samples, binary=False):
        """
        Sample using the exact computed probabilities.

        :param n_samples: Number of samples to generate.
        :param binary: If true will convert the state vector values from {-1, +1} to {0, 1}.

        :returns: Samples array of shape (n_samples, n_qubits).
        """
        # compute the h's and J's
        h = -np.concatenate((self.a, self.b)) / (self.β * self.B)
        J = np.zeros((self.n_qubits, self.n_qubits))
        J[: self.n_visible, self.n_visible :] = -self.W / (self.β * self.B)

        # compute the Hamiltonian and density matrix
        H = compute_H(h, J, self.A, self.B, self.n_qubits, self.σ)
        ρ = compute_ρ(H, self.β_exact)

        # sample using the probabilities on the diagonal of ρ
        samples = {}
        samples["E"] = np.diag(H).copy()
        samples["p"] = np.diag(ρ).copy()
        samples["states"] = self.rng.choice(self.states, size=n_samples, p=samples["p"])
        samples["state_vectors"] = self.state_vectors[samples["states"]]

        # convert to binary if specified
        if binary:
            samples["state_vectors"] = ((1 - samples["state_vectors"]) / 2).astype(np.int8)

        return samples

    def _get_state_vectors(self, samples):
        """
        Get the state vectors from the samples (depending on exact or annealer generated).

        :param samples: Return value out of BQRBM.sample().

        :returns: Array of state vectors, shape (n_samples, n_qubits).
        """
        if self.exact:
            return samples["state_vectors"]
        else:
            return samples.record.sample

    def train(
        self,
        n_epochs=1000,
        learning_rate=1e-1,
        learning_rate_β=1e-1,
        mini_batch_size=16,
        n_samples=10_000,
        callback=None,
    ):
        """
        Fits the model to the training data.

        :param n_epochs: Number of epochs to train for.
        :param learning_rate: Learning rate. If a list or array, then it will represent the
            learning rate over the epochs, must be of length n_epochs.
        :param learning_rate: Learning rate for the effective temperature. If a list or
            array, then it will represent the learning rate over the epochs, must be of
            length n_epochs.
            Note: It might be useful to use a larger learning_rate_β in the beginning to
            help the model find a good temperature, then drop it after a number of epochs.
        :param mini_batch_size: Size of the mini-batches.
        :param n_samples: Number of samples to generate after every epoch. Used for
            computing β gradient, as well as the callback.
        :param callback: A function called at the end of each epoch. It takes the arguments
            (X_train, sample_state_vectors), and returns a dictionary with keys ["value",
            "print"], where the "print" value is a string to be printed at the end of each
            epoch.
        """
        learning_rates = None
        if type(learning_rate) in (list, np.ndarray):
            assert len(learning_rate) == n_epochs
            learning_rates = learning_rate

        learning_rates_β = None
        if type(learning_rate_β) in (list, np.ndarray):
            assert len(learning_rate_β) == n_epochs
            learning_rates_β = learning_rate_β

        if not hasattr(self, "callback_outputs"):
            self.callback_outputs = []

        for epoch in range(1, n_epochs + 1):
            start_time = time()

            # set the effective learning rates
            if learning_rates is not None:
                learning_rate = learning_rates[epoch - 1]
            if learning_rates_β is not None:
                learning_rate_β = learning_rates_β[epoch - 1]

            # loop over the mini-batches and compute/apply the gradients for each
            for mini_batch_indices in self._random_mini_batch_indices(mini_batch_size):
                V = self.X[mini_batch_indices]
                self._compute_positive_grads(V)
                self._compute_negative_grads(V.shape[0])
                self._apply_grads(learning_rate / V.shape[0])
                self._clip_weights_and_biases()

            # update β and clip the weights and biases into proper range
            samples = self.sample(n_samples)
            self._update_β(samples, learning_rate_β)

            # callback function
            if callback is not None:
                callback_output = callback(self.X, self._get_state_vectors(samples))
                self.callback_outputs.append(callback_output)

            # print diagnostics
            end_time = time()
            print(
                f"[{type(self).__name__}] epoch {epoch}:",
                f"learning rate = {learning_rate:.2e},",
                f"effective β = {self.β:.3f},",
                f"epoch duration = {(end_time - start_time) * 1e3:.3f}ms,",
                callback_output["print"] if callback is not None else "",
            )
