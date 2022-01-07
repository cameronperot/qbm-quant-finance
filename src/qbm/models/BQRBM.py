import numpy as np
from dwave.system import DWaveSampler, FixedEmbeddingComposite
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
        anneal_params,
        anneal_schedule_data,
        β_initial=1.5,
        qpu_params={"region": "eu-central-1", "solver": "Advantage_system5.1"},
        seed=42,
        exact=False,
    ):
        """
        :param X: Training data.
        :param n_hidden: Number of hidden units.
        :param embedding: Embedding of the problem onto the QPU.
        :param anneal_params: Dictionary with the keys
            ["schedule", "s", "relative_chain_strength"]
            representing the annealing schedule (list of tuples), relative freeze time,
            A(s_freeze), B(s_freeze), and relative chain strength, respectively.
        :param β: Initial effective β of the D-Wave. Used for scaling the coefficients.
        :param qpu_params: Parameters dict to unpack for the qpu.
        :param seed: Seed for the random number generator (used for randomizing minibatches).
        :param exact: If True will use an exact QBM computation rather than the annealer.
        """
        # convert from binary to ±1 if necessary
        if set(np.unique(X)) == set([0, 1]):
            X = 2 * X - 1
        assert set(np.unique(X)) == set([-1, 1])
        self.X_binary = (self.X + 1) / 2

        super().__init__(X=X, n_hidden=n_hidden, seed=seed)
        self.embedding = embedding
        self.anneal_params = anneal_params
        self.β = β_initial
        self.βs = [β_initial]
        self.A = anneal_schedule_data.loc[s, "A(s) (GHz)"]
        self.B = anneal_schedule_data.loc[s, "B(s) (GHz)"]
        self.exact = exact

        self._initialize_sampler()

        if self.exact:
            from qbm.utils.exact import (
                compute_H,
                compute_ρ,
                sparse_X,
                sparse_Z,
                sparse_kron,
            )

            # set Kronecker product Pauli matrices
            self.σ = {}
            for i in range(self.n_qubits):
                self.σ["x", i] = sparse_kron(i, self.n_qubits, sparse_X)
                self.σ["z", i] = sparse_kron(i, self.n_qubits, sparse_Z)

            # initialize states and state vectors
            self.states = np.arange(2 ** self.n_qubits)
            self.state_vectors = 1 - 2 * np.vstack(
                [convert_bin_str_to_list(f"{x:{f'0{self.n_qubits}b'}}") for x in range(8)]
            )

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

    def _update_β(self):
        """
        Updates the effective β = 1 / kT. Used for scaling the coefficients sent to the
        annealer.
        """
        n_samples = 10000

        # sample state vectors
        if self.exact:
            state_vectors = self.sample_exact(n_samples)
        else:
            samples = self.sample(n_samples)
            state_vectors = samples.record.sample

        # compute the train energy
        V_train = self.X
        VW_train = V_train @ self.W
        H_train = expit(VW_train + self.b)
        E_train = self._compute_energies(V_train, H_train, VW_train)

        # compute the model energy
        V_model = state_vectors[:, : self.n_visible]
        VW_model = V_model @ self.W
        H_model = state_vectors[:, self.n_visible :]
        E_model = self._compute_energies(V_model, H_model, VW_model)

        # update the params
        self.β += self.learning_rate / self.β ** 2 * (E_train - E_model)
        self.βs.append(self.β)
        self.scaling_factor = self.β * self.B

    def _compute_positive_grads(self, V_pos):
        """
        Computes the gradients for the positive phase, i.e., the expectation values w.r.t.
        the clamped Hamiltonian.

        :param V_pos: Numpy array where the rows are data vectors which to clamp the Hamiltonian
            with.
        """
        Γ = self.β * self.anneal_params["A"]
        b_eff = self.b + V_pos @ self.W
        D = np.sqrt(Γ ** 2 + b_eff ** 2)
        H_pos = (b_eff / D) * np.tanh(D)

        self.grads["a_pos"] = np.mean(V_pos, axis=0)
        self.grads["b_pos"] = np.mean(H_pos, axis=0)
        self.grads["W_pos"] = V_pos.T @ H_pos / V_pos.shape[0]

    def _compute_negative_grads(self, n_samples):
        """
        Computes the gradients for the negative phase, i.e., the expectation values w.r.t.
        the model distribution.

        :param n_samples: Number of samples to obtain from the annealer.
        """
        if self.exact:
            state_vectors = self.sample_exact(n_samples)
        else:
            samples = self.sample(n_samples)
            state_vectors = samples.record.sample

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
        bias_range = self.h_range * self.scaling_factor * (1 - 1e-15)
        weight_range = self.J_range * self.scaling_factor * (1 - 1e-15)

        self.a = np.clip(self.a, *bias_range)
        self.b = np.clip(self.b, *bias_range)
        self.W = np.clip(self.W, *weight_range)

    def sample(self, n_samples, answer_mode="raw", use_gauge=True):
        """
        Obtain a sample set using the annealer.

        :param n_samples: Number of samples to generate (num_reads param in sample_ising).
        :param answer_mode: "raw" or "histogram".
        :param use_gauge: If True will use a random gauge transformation (recommended for
            more robust sample generation).

        :returns: Ocean SDK SampleSet object.
        """
        # compute the h's and J's
        h = np.concatenate((self.a, self.b)) / self.scaling_factor
        J = np.zeros((self.n_qubits, self.n_qubits)) / self.scaling_factor
        J[: self.n_visible, self.n_visible :] = self.W / self.scaling_factor

        # apply a random gauge
        if use_gauge:
            gauge = self.rng.choice([-1, 1], self.n_qubits)
            h *= gauge
            J *= np.outer(gauge, gauge)

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
            answer_mode=answer_mode,
            auto_scale=False,
        )

        # undo the gauge
        if use_gauge:
            samples.record.samples *= gauge

        return samples

    def sample_exact(self, n_samples):
        """
        Sample using the exact computed probabilities.

        :param n_samples: Number of samples to generate.

        :returns: Samples array of shape (n_samples, n_qubits).
        """
        # compute the h's and J's
        h = np.concatenate((self.a, self.b)) / self.scaling_factor
        J = np.zeros((self.n_qubits, self.n_qubits)) / self.scaling_factor
        J[: self.n_visible, self.n_visible :] = self.W / self.scaling_factor

        # compute the Hamiltonian and density matrix
        H = compute_H(h, J, self.A, self.B, self.n_qubits, self.σ)
        ρ = compute_ρ(H, self.T)

        # sample using the probabilities on the diagonal of ρ
        sample_indices = self.rng.choice(self.states, size=n_samples, p=np.diag(ρ))
        state_vectors = self.state_vectors[sample_indices]

        return state_vectors

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
