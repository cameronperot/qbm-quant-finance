from time import time

import numpy as np
import pandas as pd
import torch
from dwave.system import DWaveSampler
from scipy.sparse import csr_matrix, kron, identity
from scipy.linalg import expm

from qbm.utils import get_project_dir, get_rng, load_artifact, save_artifact

project_dir = get_project_dir()

anneal_schedule_data = pd.read_csv(
    project_dir
    / "data/anneal_schedules/csv/09-1265A-A_Advantage_system5_1_annealing_schedule.csv",
    index_col="s",
)
if 0.5 not in anneal_schedule_data.index:
    anneal_schedule_data.loc[0.5] = (
        anneal_schedule_data.loc[0.499] + anneal_schedule_data.loc[0.501]
    ) / 2

k_B = 20.83661912  # GHz/K
X = csr_matrix(([1, 1], ([0, 1], [1, 0])))
Z = csr_matrix(([1, -1], ([0, 1], [0, 1])))


def sparse_σ(i, n_qubits, A):
    """
    Compute I_{i-1} ⊗ A ⊗ I_{n_qubits-i}.

    :param i: Index of the "A" matrix.
    :param n_qubits: Total number of qubits.
    :param A: Matrix to tensor with identities.

    :returns: σ_i^x
    """
    if i != 0 and i != n_qubits - 1:
        return kron(kron(identity(2 ** i), A), identity(2 ** (n_qubits - i - 1)))
    if i == 0:
        return kron(A, identity(2 ** (n_qubits - 1)))
    if i == n_qubits - 1:
        return kron(identity(2 ** (n_qubits - 1)), A)


def compute_H(h, J, s, n_visible, n_hidden):
    """
    Computes the Hamiltonian of the annealer at the freeze-out
    point s*.

    :param h: Linear Ising terms.
    :param J: Quadratic Ising terms.
    :param s: Where in the anneal schedule to compute H.
    :param n_visible: Number of visible units.
    :param n_hidden: Number of hidden units.

    :returns: Hamiltonian matrix H.
    """
    A = anneal_schedule_data.loc[s, "A(s) (GHz)"]
    B = anneal_schedule_data.loc[s, "B(s) (GHz)"]
    n_qubits = n_visible + n_hidden
    H = csr_matrix((2 ** n_qubits, 2 ** n_qubits), dtype=np.float64)

    # off-diagonal terms
    for i in range(n_qubits):
        H -= A * sparse_σ(i, n_qubits, X)
    # linear terms
    for i in range(n_qubits):
        H -= B * h[i] * sparse_σ(i, n_qubits, Z)
    # quadratic terms
    for i in range(n_visible):
        for j in range(n_visible, n_qubits):
            H -= B * J[i, j] * sparse_σ(i, n_qubits, Z) * sparse_σ(j, n_qubits, Z)

    return H.toarray()


def compute_ρ(H, T, matrix_exp="torch"):
    """
    Computes the trace normalized density matrix ρ.

    :param H: Hamiltonian matrix (in GHz units)
    :param T: Temperature.
    :param matrix_exp: If "torch" will use torch's matrix exp, if "scipy" will use scipy's.

    :return: Density matrix ρ.
    """
    β = 1 / (k_B * T)

    if matrix_exp == "torch":
        ρ = torch.matrix_exp(-β * torch.from_numpy(H)).numpy()
    if matrix_exp == "scipy":
        ρ = expm(-β * H)

    return ρ / ρ.trace()


if __name__ == "__main__":
    config_id = 1

    project_dir = get_project_dir()
    config_dir = project_dir / f"artifacts/exact_analysis/{config_id:02}"

    config = load_artifact(config_dir / "config.json")
    n_visible = config["n_visible"]
    n_hidden = config["n_hidden"]
    n_qubits = config["n_qubits"]

    qpu = DWaveSampler(**config["qpu_params"])

    if (config_dir / "h.pkl").exists() and (config_dir / "J.pkl").exists():
        print(f"Loading h's and J's at {config_dir}")
        h = load_artifact(config_dir / "h.pkl")
        J = load_artifact(config_dir / "J.pkl")
    else:
        μ = config["mu"]
        σ = config["sigma"]
        rng = get_rng(config["seed"])
        a = rng.normal(μ, σ, n_visible)
        b = rng.normal(μ, σ, n_hidden)
        W = rng.normal(μ, σ, (n_visible, n_hidden))

        h = np.concatenate((a, b))
        J = np.zeros((n_qubits, n_qubits))
        J[:n_visible, n_visible:] = W

        h = np.clip(h, *qpu.properties["h_range"])
        J = np.clip(J, *qpu.properties["j_range"])

        save_artifact(h, config_dir / "h.pkl")
        save_artifact(J, config_dir / "J.pkl")

    T_values = np.round(np.arange(2e-3, 51e-3, 2e-3), 3)  # mK
    s_values = np.round(np.arange(0.2, 1.01, 0.01), 2)
    errors = {}
    data = {}
    for s in s_values:
        s = round(s, 3)
        for T in T_values:
            t = time()
            print(f"s = {s}, T = {T*1e3:.0f}mK")
            try:
                H = compute_H(h, J, s, n_visible, n_hidden)
                ρ = compute_ρ(H, T)
                data[(s, T)] = {"E": np.diag(H).copy(), "p": np.diag(ρ).copy()}
                print(f"    Completed in {time() - t:.3f}s")
            except Exception as error:
                errors[(s, T)] = error
                print("    Error")

    save_artifact(data, config_dir / "energies_probabilities.pkl")
    if errors:
        save_artifact(errors, config_dir / "errors.pkl")
