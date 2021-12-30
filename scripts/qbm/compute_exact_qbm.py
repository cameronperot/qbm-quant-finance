import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, kron, identity
from scipy.linalg import expm
from tqdm import tqdm

from qbm.utils import get_project_dir, load_artifact, save_artifact

project_dir = get_project_dir()

# load the anneal schedule data
anneal_schedule_data = pd.read_csv(
    project_dir
    / "data/anneal_schedules/csv/09-1265A-A_Advantage_system5_1_annealing_schedule.csv",
    index_col="s",
)
# for some reason 0.5 is missing for Advantage_system5.1 so we need to interpolate
if 0.5 not in anneal_schedule_data.index:
    anneal_schedule_data.loc[0.5] = (
        anneal_schedule_data.loc[0.499] + anneal_schedule_data.loc[0.501]
    ) / 2

# set global constants
k_B = 20.83661912  # [GHz/K]
X = csr_matrix(([1, 1], ([0, 1], [1, 0])), dtype=np.float64)
Z = csr_matrix(([1, -1], ([0, 1], [0, 1])), dtype=np.float64)


def sparse_kron(i, n_qubits, A):
    """
    Compute I_{2^i} ⊗ A ⊗ I_{2^(n_qubits-i-1)}.

    :param i: Index of the "A" matrix.
    :param n_qubits: Total number of qubits.
    :param A: Matrix to tensor with identities.

    :returns: I_{2^i} ⊗ A ⊗ I_{2^(n_qubits-i-1)}.
    """
    if i != 0 and i != n_qubits - 1:
        return kron(kron(identity(2 ** i), A), identity(2 ** (n_qubits - i - 1)))
    if i == 0:
        return kron(A, identity(2 ** (n_qubits - 1)))
    if i == n_qubits - 1:
        return kron(identity(2 ** (n_qubits - 1)), A)


def compute_H(h, J, s, n_qubits, σ):
    """
    Computes the Hamiltonian of the annealer at relative time s.

    :param h: Linear Ising terms.
    :param J: Quadratic Ising terms.
    :param s: Where in the anneal schedule to compute H.
    :param n_qubits: Number of qubits.
    :param σ: Kronecker product Pauli matrices dict.

    :returns: Hamiltonian matrix H.
    """
    A = anneal_schedule_data.loc[s, "A(s) (GHz)"]
    B = anneal_schedule_data.loc[s, "B(s) (GHz)"]
    H = csr_matrix((2 ** n_qubits, 2 ** n_qubits), dtype=np.float64)

    for i in range(n_qubits):
        # off-diagonal terms
        H -= A * σ["x", i]

        # linear terms
        if h[i] != 0:
            H += (B * h[i]) * σ["z", i]

        # quadratic terms
        for j in range(i + 1, n_qubits):
            if J[i, j] != 0:
                H += (B * J[i, j]) * (σ["z", i] @ σ["z", j])

    return H.toarray()


def compute_ρ(H, T, matrix_exp="torch"):
    """
    Computes the trace normalized density matrix ρ.

    Note: torch's matrix exp is faster than scipy's even when accounting for conversions.

    :param H: Hamiltonian matrix.
    :param T: Temperature.
    :param matrix_exp: If "torch" will use torch's matrix exp, if "scipy" will use scipy's.

    :return: Density matrix ρ.
    """
    β = 1 / (k_B * T)

    if matrix_exp == "torch":
        exp_βH = torch.matrix_exp(-β * torch.from_numpy(H)).numpy()
    elif matrix_exp == "scipy":
        exp_βH = expm(-β * H)

    return exp_βH / exp_βH.trace()


if __name__ == "__main__":
    # compute exact data for all specified configs
    config_ids = (1, 2, 3, 4)

    # set s and T values
    T_values = np.round(np.arange(2e-3, 52e-3, 2e-3), 3)  # [K]
    s_values = np.round(np.arange(0.2, 1.01, 0.01), 2)

    # configure tqdm bars
    config_bar = tqdm(range(len(config_ids)), desc="configs")
    s_bar = tqdm(range(len(s_values)), desc="s values")
    T_bar = tqdm(range(len(T_values)), desc="T values")

    for config_id in config_ids:
        # load the config
        config_dir = project_dir / f"artifacts/exact_analysis/{config_id:02}"
        config = load_artifact(config_dir / "config.json")
        n_qubits = config["n_qubits"]

        # create Kronecker σ matrices
        σ = {}
        for i in range(n_qubits):
            σ["x", i] = sparse_kron(i, n_qubits, X)
            σ["z", i] = sparse_kron(i, n_qubits, Z)

        # load h's and J's
        h = load_artifact(config_dir / "h.pkl")
        J = load_artifact(config_dir / "J.pkl")

        # compute the exact E and p for all s and T values
        data = {}
        errors = {}
        for s in s_values:
            for T in T_values:
                try:
                    H = compute_H(h, J, s, n_qubits, σ)
                    ρ = compute_ρ(H, T)
                    data[s, T] = {"E": np.diag(H).copy(), "p": np.diag(ρ).copy()}
                except Exception as error:
                    errors[s, T] = error

                T_bar.update(1)

            T_bar.reset()
            s_bar.update(1)

        # save the exact data and errors (if any)
        save_artifact(data, config_dir / "exact_data.pkl")
        if errors:
            save_artifact(errors, config_dir / "errors.pkl")

        s_bar.reset()
        config_bar.update(1)
