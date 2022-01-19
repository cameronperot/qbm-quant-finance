import numpy as np
from scipy.sparse import csr_matrix, kron, identity, diags
from scipy.linalg import eigh


# set constants
sparse_X = csr_matrix(([1, 1], ([0, 1], [1, 0])), dtype=np.float64)
sparse_Z = csr_matrix(([1, -1], ([0, 1], [0, 1])), dtype=np.float64)


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


def compute_H(h, J, A, B, n_qubits, σ):
    """
    Computes the Hamiltonian of the annealer at relative time s.

    :param h: Linear Ising terms.
    :param J: Quadratic Ising terms.
    :param A: Coefficient of the off-diagonal terms, e.g. A(s).
    :param B: Coefficient of the diagonal terms, e.g. B(s).
    :param n_qubits: Number of qubits.
    :param σ: Kronecker product Pauli matrices dict.

    :returns: Hamiltonian matrix H.
    """
    # diagonal terms
    H_diag = np.zeros(2 ** n_qubits)
    for i in range(n_qubits):
        # linear terms
        if h[i] != 0:
            H_diag += (B * h[i]) * σ["z_diag", i]

        # quadratic terms
        for j in range(i + 1, n_qubits):
            if J[i, j] != 0:
                H_diag += (B * J[i, j]) * σ["zz_diag", i, j]

    # return just the diagonal if H is a diagonal matrix
    if A == 0:
        return np.diag(H_diag)

    # off-diagonal terms
    H = csr_matrix((2 ** n_qubits, 2 ** n_qubits), dtype=np.float64)
    for i in range(n_qubits):
        H -= A * σ["x", i]

    return (H + diags(H_diag, format="csr")).toarray()


def compute_ρ(H, β, diagonal=False):
    """
    Computes the trace normalized density matrix ρ.

    :param H: Hamiltonian matrix.
    :param β: Inverse temperature β = 1 / (k_B * T).
    :param diagonal: Flag to indicate whether H is a diagonal matrix or not.

    :return: Density matrix ρ.
    """
    # if diagonal then compute directly, else use eigen decomposition
    if diagonal:
        Λ = H.diagonal()
        exp_βΛ = np.exp(-β * (Λ - Λ.min()))
        return np.diag(exp_βΛ / exp_βΛ.sum())
    else:
        Λ, S = eigh(H)
        exp_βΛ = np.exp(-β * (Λ - Λ.min()))
        return (S * (exp_βΛ / exp_βΛ.sum())) @ S.T
