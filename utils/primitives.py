import numpy as np
import scipy.linalg as la

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def pauli_embed(P, qubit, n):
    if qubit == 0:
        return np.kron(P, np.eye(2**(n-1), dtype=complex))
    elif qubit == n - 1:
        return np.kron(np.eye(2**qubit, dtype=complex), P)
    else:
        return np.kron(np.kron(np.eye(2**qubit, dtype=complex), P), np.eye(2**(n-qubit-1), dtype=complex))

def hea(n, depth, params):
    U = np.eye(2**n, dtype=complex)
    idx = 0
    for d in range(depth):
        for q in range(n):
            theta = params[idx]
            Rz_q = pauli_embed(la.expm(-1j * theta / 2 * Z), q, n)
            U = Rz_q @ U
            idx += 1
        for q in range(n - 1):
            theta = params[idx]
            XX_q = pauli_embed(X, q, n) @ pauli_embed(X, q + 1, n)
            U_xx = la.expm(-1j * theta / 2 * XX_q)
            U = U_xx @ U
            idx += 1
    return U

def G_tilde(U_R, G):
    return U_R @ G @ U_R.conj().T

def proj_perp(A):
    return A - np.diag(np.diag(A))

def d_k(Gtilde):
    return np.linalg.norm(proj_perp(Gtilde), 'fro')

def commutator(A, B):
    return A @ B - B @ A

def commutator_frob2(A, B):
    comm = commutator(A, B)
    return float(np.real(np.trace(comm.conj().T @ comm)))

def var_exact(Gtilde, H):
    comm_norm_sq = commutator_frob2(Gtilde, H)
    d = Gtilde.shape[0]
    return comm_norm_sq / (d * (d + 1))

def var_measured_haar(Gtilde, H, n_trials=2000, rng_=None):
    if rng_ is None:
        rng_ = np.random.RandomState()
    d = Gtilde.shape[0]
    n = int(np.log2(d))
    B_k = 1j * commutator(Gtilde, H)
    grads = []
    for _ in range(n_trials):
        psi = rng_.randn(d) + 1j * rng_.randn(d)
        psi = psi / np.linalg.norm(psi)
        g = float(np.real(psi.conj() @ B_k @ psi))
        grads.append(g**2)
    return float(np.mean(grads))

def kron_n(ops):
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def rz_only(n, depth, params):
    U = np.eye(2**n, dtype=complex)
    idx = 0
    for d in range(depth):
        for q in range(n):
            theta = params[idx]
            Rz_q = pauli_embed(la.expm(-1j * theta / 2 * Z), q, n)
            U = Rz_q @ U
            idx += 1
    return U

def all_paulis(n):
    single = [(I2, 'I'), (X, 'X'), (Y, 'Y'), (Z, 'Z')]
    if n == 1:
        return single
    paulis_lower = all_paulis(n - 1)
    paulis = []
    for s_mat, s_name in single:
        for lower_mat, lower_name in paulis_lower:
            paulis.append((np.kron(s_mat, lower_mat), s_name + lower_name))
    return paulis

def anticommutes(A, B):
    return np.allclose(A @ B + B @ A, 0)

def build_H_from_paulis(pauli_mats, coeffs):
    H = np.zeros_like(pauli_mats[0])
    for P, c in zip(pauli_mats, coeffs):
        H += c * P
    return H
