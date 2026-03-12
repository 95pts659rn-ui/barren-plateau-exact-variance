import numpy as np
from utils.primitives import pauli_embed, X, Z, I2

def H_ghz(n):
    proj_minus = (np.eye(2**n, dtype=complex) - np.kron(np.kron(Z, np.eye(2**(n-2), dtype=complex)), Z)) / 2
    return proj_minus

def H_ising_zz(n):
    H = np.zeros((2**n, 2**n), dtype=complex)
    for q in range(n - 1):
        ZZ_q = pauli_embed(Z, q, n) @ pauli_embed(Z, q + 1, n)
        H += ZZ_q / (n - 1)
    return H

def H_random_traceless(n, rng_=None):
    if rng_ is None:
        rng_ = np.random.RandomState()
    H = rng_.randn(2**n, 2**n) + 1j * rng_.randn(2**n, 2**n)
    H = (H + H.conj().T) / 2
    H = H - np.trace(H) * np.eye(2**n, dtype=complex) / (2**n)
    return H / np.linalg.norm(H, 'fro')
