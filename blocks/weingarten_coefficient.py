import numpy as np
from utils.primitives import hea, G_tilde, pauli_embed, Z, commutator_frob2
from utils.hamiltonians import H_ghz
from utils.io import save_results

rng = np.random.RandomState(42)

n_VIII_vals = [3, 4, 5, 6]

results = {
    'block': 'Block VIII',
    'title': 'Weingarten Coefficient / 2-Design Formula',
    'description': 'E[||[G~_k, H]||_F^2] matches 2-design formula',
    'data': []
}

for n in n_VIII_vals:
    d = 2**n
    G0n = pauli_embed(Z/2, 0, n)
    H_n = H_ghz(n)
    
    comm_norms_sq = []
    for _ in range(500):
        p = rng.uniform(0, 2*np.pi, n*2*max(2, 2*n))
        U = hea(n, max(2, 2*n), p)
        Gt = G_tilde(U, G0n)
        comm_norm = commutator_frob2(Gt, H_n)
        comm_norms_sq.append(float(comm_norm))
    
    E_comm = float(np.mean(comm_norms_sq))
    
    TrG2 = float(np.real(np.trace(G0n @ G0n)))
    TrH2 = float(np.real(np.trace(H_n @ H_n)))
    TrG2H2 = float(np.real(np.trace(G0n @ G0n @ H_n @ H_n)))
    
    weingarten = 2.0 / (d**2 - 1) * (TrG2 * TrH2 - TrG2H2)
    ratio_w = E_comm / (weingarten + 1e-30)
    
    results['data'].append({
        'n': n,
        'dim': d,
        'E_commutator_frob2': E_comm,
        'weingarten_formula': weingarten,
        'ratio': ratio_w,
        'Tr_G_squared': TrG2,
        'Tr_H_squared': TrH2
    })

save_results('weingarten_coefficient', results)
