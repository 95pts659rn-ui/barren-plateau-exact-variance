import numpy as np
from utils.primitives import hea, G_tilde, var_exact, pauli_embed, Z, commutator_frob2
from utils.hamiltonians import H_ghz
from utils.io import save_results

rng = np.random.RandomState(42)

n_V = 4
d_V = 2**n_V
G0_V = pauli_embed(Z/2, 0, n_V)
H_base = H_ghz(n_V)

results = {
    'block': 'Block V',
    'title': 'Multi-Observable Hamiltonian',
    'description': 'Formula holds for H = ZZ + eps*XX (multi-term)',
    'data': []
}

epsilons = [0, 0.1, 0.5, 1.0, 2.0]
dep_V = 3
np_V = n_V * 2 * dep_V

for eps in epsilons:
    XX_term = np.zeros((d_V, d_V), dtype=complex)
    for q in range(n_V - 1):
        from utils.primitives import X
        XX_q = pauli_embed(X, q, n_V) @ pauli_embed(X, q+1, n_V)
        XX_term += XX_q / (n_V - 1)
    
    H_multi = H_base + eps * XX_term
    
    preds = []
    for _ in range(300):
        p = rng.uniform(0, 2*np.pi, np_V)
        U = hea(n_V, dep_V, p)
        Gt = G_tilde(U, G0_V)
        preds.append(float(var_exact(Gt, H_multi)))
    
    results['data'].append({
        'epsilon': float(eps),
        'H': 'ZZ + eps*XX',
        'var_mean': float(np.mean(preds)),
        'var_std': float(np.std(preds))
    })

save_results('multi_observable', results)
