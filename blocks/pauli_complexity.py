import numpy as np
from utils.primitives import hea, G_tilde, pauli_embed, Z, commutator_frob2, all_paulis, build_H_from_paulis
from utils.io import save_results

rng = np.random.RandomState(43)

n = 4
d = 2**n
G0n = pauli_embed(Z/2, 0, n)

all_paulis_n = all_paulis(n)
N_paulis = len(all_paulis_n)

N_terms_list = [1, 5, 10, 20, 50, 100]

results = {
    'block': 'Block IX',
    'title': 'Pauli Complexity Scaling',
    'description': 'Variance scaling with Pauli observable sparsity',
    'data': []
}

for N_terms in N_terms_list:
    vars_unit = []
    vars_norm = []
    
    for trial in range(200):
        n_layers = max(2, 2*n)
        n_params = n_layers * (2*n - 1)
        p = rng.uniform(0, 2*np.pi, n_params)
        U = hea(n, n_layers, p)
        Gt = G_tilde(U, G0n)
        
        pauli_indices = rng.choice(N_paulis, size=N_terms, replace=False)
        pauli_coeffs = rng.randn(N_terms)
        pauli_mats = [all_paulis_n[idx][0] for idx in pauli_indices]
        
        H_unit = build_H_from_paulis(pauli_mats, pauli_coeffs)
        
        comm_unit = commutator_frob2(Gt, H_unit)
        vars_unit.append(float(comm_unit) / (d*(d+1)))
        
        H_norm = H_unit / np.linalg.norm(H_unit, 'fro')
        comm_norm = commutator_frob2(Gt, H_norm)
        vars_norm.append(float(comm_norm) / (d*(d+1)))
    
    results['data'].append({
        'n': n,
        'N_terms': N_terms,
        'variance_unit_mean': float(np.mean(vars_unit)),
        'variance_unit_std': float(np.std(vars_unit)),
        'variance_normalized_mean': float(np.mean(vars_norm)),
        'variance_normalized_std': float(np.std(vars_norm)),
        'ratio_normalized_to_unit': float(np.mean(vars_norm)) / (float(np.mean(vars_unit)) + 1e-30)
    })

save_results('pauli_complexity', results)
