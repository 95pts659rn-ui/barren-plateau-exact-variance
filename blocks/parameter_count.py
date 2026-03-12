import numpy as np
from utils.primitives import hea, G_tilde, var_exact, d_k, pauli_embed, Z
from utils.hamiltonians import H_ghz
from utils.io import save_results

rng = np.random.RandomState(42)

n_VI = 5
d_VI = 2**n_VI
G0_VI = pauli_embed(Z/2, 0, n_VI)
H_VI = H_ghz(n_VI)
N_VI = 500
depths_VI = [1, 2, 3, 4, 6, 8, 10]

results = {
    'block': 'Block VI',
    'title': 'Parameter-Count Sweep',
    'description': 'Var is independent of total parameter count p = 2*n*depth',
    'data': []
}

for dep in depths_VI:
    np_dep = n_VI * 2 * dep
    preds = []
    dks = []
    
    for _ in range(N_VI):
        p = rng.uniform(0, 2*np.pi, np_dep)
        U = hea(n_VI, dep, p)
        Gt = G_tilde(U, G0_VI)
        preds.append(float(var_exact(Gt, H_VI)))
        dks.append(float(d_k(Gt)))
    
    results['data'].append({
        'depth': dep,
        'param_count': np_dep,
        'var_mean': float(np.mean(preds)),
        'd_k_mean': float(np.mean(dks))
    })

save_results('parameter_count', results)
