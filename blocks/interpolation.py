import numpy as np
from utils.primitives import hea, G_tilde, d_k, var_exact, pauli_embed, Z
from utils.hamiltonians import H_ghz
from utils.io import save_results

rng = np.random.RandomState(42)

n_III = 4
d_III = 2**n_III
G0_III = pauli_embed(Z/2, 0, n_III)
H_III = H_ghz(n_III)
N_III = 300

depths_III = list(range(1, 13))
results = {
    'block': 'Block III',
    'title': 'Intermediate Regime: d_k Interpolation',
    'description': 'd_k varies continuously between 0 and sqrt(d)/2 as depth increases',
    'data': []
}

for dep in depths_III:
    np_dep = n_III * 2 * dep
    dks = []
    vars_ = []
    
    for _ in range(N_III):
        p = rng.uniform(0, 2*np.pi, np_dep)
        U = hea(n_III, dep, p)
        Gt = G_tilde(U, G0_III)
        dks.append(float(d_k(Gt)))
        vars_.append(float(var_exact(Gt, H_III)))
    
    mdk = float(np.mean(dks))
    var_mean = float(np.mean(vars_))
    
    results['data'].append({
        'depth': dep,
        'd_k_mean': mdk,
        'd_k_over_sqrt_d': mdk / np.sqrt(d_III),
        'var_exact_mean': var_mean,
        'd_k_squared_over_d_d_plus_1': float((mdk**2) / (d_III * (d_III + 1)))
    })

save_results('interpolation', results)
