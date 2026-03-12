import numpy as np
from utils.primitives import hea, G_tilde, var_exact, pauli_embed, Z
from utils.hamiltonians import H_ghz, H_ising_zz, H_random_traceless
from utils.io import save_results

rng = np.random.RandomState(42)
rng_b = np.random.RandomState(43)

N_IV = 400
n_IV_boot = 500
H_fns = {
    'GHZ': H_ghz,
    'Ising_ZZ': H_ising_zz,
    'Random_traceless': H_random_traceless
}

results = {
    'block': 'Block IV',
    'title': 'Extended Validation',
    'description': 'Var scaling across n=2..9 with bootstrap confidence intervals',
    'data': []
}

for n in [2, 3, 4, 5, 6, 7, 8, 9]:
    d = 2**n
    if d > 512:
        continue
    
    G0n = pauli_embed(Z/2, 0, n)
    dep = max(2, 2*n)
    np_n = n * 2 * dep
    n_circuits_iv = N_IV if n <= 6 else (200 if n <= 8 else 100)
    
    circuits_n = []
    for _ in range(n_circuits_iv):
        p = rng.uniform(0, 2*np.pi, np_n)
        U = hea(n, dep, p)
        Gt = G_tilde(U, G0n)
        circuits_n.append(Gt)
    
    for h_name, h_fn in H_fns.items():
        H = h_fn(n)
        preds = np.array([var_exact(Gt, H) for Gt in circuits_n])
        pred_v = float(np.mean(preds))
        
        boot_pred = []
        for _ in range(n_IV_boot):
            idx = rng_b.randint(0, n_circuits_iv, n_circuits_iv)
            boot_pred.append(float(np.mean(preds[idx])))
        pred_lo, pred_hi = np.percentile(boot_pred, [2.5, 97.5])
        
        results['data'].append({
            'n': n,
            'dim': d,
            'hamiltonian': h_name,
            'pred_var': pred_v,
            'ci_lower': float(pred_lo),
            'ci_upper': float(pred_hi),
            'log2_var_over_n': float(np.log2(pred_v + 1e-30) / n)
        })

save_results('extended_validation', results)
