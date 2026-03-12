import numpy as np
from utils.primitives import hea, G_tilde, var_exact, var_measured_haar, pauli_embed, Z
from utils.hamiltonians import H_ghz, H_ising_zz, H_random_traceless
from utils.io import save_results

rng = np.random.RandomState(42)

N_exact = 1000
n_values = [2, 3, 4, 5]
H_types = {'GHZ': H_ghz, 'Ising_ZZ': H_ising_zz, 'Random_traceless': H_random_traceless}

results = {
    'block': 'Block I',
    'title': 'Exact Formula Verification',
    'description': 'Var[dC/dtheta_k] = ||[G~_k, H]||_F^2 / (dim(dim+1))',
    'data': []
}

for n in n_values:
    d = 2**n
    G0 = pauli_embed(Z/2, 0, n)
    
    for h_name, h_fn in H_types.items():
        H = h_fn(n)
        preds = []
        meas = []
        
        for _ in range(50):
            params_r = rng.uniform(0, 2*np.pi, n*2*max(2, 2*n))
            U_r = hea(n, max(2, 2*n), params_r)
            Gt = G_tilde(U_r, G0)
            
            pred = var_exact(Gt, H)
            meas_v = var_measured_haar(Gt, H, n_trials=N_exact, rng_=rng)
            preds.append(float(pred))
            meas.append(float(meas_v))
        
        mp = float(np.mean(preds))
        mm = float(np.mean(meas))
        ratio = mm / (mp + 1e-30)
        
        results['data'].append({
            'n': n,
            'dim': d,
            'hamiltonian': h_name,
            'pred_var_mean': mp,
            'meas_var_mean': mm,
            'ratio': ratio,
            'error_percent': abs(ratio - 1) * 100
        })

save_results('exact_formula', results)
