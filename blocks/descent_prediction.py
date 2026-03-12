import numpy as np
import scipy.linalg as la
from utils.primitives import (hea, rz_only, G_tilde, d_k, var_exact, pauli_embed, 
                              Z, X, I2, kron_n, commutator)
from utils.hamiltonians import H_ghz
from utils.io import save_results

rng = np.random.RandomState(42)

n_VII = 5
d_VII = 2**n_VII
H_VII = H_ghz(n_VII)
N_VII = 500
G07 = pauli_embed(Z/2, 0, n_VII)
dep_VII = max(2, 2*n_VII)
np_VII = n_VII * 2 * dep_VII
XXn7 = kron_n([X, X] + [I2]*(n_VII - 2))

results = {
    'block': 'Block VII',
    'title': 'A Priori DESCENT Prediction',
    'description': 'd_k computed from circuit type before running experiment',
    'scenarios': []
}

scenarios_list = [
    ('ECBP', False, 'ECBP_no_descent'),
    ('ECBP', True, 'ECBP_with_descent'),
    ('SBP', False, 'SBP_deep_HEA'),
]

for ctype, desc, label in scenarios_list:
    if ctype == 'ECBP':
        pred_dk = 0.0
        reason = 'Rz_only: G~_k diagonal, d_k = 0'
        if desc:
            XXn = kron_n([X, X] + [I2]*(n_VII - 2))
            sample_dk = []
            for _ in range(200):
                alpha = rng.uniform(0.05, np.pi/4)
                U_xx = la.expm(1j * alpha * XXn)
                Gt = G_tilde(U_xx, G07)
                sample_dk.append(float(d_k(Gt)))
            pred_dk = float(np.mean(sample_dk))
            reason = f'Rz + XX_descent: d_k ~ {pred_dk:.3f}'
    else:
        d_k_expect = float(np.linalg.norm(G07, 'fro')) * np.sqrt(1 - 1/d_VII)
        pred_dk = d_k_expect
        reason = f'Haar limit: d_k ~ {pred_dk:.3f}'
    
    pred_var = pred_dk**2 / (d_VII * (d_VII + 1)) if pred_dk > 0 else 0.0
    
    meas_dks = []
    meas_grads = []
    psi0_VII = np.zeros(d_VII, dtype=complex)
    psi0_VII[0] = 1.0
    
    for _ in range(N_VII):
        if ctype == 'ECBP':
            p = rng.uniform(0, 2*np.pi, n_VII * 4)
            from utils.primitives import rz_only
            Ue = rz_only(n_VII, 4, p)
            if desc:
                alpha = rng.uniform(0.05, np.pi/4)
                Ue = la.expm(1j * alpha * XXn7) @ Ue
        else:
            p = rng.uniform(0, 2*np.pi, np_VII)
            Ue = hea(n_VII, dep_VII, p)
        
        Gt = G_tilde(Ue, G07)
        meas_dks.append(float(d_k(Gt)))
        B_k = 1j * commutator(Gt, H_VII)
        g = float(np.real(psi0_VII.conj() @ B_k @ psi0_VII))
        meas_grads.append(g**2)
    
    meas_dk = float(np.mean(meas_dks))
    meas_var = float(np.mean(meas_grads))
    
    results['scenarios'].append({
        'scenario': label,
        'circuit_type': ctype,
        'descent': desc,
        'pred_d_k': pred_dk,
        'pred_var': pred_var,
        'meas_d_k': meas_dk,
        'meas_var': meas_var,
        'reasoning': reason
    })

save_results('descent_prediction', results)
