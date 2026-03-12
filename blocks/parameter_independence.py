import numpy as np
from utils.primitives import (hea, rz_only, G_tilde, d_k, pauli_embed, Z, X, I2,
                              commutator, var_exact)
from utils.hamiltonians import H_ghz
from utils.io import save_results

rng = np.random.RandomState(42)

n_II = 3
d_II = 2**n_II
G0_II = pauli_embed(Z/2, 0, n_II)
H_II = H_ghz(n_II)
N_II = 300
np_rz = n_II * 4

results = {
    'block': 'Block II',
    'title': 'L is Parameter-Free, Basis-Fixed',
    'description': 'L = {Z_0, ..., Z_{n-1}} is fixed by measurement basis, not by circuit',
    'parts': []
}

part_a = {
    'name': 'Part A: ECBP d_k = 0 on dense grid',
    'data': []
}

thetas_grid = np.linspace(0, 2*np.pi, 100)
for theta_test in thetas_grid[::10]:
    p_grid = np.full(np_rz, theta_test)
    U_ecbp = rz_only(n_II, 4, p_grid)
    Gt_ecbp = G_tilde(U_ecbp, G0_II)
    d_k_val = d_k(Gt_ecbp)
    part_a['data'].append({'theta': float(theta_test), 'd_k': float(d_k_val)})

results['parts'].append(part_a)

part_b = {
    'name': 'Part B: Wrong basis L_prime = {X_q}',
    'data': []
}

dks_z = []
dks_x = []

for _ in range(N_II):
    p = rng.uniform(0, 2*np.pi, n_II * 4)
    U_ecbp = rz_only(n_II, 4, p)
    Gt_ecbp = G_tilde(U_ecbp, G0_II)
    
    d_k_z_val = d_k(Gt_ecbp)
    dks_z.append(float(d_k_z_val))
    
    A_rot = np.zeros_like(Gt_ecbp)
    for q in range(n_II):
        X_q = pauli_embed(X, q, n_II)
        A_q = X_q @ Gt_ecbp @ X_q
        A_rot = A_rot + A_q
    A_rot = A_rot - np.diag(np.diag(A_rot))
    d_k_x_val = np.linalg.norm(A_rot, 'fro')
    dks_x.append(float(d_k_x_val))

part_b['data'] = {
    'L_correct_d_k_mean': float(np.mean(dks_z)),
    'L_wrong_d_k_mean': float(np.mean(dks_x)),
    'separation': float(np.mean(dks_x) - np.mean(dks_z))
}

results['parts'].append(part_b)

part_c = {
    'name': 'Part C: Random theta vs near fixed theta',
    'data': []
}

Var_rand = []
Var_fixed = []

for _ in range(100):
    theta_base = rng.uniform(0, 2*np.pi, np_rz)
    p_rand = rng.uniform(0, 2*np.pi, n_II * 2 * 4)
    U_rand = hea(n_II, 4, p_rand)
    Gt_rand = G_tilde(U_rand, G0_II)
    Var_rand.append(float(var_exact(Gt_rand, H_II)))
    
    p_near = theta_base + rng.normal(0, 0.01, np_rz)
    U_near = rz_only(n_II, 4, p_near)
    Gt_near = G_tilde(U_near, G0_II)
    Var_fixed.append(float(var_exact(Gt_near, H_II)))

part_c['data'] = {
    'Var_random_init_mean': float(np.mean(Var_rand)),
    'Var_near_fixed_mean': float(np.mean(Var_fixed)),
    'ratio': float(np.mean(Var_fixed) / (np.mean(Var_rand) + 1e-30))
}

results['parts'].append(part_c)

save_results('parameter_independence', results)
