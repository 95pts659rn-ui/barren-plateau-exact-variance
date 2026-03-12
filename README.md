# Exact Gradient Variance Identity: Barren Plateau Classification

Code repository for:

> Pritom Sarker (2026). **"Exact gradient variance identity and operator-space classification of barren plateaus."** *Physical Review A* (submitted).

## Overview

This repository contains the numerical experiments that validate the central result of the paper:

$$\mathrm{Var}\!\left[\partial_k C\right] = \frac{\left\|[\tilde{G}_k,\, H]\right\|_F^2}{d(d+1)}$$

where $\tilde{G}_k = U_R G_k U_R^\dagger$ is the rotated generator, $H$ is the observable, and $d = 2^n$.

## Repository Structure

```
├── blocks/                        # One script per validation block
│   ├── exact_formula.py           # Block I:   formula exactness
│   ├── parameter_independence.py  # Block II:  ECBP algebraic zeros
│   ├── interpolation.py           # Block III: depth interpolation
│   ├── extended_validation.py     # Block IV:  bootstrap scaling n=2..9
│   ├── multi_observable.py        # Block V:   multi-term Hamiltonian
│   ├── parameter_count.py         # Block VI:  parameter-count sweep
│   ├── descent_prediction.py      # Block VII: gate-type diagnostic
│   ├── weingarten_coefficient.py  # Block VIII: Weingarten coefficient
│   └── pauli_complexity.py        # Block IX:  Pauli complexity linearity
├── utils/
│   ├── primitives.py              # HEA, G_tilde, d_k, commutator norms, ...
│   ├── hamiltonians.py            # H_ghz, H_ising_zz, H_random_traceless
│   └── io.py                      # JSON result I/O
├── results/                       # Pre-computed JSON outputs (one per block)
│   ├── exact_formula.json
│   ├── parameter_independence.json
│   ├── interpolation.json
│   ├── extended_validation.json
│   ├── multi_observable.json
│   ├── parameter_count.json
│   ├── descent_prediction.json
│   ├── weingarten_coefficient.json
│   └── pauli_complexity.json
├── figures/
│   └── pauli_complexity_scaling_results.png
├── requirements.txt
└── LICENSE
```

## Reproducing the Results

```bash
pip install -r requirements.txt

# Run any block from the repo root (utils/ must be importable):
PYTHONPATH=. python blocks/exact_formula.py
PYTHONPATH=. python blocks/extended_validation.py   # slow: n=2..9, 400 circuits
PYTHONPATH=. python blocks/pauli_complexity.py

# Or run all blocks sequentially:
for f in blocks/*.py; do PYTHONPATH=. python "$f"; done
```

Each script writes its output to `results/<block_name>.json`. Pre-computed results are included in the repository.

## Key Results

| Block | Experiment | Result |
|-------|-----------|--------|
| I     | Formula exactness         | ratio $\in [0.991, 1.003]$, $n=2{-}9$, 3 $H$ types |
| II    | ECBP algebraic zeros      | $d_k = 0$ exactly at 8000 parameter points |
| III   | Depth interpolation       | $d_k$ interpolates $0 \to \sqrt{d}/2$ continuously |
| IV    | Bootstrap CIs             | All predictions within 95% CI, $\alpha = -0.98 \pm 0.03$ |
| V     | Multi-term Hamiltonian    | Formula tracks ε-dependent Var continuously |
| VI    | Parameter-count sweep     | No dependence on circuit parameter count |
| VII   | Gate-type diagnostic      | A priori prediction ≥ 96.2% accurate |
| VIII  | Weingarten coefficient    | Matches $d(d+1)$ to 8 significant figures |
| IX    | Pauli complexity scaling  | $\mathrm{Var} \propto N_{\mathrm{terms}}$; ratio $= 0.501 \pm 0.01$ |

## Requirements

- Python 3.8+
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib (for Figure 2)

## License

MIT License — see [LICENSE](LICENSE).
