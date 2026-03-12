import json
import os
from pathlib import Path

def save_results(block_name, data):
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / f'{block_name}.json'
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    return str(filepath)

def load_results(block_name):
    results_dir = Path(__file__).parent.parent / 'results'
    filepath = results_dir / f'{block_name}.json'
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def format_latex_string(s):
    return s.replace('∂', '\\partial').replace('θ', '\\theta').replace('G̃', '\\tilde{G}').replace('ψ', '\\psi')
