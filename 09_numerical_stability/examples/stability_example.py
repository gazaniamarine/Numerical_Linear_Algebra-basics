import sys
import os
import numpy as np

# set sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

from conditioning import hilbert_matrix, perturbation_experiment

def run_stability_example():
    print("--- Numerical Stability Example ---")
    
    for n in [5, 10, 12]:
        print(f"\nExperiment with Hilbert Matrix size {n}x{n}:")
        A = hilbert_matrix(n)
        x_true = np.ones(n)
        
        results = perturbation_experiment(A, x_true, perturbation_scale=1e-10)
        
        print(f"  Condition number: {results['condition_number']:.2e}")
        print(f"  Relative RHS perturbation: {results['relative_rhs_change']:.2e}")
        print(f"  Relative solution change: {results['relative_solution_change']:.2e}")
        
    print("\nObservation: As the condition number exceeds 1/perturbation_scale,")
    print("the solution can become completely inaccurate.")

if __name__ == "__main__":
    run_stability_example()
