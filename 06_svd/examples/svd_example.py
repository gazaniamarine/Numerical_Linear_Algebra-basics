import sys
import os
import numpy as np

# set sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

from svd_computation import compute_svd
from low_rank_approx import low_rank_approximation, approximation_error, energy_captured

def run_svd_example():
    print("--- SVD Example ---")
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    
    print("Original matrix A:")
    print(A)
    
    U, s, Vt = compute_svd(A)
    print("\nSingular values:")
    print(s)
    
    print("\nRank-1 Approximation:")
    A1 = low_rank_approximation(A, 1)
    print(A1)
    
    print(f"\nError (Frobenius): {approximation_error(A, 1):.4f}")
    print(f"Energy captured: {energy_captured(A, 1):.2%}")

if __name__ == "__main__":
    run_svd_example()
