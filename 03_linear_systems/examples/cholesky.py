# Cholesky Decomposition

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.cholesky_decomposition import cholesky, solve_system_cholesky

def run_example():
    print("--- Cholesky Decomposition Example ---")
    
    # Square Symmetric Positive Definite Matrix A
    B = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]], dtype=float)
    A = np.dot(B.T, B)
    
    # 1. Perform Cholesky factor derivation
    L = cholesky(A)
    
    print(f"Matrix A (SPD):\n{A}")
    print(f"\nCholesky factor L:\n{L}")
    
    # 2. Verify A = LL^T
    reconstruction = np.dot(L, L.T)
    print(f"\nReconstruction (L * L^T):\n{reconstruction}")
    
    # 3. Solve a linear system Ax = b
    b = np.array([1, 2, 3], dtype=float)
    print(f"\nConstant Vector b: {b}")
    
    x = solve_system_cholesky(A, b)
    print(f"Solution x: {x}")
    
    # 4. Final verification with NumPy
    x_np = np.linalg.solve(A, b)
    print(f"NumPy solve x: {x_np}")
    
if __name__ == "__main__":
    run_example()
