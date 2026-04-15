# LU Decomposition

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lu_decomposition import LU, solve_system_lu

def run_example():
    print("--- LU Decomposition Example ---")
    
    # Square matrix A
    A = np.array([
        [2, 1, 1],
        [4, 3, 3],
        [8, 7, 9]
    ], dtype=float)
    
    print(f"Matrix A:\n{A}")
    
    # 1. Perform LU decomposition
    L, U = LU(A)
    
    print(f"\nMatrix L (Lower Triangular):\n{L}")
    print(f"Matrix U (Upper Triangular):\n{U}")
    
    # 2. Verify A = LU
    reconstruction = np.dot(L, U)
    print(f"\nReconstruction (L * U):\n{reconstruction}")
    
    # 3. Solve a linear system Ax = b
    b = np.array([4, 10, 24], dtype=float)
    print(f"\nConstant Vector b: {b}")
    
    x = solve_system_lu(A, b)
    print(f"Solution x: {x}")
    
    # 4. Final verification with NumPy
    x_np = np.linalg.solve(A, b)
    print(f"NumPy solve x: {x_np}")
    
if __name__ == "__main__":
    run_example()
