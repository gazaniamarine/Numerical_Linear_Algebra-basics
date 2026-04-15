# Least Squares — QR Solver Example
import numpy as np
import sys
import os

# Custom Module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qr import solve_least_squares_qr

def run_example():
    print("--- Least Squares via QR Factorization (Gram-Schmidt) ---")
    
    # 1. Setup an overdetermined system Ax = b
    # Target points: (1, 1), (2, 2.1), (3, 2.9)
    A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    b = np.array([1, 2.1, 2.9], dtype=float)
    
    print(f"Matrix A (Overdetermined):\n{A}")
    print(f"Vector b: {b}")
    
    # 2. Solve using QR Solver
    x, residual = solve_least_squares_qr(A, b)
    
    print("\n--- Fit Results ---")
    print(f"Solution x: {x}")
    print(f"Intercept:  {x[0]:.6f}")
    print(f"Slope:      {x[1]:.6f}")
    print(f"Residual Norm: {residual:.6f}")
    
    # 3. Compare with NumPy
    x_np, res_np, _, _ = np.linalg.lstsq(A, b, rcond=None)
    print(f"\nNumPy's Solution x: {x_np}")
    
if __name__ == "__main__":
    run_example()
