import sys
import os
import numpy as np

# set sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

from jacobi import jacobi_method
from gauss_seidel import gauss_seidel_method
from conjugate_gradient import conjugate_gradient

def run_iterative_example():
    print("--- Iterative Methods Example ---")
    # Diagonally dominant SPD matrix
    A = np.array([
        [10, -1, 2, 0],
        [-1, 11, -1, 3],
        [2, -1, 10, -1],
        [0, 3, -1, 8]
    ], dtype=float)
    b = np.array([6, 25, -11, 15], dtype=float)
    
    print("Solving Ax = b...")
    
    # 1. Jacobi
    x_jac, info_jac = jacobi_method(A, b, tol=1e-6)
    print(f"\nJacobi: converged={info_jac['converged']} in {info_jac['iterations']} iterations")
    print(f"Residual norm: {info_jac['residual_norm']:.2e}")
    
    # 2. Gauss-Seidel
    x_gs, info_gs = gauss_seidel_method(A, b, tol=1e-6)
    print(f"\nGauss-Seidel: converged={info_gs['converged']} in {info_gs['iterations']} iterations")
    print(f"Residual norm: {info_gs['residual_norm']:.2e}")
    
    # 3. Conjugate Gradient
    x_cg, info_cg = conjugate_gradient(A, b, tol=1e-6)
    print(f"\nConjugate Gradient: converged={info_cg['converged']} in {info_cg['iterations']} iterations")
    print(f"Residual norm: {info_cg['residual_norm']:.2e}")
    
    print("\nSolutions match (CG vs GS):", np.allclose(x_cg, x_gs, atol=1e-4))

if __name__ == "__main__":
    run_iterative_example()
