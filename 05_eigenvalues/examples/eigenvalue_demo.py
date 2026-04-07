import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.power_method import power_iteration, inverse_iteration
from src.qr_algorithm import qr_algorithm

def run_demo():
    print("--- Eigenvalue Algorithms Demo ---")
    
    # Create an symmetric positive-definite matrix for nice real eigenvalues
    A = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 2.0],
        [1.0, 2.0, 5.0]
    ])
    print("\nMatrix A:")
    print(A)
    
    # Ground truth (from numpy)
    true_evals, _ = np.linalg.eig(A)
    print("\n[Ground Truth] Eigenvalues (from np.linalg.eig):")
    print(np.sort(true_evals)[::-1])
    
    # 1. Power Method (Finds largest eigenvalue)
    lambda_dom, v_dom = power_iteration(A)
    print(f"\n1. [Power Method] Dominant Eigenvalue: {lambda_dom:.6f}")
    
    # 2. Inverse Iteration (Finds eigenvalue closest to shift)
    # Let's shift near the smallest true eigenvalue (~1.5)
    shift = 1.0
    lambda_inv, v_inv = inverse_iteration(A, mu=shift)
    print(f"\n2. [Inverse Iteration] Eigenvalue closest to shift={shift}: {lambda_inv:.6f}")
    
    # 3. QR Algorithm (Finds all eigenvalues)
    all_evals = qr_algorithm(A, num_simulations=150)
    print(f"\n3. [QR Algorithm] All Eigenvalues:")
    print(np.sort(all_evals)[::-1])

if __name__ == "__main__":
    np.random.seed(42)
    run_demo()
