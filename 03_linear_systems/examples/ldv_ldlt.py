# LDV and LDLT Decompositions

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ldv_decomposition import LDV
from src.ldlt_decomposition import LDLT

def run_example():
    print("--- LDV and LDLT Decompositions Example ---")
    
    # 1. LDV Decomposition (A = LDV)
    # A generic square matrix
    A_gen = np.array([
        [2, 1, 1],
        [4, 3, 3],
        [8, 7, 9]
    ], dtype=float)
    
    print("\n--- 1. LDV Matrix Decomposition (A = LDV) ---")
    print(f"Original Matrix A:\n{A_gen}")
    
    L1, D1, V1 = LDV(A_gen)
    print(f"\nMatrix L (Unit Lower Triangular):\n{L1}")
    print(f"Matrix D (Diagonal Pivots):\n{D1}")
    print(f"Matrix V (Unit Upper Triangular):\n{V1}")
    
    # Verify LDV = A
    reconstruction1 = np.dot(np.dot(L1, D1), V1)
    print(f"\nVerification (L * D * V):\n{reconstruction1}")
    
    # 2. LDLT Decomposition (A = LDL^T)
    # Requires a symmetric positive definite matrix A
    B = np.array([[2, 1], [0, 4]], dtype=float)
    A_sym = np.dot(B.T, B) # Symmetric and SPD
    
    print("\n--- 2. LDLT Matrix Decomposition (A = LDL^T) ---")
    print(f"Original Symmetric Matrix A:\n{A_sym}")
    
    L2, D2 = LDLT(A_sym)
    print(f"\nMatrix L (Unit Lower Triangular):\n{L2}")
    print(f"Matrix D (Diagonal D^2 pivots):\n{D2}")
    
    # Verify LDLT = A
    reconstruction2 = np.dot(np.dot(L2, D2), L2.T)
    print(f"\nVerification (L * D * L^T):\n{reconstruction2}")

if __name__ == "__main__":
    run_example()
