"""Verifying matrix multiplication operations against NumPy."""
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import multiplication

def test_mat_vec_multiply():
    A = [[1.0, 2.0], [3.0, 4.0]]
    x = [5.0, 6.0]
    ours = multiplication.mat_vec_multiply(A, x)
    theirs = np.dot(np.array(A), np.array(x)).tolist()
    
    print(f"  Mat-Vec Multiply: {A} * {x}")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.allclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_mat_mat_multiply():
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ours = multiplication.mat_mat_multiply(A, B)
    theirs = np.dot(np.array(A), np.array(B)).tolist()
    
    print(f"  Mat-Mat Multiply: A(2x3) * B(3x2)")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.allclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

if __name__ == "__main__":
    print("=" * 60)
    print("VERIFICATION: UNIT 02 — MATRIX MULTIPLICATION")
    print("=" * 60)
    test_mat_vec_multiply()
    test_mat_mat_multiply()
    print("ALL MATRIX MULTIPLICATION VERIFIED ✓")
