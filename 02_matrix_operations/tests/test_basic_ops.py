"""Verifying basic matrix operations against NumPy."""
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import basic_ops

def test_add():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    ours = basic_ops.add(A, B)
    theirs = (np.array(A) + np.array(B)).tolist()
    
    print(f"  Add: {A} + {B}")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.allclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_subtract():
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    ours = basic_ops.subtract(A, B)
    theirs = (np.array(A) - np.array(B)).tolist()
    
    print(f"  Subtract: {A} - {B}")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.allclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_scalar_multiply():
    A = [[1.0, 2.0], [3.0, 4.0]]
    alpha = 2.0
    ours = basic_ops.scalar_multiply(alpha, A)
    theirs = (alpha * np.array(A)).tolist()
    
    print(f"  Scalar Multiply: {alpha} * {A}")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.allclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_transpose():
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    ours = basic_ops.transpose(A)
    theirs = np.array(A).T.tolist()
    
    print(f"  Transpose: {A}^T")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.allclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

def test_trace():
    A = [[1.0, 2.0], [3.0, 4.0]]
    ours = basic_ops.trace(A)
    theirs = float(np.trace(A))
    
    print(f"  Trace: tr({A})")
    print(f"    - Ours:  {ours}")
    print(f"    - NumPy: {theirs}")
    assert np.isclose(ours, theirs)
    print("    ✓ VERIFIED MATCH\n")

if __name__ == "__main__":
    print("=" * 60)
    print("VERIFICATION: UNIT 02 — BASIC MATRIX OPS")
    print("=" * 60)
    test_add()
    test_subtract()
    test_scalar_multiply()
    test_transpose()
    test_trace()
    print("ALL BASIC MATRIX OPS VERIFIED ✓")
