import sys
import os
import numpy as np
import pytest

# set sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

from src.power_method import power_iteration, inverse_iteration
from src.qr_algorithm import qr_algorithm

def test_power_iteration():
    np.random.seed(42)
    A = np.array([[2, 1], [1, 2]], dtype=float)
    # Eigenvalues are 3 and 1
    val, vec = power_iteration(A, num_simulations=1000, tol=1e-15)
    assert np.isclose(val, 3.0)
    # Check vec is eigenvector
    np.testing.assert_allclose(A @ vec, val * vec, atol=1e-8)

def test_inverse_iteration():
    np.random.seed(42)
    A = np.array([[2, 1], [1, 2]], dtype=float)
    # Target eigenvalue 1.0 by shifting near it
    val, vec = inverse_iteration(A, mu=0.8, num_simulations=1000, tol=1e-15)
    assert np.isclose(val, 1.0)
    np.testing.assert_allclose(A @ vec, val * vec, atol=1e-8)

def test_qr_algorithm():
    A = np.array([[2, 1], [1, 2]], dtype=float)
    eigenvalues = qr_algorithm(A)
    # Should find [3, 1] or [1, 3]
    assert np.any(np.isclose(eigenvalues, 3.0))
    assert np.any(np.isclose(eigenvalues, 1.0))
