import sys
import os
import numpy as np
import pytest

# set sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

from svd_computation import compute_svd, pseudoinverse, solve_least_norm
from low_rank_approx import low_rank_approximation, approximation_error, energy_captured

def test_svd_reconstruction():
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    U, s, Vt = compute_svd(A)
    
    # 1. Check dimensions
    assert U.shape == (3, 2)
    assert s.shape == (2,)
    assert Vt.shape == (2, 2)
    
    # 2. Check orthogonality of U and Vt
    np.testing.assert_allclose(U.T @ U, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(Vt @ Vt.T, np.eye(2), atol=1e-12)
    
    # 3. Check A = U S Vt
    A_reconstructed = U @ np.diag(s) @ Vt
    np.testing.assert_allclose(A_reconstructed, A, atol=1e-12)

def test_svd_wide_matrix():
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    U, s, Vt = compute_svd(A)
    
    # 1. Check dimensions
    assert U.shape == (2, 2)
    assert s.shape == (2,)
    assert Vt.shape == (2, 3)
    
    # 2. Check A = U S Vt
    A_reconstructed = U @ np.diag(s) @ Vt
    np.testing.assert_allclose(A_reconstructed, A, atol=1e-12)

def test_pseudoinverse():
    A = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
    A_plus = pseudoinverse(A)
    
    # Pseudo-inverse of this A should be [[1, 0, 0], [0, 1, 0]]
    expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    np.testing.assert_allclose(A_plus, expected, atol=1e-12)
    
    # Check Penrose properties
    # 1. A A+ A = A
    np.testing.assert_allclose(A @ A_plus @ A, A, atol=1e-12)
    # 2. A+ A A+ = A+
    np.testing.assert_allclose(A_plus @ A @ A_plus, A_plus, atol=1e-12)

def test_low_rank_approximation():
    A = np.array([[1, 1], [1, 1.0001]], dtype=float)
    # This matrix is nearly rank 1
    A_1 = low_rank_approximation(A, 1)
    
    # Check rank of A_1
    _, s, _ = np.linalg.svd(A_1)
    assert np.sum(s > 1e-10) == 1
    
    # Check error
    error = approximation_error(A, 1)
    _, s_full, _ = np.linalg.svd(A)
    np.testing.assert_allclose(error, s_full[1], atol=1e-12)
    
    # Check energy
    energy = energy_captured(A, 1)
    expected_energy = s_full[0]**2 / np.sum(s_full**2)
    np.testing.assert_allclose(energy, expected_energy, atol=1e-12)
