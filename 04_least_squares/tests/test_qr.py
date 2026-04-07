import sys
import os
import numpy as np
import pytest

# set sys.path for local src imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'src'))
from qr_decomposition import qr_factor_mgs, solve_least_squares_qr

def test_qr_reconstruction():
    # Test A = QR reconstruction
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    Q, R = qr_factor_mgs(A)
    
    # 1. Check Q is orthogonal (Q.T * Q = I)
    identity = np.eye(Q.shape[1])
    np.testing.assert_allclose(np.dot(Q.T, Q), identity, atol=1e-12)
    
    # 2. Check R is upper triangular
    assert np.all(np.tril(R, k=-1) == 0)
    
    # 3. Check A = QR
    np.testing.assert_allclose(np.dot(Q, R), A, atol=1e-12)

def test_qr_solver():
    # Simple linear regression solve via QR
    A = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
    b = np.array([1, 2.1, 2.9], dtype=float)
    
    x_man, res_man = solve_least_squares_qr(A, b)
    
    # Compare with NumPy
    x_np, res_np, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    np.testing.assert_allclose(x_man, x_np, atol=1e-12)
    # NumPy's lsq return squared residual norm sum if it computes it
    # Our solver returns ||b - Ax||. 
    # lsq[1] is sum of squared residuals.
    np.testing.assert_allclose(res_man, np.sqrt(res_np[0]), atol=1e-12)

def test_qr_singular():
    # Test with a singular matrix column-wise
    A = np.array([[1, 1], [1, 1], [1, 1]], dtype=float)
    Q, R = qr_factor_mgs(A)
    # The current MGS implementation prints a warning but continues. 
    # R[1,1] should be 0.
    assert np.abs(R[1,1]) < 1e-12
