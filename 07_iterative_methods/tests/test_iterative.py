import sys
import os
import numpy as np
import pytest

# set sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

from jacobi import jacobi_method, jacobi_convergence_factor
from gauss_seidel import gauss_seidel_method, sor_method
from conjugate_gradient import conjugate_gradient

def test_jacobi_convergence():
    # Diagonally dominant matrix
    A = np.array([[4, 1, 1], [1, 3, 1], [1, 1, 5]], dtype=float)
    b = np.array([6, 5, 7], dtype=float)
    x_true = np.linalg.solve(A, b)
    
    x, info = jacobi_method(A, b, tol=1e-10)
    assert info['converged']
    np.testing.assert_allclose(x, x_true, atol=1e-8)
    
    rho = jacobi_convergence_factor(A)
    assert rho < 1.0

def test_gauss_seidel_convergence():
    A = np.array([[4, 1, 1], [1, 3, 1], [1, 1, 5]], dtype=float)
    b = np.array([6, 5, 7], dtype=float)
    x_true = np.linalg.solve(A, b)
    
    x, info = gauss_seidel_method(A, b, tol=1e-10)
    assert info['converged']
    np.testing.assert_allclose(x, x_true, atol=1e-8)

def test_sor_convergence():
    A = np.array([[4, 1, 1], [1, 3, 1], [1, 1, 5]], dtype=float)
    b = np.array([6, 5, 7], dtype=float)
    x_true = np.linalg.solve(A, b)
    
    # omega = 1 should be Gauss-Seidel
    x, info = sor_method(A, b, omega=1.0, tol=1e-10)
    assert info['converged']
    np.testing.assert_allclose(x, x_true, atol=1e-8)
    
    # Try a slightly different omega
    x_sor, info_sor = sor_method(A, b, omega=1.1, tol=1e-10)
    assert info_sor['converged']
    np.testing.assert_allclose(x_sor, x_true, atol=1e-8)

def test_conjugate_gradient():
    # SPD matrix
    A = np.array([[4, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    x_true = np.linalg.solve(A, b)
    
    x, info = conjugate_gradient(A, b, tol=1e-10)
    assert info['converged']
    np.testing.assert_allclose(x, x_true, atol=1e-8)

def test_cg_non_spd():
    # Not symmetric
    A = np.array([[4, 1], [2, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    with pytest.raises(ValueError, match="symmetric positive definite"):
        conjugate_gradient(A, b)
