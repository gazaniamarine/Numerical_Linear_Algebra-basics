import sys
import os
import numpy as np
import pytest

# set sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

from conditioning import (
    machine_epsilon, condition_number, relative_error, 
    residual_norm, forward_backward_error, hilbert_matrix,
    perturbation_experiment
)

def test_machine_epsilon():
    eps = machine_epsilon()
    assert 1.0 + eps > 1.0
    assert 1.0 + eps/2 == 1.0

def test_condition_number():
    A = np.array([[1, 0], [0, 100]], dtype=float)
    cond = condition_number(A, ord=2)
    assert np.isclose(cond, 100.0)

def test_errors():
    x_true = np.array([1, 1], dtype=float)
    x_approx = np.array([1.1, 0.9], dtype=float)
    
    rel_err = relative_error(x_true, x_approx)
    # ||[0.1, -0.1]||_2 = sqrt(0.02)
    # ||[1, 1]||_2 = sqrt(2)
    # rel_err = sqrt(0.01) = 0.1
    assert np.isclose(rel_err, 0.1)

def test_hilbert_matrix():
    H = hilbert_matrix(3)
    expected = np.array([
        [1.0, 1.0/2, 1.0/3],
        [1.0/2, 1.0/3, 1.0/4],
        [1.0/3, 1.0/4, 1.0/5]
    ])
    np.testing.assert_allclose(H, expected, atol=1e-12)

def test_perturbation_experiment():
    A = hilbert_matrix(5)
    x_true = np.ones(5)
    results = perturbation_experiment(A, x_true)
    
    assert 'condition_number' in results
    assert 'relative_rhs_change' in results
    assert 'relative_solution_change' in results
    assert results['condition_number'] > 1e4  # Hilbert(5) is ill-conditioned
