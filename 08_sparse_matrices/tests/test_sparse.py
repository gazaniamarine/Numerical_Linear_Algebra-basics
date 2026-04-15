import sys
import os
import numpy as np
import pytest

# set sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

from sparse_formats import (
    dense_to_coo, coo_to_dense, 
    dense_to_csr, csr_matvec,
    dense_to_csc, csc_matvec,
    storage_efficiency
)

def test_coo_conversion():
    A = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 0]], dtype=float)
    rows, cols, data = dense_to_coo(A)
    
    # Check A_recovered
    A_recovered = coo_to_dense(rows, cols, data, A.shape)
    np.testing.assert_allclose(A_recovered, A, atol=1e-12)

def test_csr_matvec():
    A = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 0]], dtype=float)
    x = np.array([1, 2, 3], dtype=float)
    y_true = A @ x
    
    data, cols, indptr = dense_to_csr(A)
    y_sparse = csr_matvec(data, cols, indptr, x)
    
    np.testing.assert_allclose(y_sparse, y_true, atol=1e-12)

def test_csc_matvec():
    A = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 0]], dtype=float)
    x = np.array([1, 2, 3], dtype=float)
    y_true = A @ x
    
    data, rows, indptr = dense_to_csc(A)
    y_sparse = csc_matvec(data, rows, indptr, x)
    
    np.testing.assert_allclose(y_sparse, y_true, atol=1e-12)

def test_storage_efficiency():
    shape = (10, 10)
    nnz = 10
    eff = storage_efficiency(shape, nnz)
    
    assert eff['dense'] == 100
    assert eff['coo'] == 30
    assert eff['csr'] == 2 * 10 + 10 + 1
    assert eff['csc'] == 2 * 10 + 10 + 1
