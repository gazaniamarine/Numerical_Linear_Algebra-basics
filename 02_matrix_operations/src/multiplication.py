"""Matrix multiplication operations implemented from scratch."""
from src import basic_ops

def mat_vec_multiply(A, x):
    """Matrix-vector multiplication: A * x"""
    rows, cols = basic_ops.shape(A)
    assert cols == len(x), "Matrix columns must equal vector length"
    return [sum(A[i][j] * x[j] for j in range(cols)) for i in range(rows)]

def mat_mat_multiply(A, B):
    """Matrix-matrix multiplication: A * B"""
    rows_A, cols_A = basic_ops.shape(A)
    rows_B, cols_B = basic_ops.shape(B)
    assert cols_A == rows_B, "Matrix A columns must equal Matrix B rows"
    
    C = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return C
