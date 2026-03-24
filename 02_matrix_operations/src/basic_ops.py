"""Basic matrix operations implemented from scratch."""

def shape(A):
    """Return the dimensions (rows, cols) of a matrix A."""
    return len(A), len(A[0]) if A else 0

def add(A, B):
    """Element-wise addition: A + B"""
    rows_A, cols_A = shape(A)
    rows_B, cols_B = shape(B)
    assert rows_A == rows_B and cols_A == cols_B, "Matrices must have same dimensions"
    return [[A[i][j] + B[i][j] for j in range(cols_A)] for i in range(rows_A)]

def subtract(A, B):
    """Element-wise subtraction: A - B"""
    rows_A, cols_A = shape(A)
    rows_B, cols_B = shape(B)
    assert rows_A == rows_B and cols_A == cols_B, "Matrices must have same dimensions"
    return [[A[i][j] - B[i][j] for j in range(cols_A)] for i in range(rows_A)]

def scalar_multiply(alpha, A):
    """Scalar multiplication: alpha * A"""
    rows, cols = shape(A)
    return [[alpha * A[i][j] for j in range(cols)] for i in range(rows)]

def transpose(A):
    """Transpose of a matrix: A^T"""
    rows, cols = shape(A)
    return [[A[i][j] for i in range(rows)] for j in range(cols)]

def trace(A):
    """Trace of a square matrix A."""
    rows, cols = shape(A)
    assert rows == cols, "Matrix must be square"
    return sum(A[i][i] for i in range(rows))
