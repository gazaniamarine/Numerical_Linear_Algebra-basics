import numpy as np


def compute_svd(A, tol=1e-12):
    """
    Compute the thin SVD of A using the eigen-decomposition of A^T A.

    Returns U, singular_values, Vt such that A ~= U @ diag(singular_values) @ Vt.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape

    gram = A.T @ A
    eigenvalues, V = np.linalg.eigh(gram)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    V = V[:, order]

    singular_values = np.sqrt(np.clip(eigenvalues, 0.0, None))
    nonzero = singular_values > tol

    singular_values = singular_values[nonzero]
    V = V[:, nonzero]

    if singular_values.size == 0:
        return np.zeros((m, 0)), singular_values, np.zeros((0, n))

    U = (A @ V) / singular_values
    Vt = V.T
    return U, singular_values, Vt


def pseudoinverse(A, tol=1e-12):
    """
    Compute the Moore-Penrose pseudoinverse using the SVD.
    """
    U, singular_values, Vt = compute_svd(A, tol=tol)

    if singular_values.size == 0:
        return np.zeros((A.shape[1], A.shape[0]))

    Sigma_plus = np.diag(1.0 / singular_values)
    return Vt.T @ Sigma_plus @ U.T


def solve_least_norm(A, b, tol=1e-12):
    """
    Solve Ax = b in the least-squares / minimum-norm sense using the pseudoinverse.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return pseudoinverse(A, tol=tol) @ b
