import numpy as np


def iteration_matrix_jacobi(A):
    """
    Build the Jacobi iteration matrix: B = -D^{-1}(L + U).
    """
    A = np.asarray(A, dtype=np.float64)
    D = np.diag(np.diag(A))
    R = A - D
    diagonal = np.diag(D)

    if np.any(np.isclose(diagonal, 0.0)):
        raise ValueError("Jacobi method requires nonzero diagonal entries.")

    return -np.diag(1.0 / diagonal) @ R


def jacobi_method(A, b, x0=None, tol=1e-8, max_iterations=100, return_history=False):
    """
    Solve Ax = b using the Jacobi fixed-point iteration.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must match the size of A.")

    diagonal = np.diag(A)
    if np.any(np.isclose(diagonal, 0.0)):
        raise ValueError("Jacobi method requires nonzero diagonal entries.")

    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    history = [x.copy()]

    for iteration in range(1, max_iterations + 1):
        x_new = (b - (A @ x - diagonal * x)) / diagonal
        history.append(x_new.copy())

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            info = {
                "iterations": iteration,
                "converged": True,
                "residual_norm": np.linalg.norm(b - A @ x_new),
            }
            if return_history:
                return x_new, info, history
            return x_new, info

        x = x_new

    info = {
        "iterations": max_iterations,
        "converged": False,
        "residual_norm": np.linalg.norm(b - A @ x),
    }
    if return_history:
        return x, info, history
    return x, info


def jacobi_convergence_factor(A):
    """
    Estimate convergence using the spectral radius of the Jacobi iteration matrix.
    """
    eigenvalues = np.linalg.eigvals(iteration_matrix_jacobi(A))
    return np.max(np.abs(eigenvalues))
