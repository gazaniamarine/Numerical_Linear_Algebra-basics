import numpy as np


def gauss_seidel_method(A, b, x0=None, tol=1e-8, max_iterations=100, return_history=False):
    """
    Solve Ax = b using the Gauss-Seidel iteration.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must match the size of A.")
    if np.any(np.isclose(np.diag(A), 0.0)):
        raise ValueError("Gauss-Seidel method requires nonzero diagonal entries.")

    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    history = [x.copy()]
    n = A.shape[0]

    for iteration in range(1, max_iterations + 1):
        x_new = x.copy()
        for i in range(n):
            left_sum = np.dot(A[i, :i], x_new[:i])
            right_sum = np.dot(A[i, i + 1 :], x[i + 1 :])
            x_new[i] = (b[i] - left_sum - right_sum) / A[i, i]

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


def sor_method(A, b, omega=1.0, x0=None, tol=1e-8, max_iterations=100, return_history=False):
    """
    Successive Over-Relaxation iteration. omega=1 recovers Gauss-Seidel.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if not 0.0 < omega < 2.0:
        raise ValueError("omega must satisfy 0 < omega < 2.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must match the size of A.")
    if np.any(np.isclose(np.diag(A), 0.0)):
        raise ValueError("SOR requires nonzero diagonal entries.")

    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    history = [x.copy()]
    n = A.shape[0]

    for iteration in range(1, max_iterations + 1):
        x_new = x.copy()
        for i in range(n):
            left_sum = np.dot(A[i, :i], x_new[:i])
            right_sum = np.dot(A[i, i + 1 :], x[i + 1 :])
            gs_update = (b[i] - left_sum - right_sum) / A[i, i]
            x_new[i] = (1.0 - omega) * x[i] + omega * gs_update

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
