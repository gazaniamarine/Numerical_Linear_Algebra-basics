import numpy as np


def is_symmetric_positive_definite(A):
    """
    Simple educational SPD check.
    """
    A = np.asarray(A, dtype=np.float64)
    if not np.allclose(A, A.T):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def conjugate_gradient(A, b, x0=None, tol=1e-8, max_iterations=None, return_history=False):
    """
    Solve Ax = b with the Conjugate Gradient method for SPD matrices.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b must match the size of A.")
    if not is_symmetric_positive_definite(A):
        raise ValueError("Conjugate Gradient requires a symmetric positive definite matrix.")

    n = A.shape[0]
    max_iterations = n if max_iterations is None else max_iterations
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).copy()

    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)
    history = [x.copy()]

    for iteration in range(1, max_iterations + 1):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)

        x = x + alpha * p
        r = r - alpha * Ap
        history.append(x.copy())

        residual_norm = np.linalg.norm(r)
        if residual_norm < tol:
            info = {
                "iterations": iteration,
                "converged": True,
                "residual_norm": residual_norm,
            }
            if return_history:
                return x, info, history
            return x, info

        rs_new = np.dot(r, r)
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    info = {
        "iterations": max_iterations,
        "converged": False,
        "residual_norm": np.linalg.norm(r),
    }
    if return_history:
        return x, info, history
    return x, info
