import numpy as np


def machine_epsilon():
    """
    Return double-precision machine epsilon.
    """
    return np.finfo(np.float64).eps


def condition_number(A, ord=2):
    """
    Compute cond(A) with respect to the chosen norm.
    """
    A = np.asarray(A, dtype=np.float64)
    return np.linalg.cond(A, p=ord)


def relative_error(x_true, x_approx):
    """
    Relative vector error.
    """
    x_true = np.asarray(x_true, dtype=np.float64)
    x_approx = np.asarray(x_approx, dtype=np.float64)
    return np.linalg.norm(x_true - x_approx) / np.linalg.norm(x_true)


def residual_norm(A, x, b):
    """
    Residual norm ||b - Ax||.
    """
    A = np.asarray(A, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.linalg.norm(b - A @ x)


def forward_backward_error(A, x_true, x_approx, b):
    """
    Return both forward error and relative residual.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)
    x_approx = np.asarray(x_approx, dtype=np.float64)

    forward = relative_error(x_true, x_approx)
    backward = residual_norm(A, x_approx, b) / (np.linalg.norm(A) * np.linalg.norm(x_approx) + np.linalg.norm(b))
    return {"forward_error": forward, "backward_error": backward}


def hilbert_matrix(n):
    """
    Classic ill-conditioned test matrix.
    """
    i = np.arange(1, n + 1, dtype=np.float64)
    return 1.0 / (i[:, None] + i[None, :] - 1.0)


def perturbation_experiment(A, x_true, perturbation_scale=1e-8):
    """
    Measure how a small perturbation in b changes the computed solution.
    """
    A = np.asarray(A, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)
    b = A @ x_true

    perturbation = perturbation_scale * np.random.randn(*b.shape)
    b_perturbed = b + perturbation

    x_perturbed = np.linalg.solve(A, b_perturbed)
    return {
        "condition_number": condition_number(A),
        "relative_rhs_change": np.linalg.norm(perturbation) / np.linalg.norm(b),
        "relative_solution_change": relative_error(x_true, x_perturbed),
    }
