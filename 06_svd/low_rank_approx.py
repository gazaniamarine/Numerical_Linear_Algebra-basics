import numpy as np

from svd_computation import compute_svd


def low_rank_approximation(A, rank):
    """
    Return the best rank-k approximation to A in the SVD sense.
    """
    U, singular_values, Vt = compute_svd(A)
    rank = max(0, min(rank, singular_values.size))

    if rank == 0:
        return np.zeros_like(np.asarray(A, dtype=np.float64))

    U_k = U[:, :rank]
    S_k = np.diag(singular_values[:rank])
    Vt_k = Vt[:rank, :]
    return U_k @ S_k @ Vt_k


def approximation_error(A, rank):
    """
    Frobenius norm of A - A_k.
    """
    A = np.asarray(A, dtype=np.float64)
    A_k = low_rank_approximation(A, rank)
    return np.linalg.norm(A - A_k, ord="fro")


def energy_captured(A, rank):
    """
    Fraction of squared singular-value energy captured by a rank-k approximation.
    """
    _, singular_values, _ = compute_svd(A)
    if singular_values.size == 0:
        return 0.0

    rank = max(0, min(rank, singular_values.size))
    total = np.sum(singular_values ** 2)
    kept = np.sum(singular_values[:rank] ** 2)
    return kept / total
