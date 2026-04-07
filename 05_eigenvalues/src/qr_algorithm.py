import numpy as np
import sys
import os

# Import the Modified Gram-Schmidt QR we built back in Module 04!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '04_least_squares', 'src')))
try:
    from qr_decomposition import qr_factor_mgs
except ImportError:
    # Defensive fallback
    def qr_factor_mgs(A):
        return np.linalg.qr(A)

def qr_algorithm(A, num_simulations=100, tol=1e-8):
    """
    Computes all eigenvalues of a matrix A using the iterative QR Algorithm.
    Returns an array of eigenvalues.
    """
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix must be square to compute eigenvalues.")
        
    A_k = A.astype(np.float64).copy()
    
    for k in range(num_simulations):
        # 1. Calculate QR Decomposition (A_k = Q * R)
        Q, R = qr_factor_mgs(A_k)
        
        # 2. Recombine in reverse order: (A_{k+1} = R * Q)
        A_k_next = np.dot(R, Q)
        
        # Check for convergence: the strictly lower triangular part should go to zero.
        lower_triangular = np.tril(A_k_next, -1)
        if np.linalg.norm(lower_triangular) < tol:
            A_k = A_k_next
            break
            
        A_k = A_k_next
        
    # As k -> infinity, A_k approaches an upper triangular matrix (or block triangular)
    # The eigenvalues lie precisely on the diagonal of the final A_k
    eigenvalues = np.diag(A_k)
    return eigenvalues
