import numpy as np
import sys
import os

def power_iteration(A, num_simulations=100, tol=1e-8):
    """
    Computes the dominant eigenvalue and its corresponding eigenvector.
    """
    m, n = A.shape
    # Start with a random vector
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)

    eigenvalue = 0
    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k_next = b_k1 / b_k1_norm

        # calculate Rayleigh quotient
        eigenvalue_next = np.dot(b_k_next.T, np.dot(A, b_k_next))

        if np.abs(eigenvalue_next - eigenvalue) < tol:
            b_k = b_k_next
            eigenvalue = eigenvalue_next
            break
            
        b_k = b_k_next
        eigenvalue = eigenvalue_next

    return eigenvalue, b_k

def inverse_iteration(A, mu, num_simulations=100, tol=1e-8):
    """
    Computes the eigenvalue of A closest to the given shift mu,
    and its corresponding eigenvector using Inverse Iteration.
    """
    m, n = A.shape
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)

    I = np.eye(n)
    shifted_A = A - mu * I

    eigenvalue = 0
    for _ in range(num_simulations):
        # Solve (A - mu*I) y = b_k
        try:
            # We use np.linalg.solve to solve the linear system
            y = np.linalg.solve(shifted_A, b_k)
        except np.linalg.LinAlgError:
            print("Matrix is exactly singular, shift is an exact eigenvalue!")
            return mu, b_k
            
        y_norm = np.linalg.norm(y)
        b_k_next = y / y_norm

        # Rayleigh quotient for the original matrix A to find the exact eigenvalue
        eigenvalue_next = np.dot(b_k_next.T, np.dot(A, b_k_next))

        if np.abs(eigenvalue_next - eigenvalue) < tol:
            b_k = b_k_next
            eigenvalue = eigenvalue_next
            break

        b_k = b_k_next
        eigenvalue = eigenvalue_next

    return eigenvalue, b_k
