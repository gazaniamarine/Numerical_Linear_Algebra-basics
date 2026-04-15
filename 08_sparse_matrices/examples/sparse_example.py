import sys
import os
import numpy as np

# set sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

from sparse_formats import dense_to_csr, csr_matvec, storage_efficiency

def run_sparse_example():
    print("--- Sparse Matrices Example ---")
    # A 100x100 matrix that is 95% zeros
    m, n = 100, 100
    A = np.zeros((m, n))
    nnz = 50
    for _ in range(nnz):
        i, j = np.random.randint(0, m), np.random.randint(0, n)
        A[i, j] = np.random.randn()
    
    print(f"Matrix size: {m}x{n}, non-zeros: {nnz}")
    
    # Storage efficiency
    eff = storage_efficiency(A.shape, nnz)
    print("\nStorage counts (number of values stored):")
    for fmt, count in eff.items():
        print(f"  {fmt.upper()}: {count}")
    
    # CSR operations
    data, indices, indptr = dense_to_csr(A)
    x = np.random.randn(n)
    
    y_dense = A @ x
    y_sparse = csr_matvec(data, indices, indptr, x)
    
    print("\nDense vs CSR matvec error:", np.linalg.norm(y_dense - y_sparse))

if __name__ == "__main__":
    run_sparse_example()
