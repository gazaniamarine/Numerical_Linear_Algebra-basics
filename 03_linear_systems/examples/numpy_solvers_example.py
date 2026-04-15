# NumPy Solvers — Performance Benchmark
# Comparison of full system solver vs. triangular system solver in NumPy.

import numpy as np
import time

def solve_perf_test(sz=2000):
    # 1. Full random matrix
    A_full = np.random.rand(sz, sz) + np.eye(sz) * 10
    b_vec = np.random.rand(sz, 1)

    # 2. Extract upper triangular part for comparison
    A_tri = np.triu(A_full)

    # 3. Solve full system
    s_full = time.time()
    x_full = np.linalg.solve(A_full, b_vec)
    t_full = time.time() - s_full
    print(f"Full {sz}x{sz} Matrix Solve Time: {t_full:.6f}s")

    # 4. solve triangular system
    # Note: NumPy's solve is faster on triangular matrices if identified
    s_tri = time.time()
    x_tri = np.linalg.solve(A_tri, b_vec)
    t_tri = time.time() - s_tri
    print(f"Triangular {sz}x{sz} Matrix Solve Time: {t_tri:.6f}s")

    print(f"Speedup for triangular system: {t_full / t_tri:.2f}x")

def run_example():
    print("--- NumPy Solvers Performance Example ---")
    
    # 1. Basic Small Test
    A_small = np.array([[3, 1], [1, 2]], dtype=float)
    b_small = np.array([9, 8], dtype=float)
    x_small = np.linalg.solve(A_small, b_small)
    print(f"Small System Solution x:\n{x_small}")
    
    # 2. Large Performance Test
    print("\n--- Large System Benchmark (n=2000) ---")
    solve_perf_test(sz=2000)

if __name__ == "__main__":
    run_example()
