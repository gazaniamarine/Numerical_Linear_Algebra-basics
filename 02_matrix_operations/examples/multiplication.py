"""
Matrix Multiplication Operations
Shows matrix-vector and matrix-matrix multiplication.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import multiplication

def run_example():
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] # 2x3 matrix
    B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]] # 3x2 matrix
    x = [0.5, 1.0, 1.5] # vector of size 3
    
    print("--- Matrices & Vectors ---")
    print(f"  A (2x3) = {A}")
    print(f"  B (3x2) = {B}")
    print(f"  x       = {x}\n")
    
    print("--- Multiplication Operations ---")
    print(f"  A * x   = {multiplication.mat_vec_multiply(A, x)}")
    print(f"  A * B   = {multiplication.mat_mat_multiply(A, B)}")

if __name__ == "__main__":
    run_example()
