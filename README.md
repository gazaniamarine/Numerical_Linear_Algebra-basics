# Numerical Linear Algebra — Basics

A comprehensive repository for learning and implementing fundamental numerical linear algebra algorithms from scratch. This project prioritizes educational clarity and numerical correctness, verifying each algorithm against established libraries like NumPy.

## 🚀 Key Features

- **From-Scratch Implementations**: Core algorithms built without high-level library abstractions.
- **Verification Driven**: Comprehensive test suites for every module to ensure numerical parity.
- **Practical Examples**: Hands-on example scripts demonstrating real-world applications (e.g., Image Compression, PageRank).
- **Modern Aesthetics**: Clean, organized, and professional codebase structure.

## 📚 Topics Covered

| # | Topic | Key Algorithms | Status |
|---|-------|----------------|--------|
| 01 | **Vectors & Norms** | Norms (L1, L2, Inf), Gram-Schmidt | ✅ |
| 02 | **Matrix Operations** | Block Multiplication, Transpose, Trace | ✅ |
| 03 | **Linear Systems** | Gaussian Elimination, LU, Cholesky, LDL | ✅ |
| 04 | **Least Squares** | QR Decomposition, Normal Equations | ✅ |
| 05 | **Eigenvalues** | Power Iteration, QR Algorithm | ✅ |
| 06 | **SVD** | Thin SVD, Pseudoinverse, Low-Rank Approx | ✅ |
| 07 | **Iterative Methods** | Jacobi, Gauss-Seidel, CG, SOR | ✅ |
| 08 | **Sparse Matrices** | COO, CSR, CSC storage & matvec | ✅ |
| 09 | **Numerical Stability** | Condition Numbers, Perturbation Analysis | ✅ |
| 10 | **Applications** | PageRank, PCA, Image Compression | ✅ |

## 🛠️ Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/Numerical_Linear_Algebra-basics.git
cd Numerical_Linear_Algebra-basics
pip install -r requirements.txt
```

## 🧪 Running Tests

To run the entire test suite:
```bash
pytest
```

To run tests for a specific module:
```bash
pytest 06_svd/tests/
```

## 📁 Repository Structure

Each topic folder is self-contained:
- **`src/`** / **`implementations.py`**: Core algorithm implementations.
- **`tests/`**: Unit tests verifying accuracy against NumPy.
- **`examples/`**: Scripts showing the algorithms in action.
- **`exercises/`**: Placeholder for practice problems.

## 📜 License
MIT License
