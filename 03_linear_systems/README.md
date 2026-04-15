# 03 — Linear Systems

Direct methods for solving systems of linear equations ($Ax = b$).

## 📋 Contents

- **Gaussian Elimination**: Forward substitution and back-substitution with partial pivoting.
- **LU Decomposition**: Factoring a matrix into Lower and Upper triangular components.
- **Cholesky Decomposition**: Efficient solving for symmetric positive-definite (SPD) matrices.
- **LDL / LDV**: Variant decompositions for symmetric and general matrices.

## 🚀 Examples

- `cholesky_example.py`: Solving SPD systems efficiently.
- `lu_example.py`: Step-by-step LU decomposition visualization.
- `numpy_solvers_example.py`: Comparing custom direct solvers with `numpy.linalg.solve`.

## 🧪 Testing

```bash
pytest tests/
```
