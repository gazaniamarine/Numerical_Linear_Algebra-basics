# 04 — Least Squares

Solving overdetermined systems and curve fitting using orthogonalization.

## 📋 Contents

- **Normal Equations**: The traditional $A^T A x = A^T b$ approach.
- **QR Decomposition**: More stable solving using Gram-Schmidt or Householder (MGS implemented).
- **Linear Regression**: Practical application of least squares for data fitting.

## 🚀 Examples

- `qr_example.py`: Using QR decomposition to solve least squares problems.
- `exponential_fitting.py`: Fitting non-linear data using linearized least squares.

## 🧪 Testing

```bash
pytest tests/
```
