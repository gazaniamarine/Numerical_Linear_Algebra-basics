# 07 — Iterative Methods

Solving large-scale linear systems using stationary and non-stationary iterations.

## 📋 Contents

- **Jacobi Method**: Simple component-wise iteration.
- **Gauss-Seidel**: Accelerated stationary method using updated values immediately.
- **Successive Over-Relaxation (SOR)**: Extrapolated Gauss-Seidel for faster convergence.
- **Conjugate Gradient (CG)**: Krylov subspace method for symmetric positive-definite systems.

## 🚀 Examples

- `iterative_example.py`: Comparing convergence rates of Jacobi, Gauss-Seidel, and CG on a sample system.

## 🧪 Testing

```bash
pytest tests/
```
