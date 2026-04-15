# 06 — Singular Value Decomposition (SVD)

Theory and implementation of the "Swiss Army Knife" of matrix factorizations.

## 📋 Contents

- **SVD Computation**: Thin SVD implementation using the eigenvalues of $A^T A$.
- **Low-Rank Approximation**: Best rank-$k$ approximation (Eckart-Young-Mirsky Theorem).
- **Pseudoinverse**: Moore-Penrose pseudoinverse calculation via SVD.

## 🚀 Examples

- `svd_example.py`: Demonstrating matrix reconstruction and rank-1 approximations.

## 🧪 Testing

```bash
pytest tests/
```
