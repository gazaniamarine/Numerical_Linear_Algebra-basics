# 08 — Sparse Matrices

Storage and operations for matrices with mostly zero entries.

## 📋 Contents

- **Formats**: Implementations of COO (Coordinate), CSR (Compressed Sparse Row), and CSC (Compressed Sparse Column).
- **Sparse Matvec**: Efficient matrix-vector multiplication for compressed formats.
- **Storage Efficiency**: Tools to measure memory savings of sparse vs. dense storage.

## 🚀 Examples

- `sparse_example.py`: Benchmarking storage efficiency and matvec performance on random sparse matrices.

## 🧪 Testing

```bash
pytest tests/
```
