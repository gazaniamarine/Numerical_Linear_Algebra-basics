# Unit 02: Matrix Operations

This module contains fundamental operations for matrices implemented entirely from scratch in Python.

## Structure

*   `src/`: Core implementation.
    *   `basic_ops.py`: Functions for matrix addition, subtraction, trace, transpose, and scalar multiplication.
    *   `multiplication.py`: Functions for matrix-vector and matrix-matrix multiplications.
*   `tests/`: Verification scripts ensuring exact parity with established `numpy` behavior.
*   `examples/`: Examples on how to use standard matrix operations programmatically.

## Verification

To verify the from-scratch implementations against NumPy:

```bash
python -m pytest tests/
```

Or you can run the files individually:

```bash
python tests/test_basic_ops.py
python tests/test_multiplication.py
```
