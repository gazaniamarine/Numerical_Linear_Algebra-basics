# 01 — Vectors & Norms

Modular implementation of fundamental vector operations and norms.

## Structure

```
01_vectors_and_norms/
├── src/                        # Core implementation from scratch
│   ├── basic_ops.py            # Addition, Scalar Mult, Dot, Cross
│   ├── norms.py                # L1, L2, Linf, Lp, Normalization
│   ├── geometry.py             # Angle, Projection, Decomposition
│   └── orthogonalization.py    # Gram-Schmidt
├── tests/                      # Verification against NumPy
│   ├── test_basic_ops.py
│   ├── test_norms.py
│   └── test_ortho.py
└── examples/                   # Practical usage of src/ modules
    ├── basic_ops_example.py       # Show add, sub, dot, cross
    ├── norms_example.py           # Show all norms in action
    ├── gram_schmidt_example.py    # Show GS orthonormalization
    ├── unit_ball_plot.py       # Plot ball boundaries for L1/L2/Linf
    ├── cauchy_schwarz.py       # Verifying CS inequality
    └── nearest_neighbor.py     # NN under different norms
```

## How to Run

### 1. Verify (Tests)
Ensure we match NumPy perfectly.
```bash
python tests/test_basic_ops.py
python tests/test_norms.py
python tests/test_ortho.py
```

### 2. Examples (See outputs)
See how the code works with numbers.
```bash
python examples/basic_ops_example.py
python examples/norms_example.py
python examples/gram_schmidt_example.py
```

### 3. Visuals (Charts/Applications)
More theory-heavy/visual examples.
```bash
python examples/unit_ball_plot.py
python examples/cauchy_schwarz.py
python examples/nearest_neighbor.py
```
