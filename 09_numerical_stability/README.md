# 09 — Numerical Stability

Analyzing the effects of finite precision and ill-conditioning.

## 📋 Contents

- **Machine Epsilon**: Understanding floating-point precision limits.
- **Condition Number**: Quantifying how sensitive a function is to input perturbations.
- **Perturbation Analysis**: Practical experiments measuring error propagation in linear solvers (using Hilbert matrices).

## 🚀 Examples

- `stability_example.py`: Investigating the sensitivity of Hilbert matrices to small RHS perturbations.

## 🧪 Testing

```bash
pytest tests/
```
