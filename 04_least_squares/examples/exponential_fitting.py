# Least Squares — Exponential Fitting Example
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Custom Module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.normal_equations import least_squares
from src.fitting import exponential_model

def run_example():
    # 0. Setup Dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "outputs", "exponential_fitting")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Synthetic Data for example
    np.random.seed(42)
    x = np.linspace(0, 5, 50)
    a_true, b_true = 3.0, -0.5
    y_noisy = a_true * np.exp(b_true * x) + np.random.normal(0, 0.05, size=x.shape)
    y_noisy = np.maximum(y_noisy, 1e-6) # Ensure y > 0 for log transformation
    
    # 1. Solve: Model linearized as ln(y) = ln(a) + b * x
    A = exponential_model(x)
    params_log, residual_log = least_squares(A, np.log(y_noisy))
    
    # Recover original parameters
    a_fit, b_fit = np.exp(params_log[0]), params_log[1]
    
    print(f"--- Fitting Exponential Model ---")
    print(f"Fitted 'a': {a_fit:.6f} (Expected: {a_true})")
    print(f"Fitted 'b': {b_fit:.6f} (Expected: {b_true})")
    print(f"Log-Residual: {residual_log:.6f}")
    
    # 2. Plotting (Save to Outputs folder)
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_noisy, color="aqua", s=20, label="Noisy Data", alpha=0.6)
    
    x_line = np.linspace(min(x), max(x), 100)
    y_line = a_fit * np.exp(b_fit * x_line)
    plt.plot(x_line, y_line, label=f"Exponential Fit: {a_fit:.2f}*e^({b_fit:.2f}*x)", 
             color="orangered", linewidth=2)
    
    plt.title("Exponential Fitting — Least Squares")
    plt.ylabel("Value")
    plt.xlabel("Domain")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "exponential-fit-result.png"))
    print(f"Result saved to: {output_dir}")

if __name__ == "__main__":
    run_example()