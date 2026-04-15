"""
Vector Geometry
Examples angles, projections, and decompositions.
"""
import sys
import os
import math

# Add parent directory to sys.path to enable imports from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import geometry, basic_ops, norms

def run_example():
    print("--- 2D Geometry Example ---")
    u = [1.0, 0.0]
    v = [1.0, 1.0]
    print(f"  Vector u: {u}")
    print(f"  Vector v: {v}")

    # Angle
    theta_rad = geometry.angle(u, v)
    theta_deg = math.degrees(theta_rad)
    print(f"  Angle between u and v: {theta_rad:.4f} rad ({theta_deg:.1f}°)")

    # Projection
    v_on_u = geometry.project(v, u)
    print(f"  Projection of v onto u: {v_on_u}")

    # Decomposition
    v_parallel, v_perp = geometry.decompose(v, u)
    print(f"  Decomposition of v relative to u:")
    print(f"    Parallel part (projected onto u): {v_parallel}")
    print(f"    Perpendicular part: {v_perp}")
    
    # Verification of decomposition: v_parallel + v_perp == v
    v_sum = basic_ops.add(v_parallel, v_perp)
    print(f"    Verify (v_parallel + v_perp): {v_sum} (should be {v})")
    
    # Check orthogonality of v_perp and u
    dot_perp_u = basic_ops.dot_product(v_perp, u)
    print(f"    Verify (v_perp · u): {dot_perp_u:.12f} (should be approx 0)")

    print("\n--- 3D Geometry Example ---")
    a = [3, 4, 12]
    b = [1, 2, 2]
    print(f"  Vector a: {a} (norm: {norms.l2(a):.2f})")
    print(f"  Vector b: {b} (norm: {norms.l2(b):.2f})")

    theta_3d_rad = geometry.angle(a, b)
    theta_3d_deg = math.degrees(theta_3d_rad)
    print(f"  Angle: {theta_3d_rad:.4f} rad ({theta_3d_deg:.1f}°)")
    
    a_on_b = geometry.project(a, b)
    print(f"  Projection of a onto b: {[round(x, 4) for x in a_on_b]}")

if __name__ == "__main__":
    run_example()
