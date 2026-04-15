"""
Example: Nearest neighbor search under different norms.
Uses norms.lp and basic_ops.subtract from src.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import norms, basic_ops

def find_nearest_neighbor(query, points, norm_type='l2'):
    best_dist = float('inf')
    best_idx = -1
    
    for i, p in enumerate(points):
        diff = basic_ops.subtract(query, p)
        if norm_type == 'l1': dist = norms.l1(diff)
        elif norm_type == 'l2': dist = norms.l2(diff)
        elif norm_type == 'linf': dist = norms.linf(diff)
        else: raise ValueError("Unsupported norm_type")
        
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            
    return best_idx, best_dist

def run_example():
    points = [
        [1, 1], [0, 1], [-1, 2], [2, 0.5], [1.5, 1.5]
    ]
    query = [0, 0]
    
    print("--- Nearest Neighbor Example (Unit: 01) ---")
    for nt in ['l1', 'l2', 'linf']:
        idx, dist = find_nearest_neighbor(query, points, nt)
        print(f"Norm {nt.upper():<4}: Nearest point is index {idx} (distance {dist:.4f})")

if __name__ == "__main__":
    run_example()
