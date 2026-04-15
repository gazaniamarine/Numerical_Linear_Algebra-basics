import sys
import os
import numpy as np
import pytest

# set sys.path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..'))

from image_compression_svd import compress_image_matrix, compression_ratio
from pagerank import pagerank, build_transition_matrix
from pca_example import pca_fit, pca_transform, pca_reconstruct

def test_pagerank():
    # Simple graph: 0 -> 1, 1 -> 0, 0 -> 2
    adj = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 0]
    ])
    # Note: build_transition_matrix expects adjacency where adj[i,j]=1 means j -> i?
    # Let's check the implementation: transition[:, j] = adjacency[:, j] / out_degree[j]
    # So adjacency[:, j] is the column of nodes that node j points to.
    
    ranks, info = pagerank(adj)
    assert info['converged']
    assert np.isclose(np.sum(ranks), 1.0)
    # Node 0 and 1 are in a loop, but 0 also points to 2.
    # PageRank should give some score to all.
    assert ranks[0] > 0
    assert ranks[1] > 0
    assert ranks[2] > 0

def test_pca():
    # Data along a line y = x
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4]
    ])
    model = pca_fit(X, n_components=1)
    
    assert np.isclose(model['explained_variance_ratio'][0], 1.0)
    
    X_transformed = pca_transform(X, model)
    assert X_transformed.shape == (4, 1)
    
    X_reconstructed = pca_reconstruct(X_transformed, model)
    np.testing.assert_allclose(X_reconstructed, X, atol=1e-12)

def test_image_compression_shape():
    img = np.random.rand(10, 10) * 255
    compressed = compress_image_matrix(img, rank=2)
    assert compressed.shape == (10, 10)
    
    ratio = compression_ratio((10, 10), rank=2)
    # original = 100, compressed = 2 * (10 + 10 + 1) = 42
    # ratio = 100 / 42 ~= 2.38
    assert np.isclose(ratio, 100/42)
