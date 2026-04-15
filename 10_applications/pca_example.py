import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "06_svd")))
from svd_computation import compute_svd


def pca_fit(X, n_components):
    """
    Fit PCA by centering data and taking the top right singular vectors.
    """
    X = np.asarray(X, dtype=np.float64)
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    _, singular_values, Vt = compute_svd(X_centered)
    n_components = min(n_components, Vt.shape[0])

    components = Vt[:n_components]
    explained_variance = (singular_values[:n_components] ** 2) / max(X.shape[0] - 1, 1)
    total_variance = np.sum((singular_values ** 2) / max(X.shape[0] - 1, 1))
    explained_ratio = explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)

    return {
        "mean": mean,
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_ratio,
    }


def pca_transform(X, model):
    """
    Project X onto the fitted PCA directions.
    """
    X = np.asarray(X, dtype=np.float64)
    return (X - model["mean"]) @ model["components"].T


def pca_reconstruct(scores, model):
    """
    Map PCA scores back to the original space.
    """
    scores = np.asarray(scores, dtype=np.float64)
    return scores @ model["components"] + model["mean"]
