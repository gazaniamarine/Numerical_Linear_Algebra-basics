import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "06_svd")))
from low_rank_approx import low_rank_approximation


def compress_image_matrix(image, rank):
    """
    Compress a grayscale image matrix or each channel of an RGB array using rank-k SVD.
    """
    image = np.asarray(image, dtype=np.float64)

    if image.ndim == 2:
        compressed = low_rank_approximation(image, rank)
    elif image.ndim == 3:
        channels = [low_rank_approximation(image[:, :, i], rank) for i in range(image.shape[2])]
        compressed = np.stack(channels, axis=2)
    else:
        raise ValueError("image must be a 2D grayscale matrix or a 3D color array.")

    return np.clip(compressed, 0, 255)


def compression_ratio(shape, rank):
    """
    Estimate how many stored numbers rank-k SVD needs compared with the full image.
    """
    if len(shape) == 2:
        m, n = shape
        original = m * n
        compressed = rank * (m + n + 1)
    elif len(shape) == 3:
        m, n, c = shape
        original = m * n * c
        compressed = c * rank * (m + n + 1)
    else:
        raise ValueError("shape must describe a 2D or 3D image.")

    return original / compressed
