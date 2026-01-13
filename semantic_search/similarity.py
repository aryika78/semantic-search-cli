import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measures semantic similarity based on angle between vectors.
    Range: -1 to 1
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measures straight-line distance between vectors.
    Lower value = more similar
    """
    return float(np.linalg.norm(a - b))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measures raw overlap between vectors.
    """
    return float(np.dot(a, b))
