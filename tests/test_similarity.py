import numpy as np

from semantic_search.similarity import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
)


def test_cosine_identical_vectors():
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    assert cosine_similarity(a, b) == 1.0


def test_cosine_orthogonal_vectors():
    a = np.array([1, 0])
    b = np.array([0, 1])
    assert cosine_similarity(a, b) == 0.0


def test_euclidean_distance_zero():
    a = np.array([1, 2, 3])
    assert euclidean_distance(a, a) == 0.0


def test_dot_product():
    a = np.array([1, 2])
    b = np.array([3, 4])
    assert dot_product(a, b) == 11.0
