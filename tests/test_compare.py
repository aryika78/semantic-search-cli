from semantic_search.compare import TextComparator


def test_compare_output_structure():
    comparator = TextComparator()

    result = comparator.compare(
        "Machine learning is amazing",
        "I love studying AI and ML"
    )

    assert "cosine_similarity" in result
    assert "euclidean_distance" in result
    assert "dot_product" in result
    assert "interpretation" in result


def test_compare_similarity_reasonable():
    comparator = TextComparator()

    result = comparator.compare(
        "Deep learning models",
        "Neural networks in deep learning"
    )

    assert result["cosine_similarity"] > 0.5
