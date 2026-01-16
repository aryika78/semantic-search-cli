from semantic_search.compare import TextComparator
from semantic_search.compare import interpret_score


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

def test_interpretation():
    assert interpret_score(0.95) == "Very Similar"
    assert interpret_score(0.8) == "Similar"
    assert interpret_score(0.5) == "Somewhat Similar"
    assert interpret_score(0.2) == "Not Similar"

