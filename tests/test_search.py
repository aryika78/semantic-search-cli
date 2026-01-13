from semantic_search.search import semantic_search


def test_semantic_search_returns_results():
    query = "heart disease symptoms"
    docs = [
        "Symptoms of heart failure",
        "How to bake a cake",
        "Cardiac arrest warning signs",
    ]

    results = semantic_search(query, docs, top_k=2)

    assert len(results) == 2
    assert isinstance(results[0][0], str)
    assert isinstance(results[0][1], float)


def test_semantic_search_ordering():
    query = "machine learning"
    docs = [
        "Deep learning models",
        "Cooking recipes",
        "Neural networks and ML",
    ]

    results = semantic_search(query, docs, top_k=3)

    # Most relevant should not be cooking
    assert "Cooking" not in results[0][0]
