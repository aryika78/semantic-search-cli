from semantic_search.embeddings import EmbeddingGenerator
from semantic_search.similarity import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
)


def interpret_score(score: float) -> str:
    """
    Convert cosine similarity score into human-readable meaning.
    """
    if score >= 0.9:
        return "Very Similar"
    elif score >= 0.7:
        return "Similar"
    elif score >= 0.5:
        return "Somewhat Similar"
    else:
        return "Not Similar"


class TextComparator:
    def __init__(self):
        self.embedder = EmbeddingGenerator()

    def compare(self, text1: str, text2: str) -> dict:
        """
        Compare two texts and return similarity metrics + interpretation.
        """

        # Generate embeddings
        vec1 = self.embedder.embed_single(text1)
        vec2 = self.embedder.embed_single(text2)

        # Similarity metrics
        cosine = cosine_similarity(vec1, vec2)
        euclidean = euclidean_distance(vec1, vec2)
        dot = dot_product(vec1, vec2)

        # Interpretation (✅ correct usage)
        interpretation = interpret_score(cosine)

        return {
            "cosine_similarity": float(cosine),
            "euclidean_distance": float(euclidean),
            "dot_product": float(dot),
            "interpretation": interpretation,
        }
