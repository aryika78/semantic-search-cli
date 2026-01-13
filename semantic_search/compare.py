from semantic_search.embeddings import EmbeddingGenerator
from semantic_search.similarity import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
)


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

        # Interpretation
        interpretation = self._interpret_cosine(cosine)

        return {
            "cosine_similarity": float(cosine),
            "euclidean_distance": float(euclidean),
            "dot_product": float(dot),
            "interpretation": interpretation,
        }

    def _interpret_cosine(self, score: float) -> str:
        """
        Convert cosine score into human meaning.
        """
        if score >= 0.85:
            return "Very Similar"
        elif score >= 0.65:
            return "Similar"
        elif score >= 0.45:
            return "Somewhat Related"
        else:
            return "Not Related"
