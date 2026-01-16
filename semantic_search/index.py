import numpy as np
from typing import List, Tuple

from semantic_search.embeddings import EmbeddingGenerator
from semantic_search.similarity import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
)

SIMILARITY_MAP = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_distance,
    "dot": dot_product,
}


class VectorIndex:
    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.documents: List[str] = []
        self.embeddings: np.ndarray | None = None

    def build(self, documents: List[str]):
        self.documents = documents
        self.embeddings = self.embedder.embed_batch(documents)

    def save(self, path: str):
        np.savez(
            path,
            embeddings=self.embeddings,
            documents=np.array(self.documents),
        )

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.documents = data["documents"].tolist()

    def search(
        self,
        query: str,
        top_k: int = 5,
        metric: str = "cosine",
    ) -> List[Tuple[str, float]]:

        if metric not in SIMILARITY_MAP:
            raise ValueError(f"Unsupported similarity metric: {metric}")

        query_vec = self.embedder.embed_single(query)

        scores = []
        for doc, vec in zip(self.documents, self.embeddings):
            score = SIMILARITY_MAP[metric](query_vec, vec)
            scores.append((doc, float(score)))

        reverse = metric in ("cosine", "dot")
        scores.sort(key=lambda x: x[1], reverse=reverse)

        return scores[:top_k]
