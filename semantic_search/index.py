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
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.embedder = EmbeddingGenerator(model_name)
        self.documents: List[str] = []
        self.embeddings: np.ndarray | None = None

    def build(self, documents: List[str]):
        self.documents = documents
        self.embeddings = self.embedder.embed_batch(documents)

    def save(self, path):
        np.savez(
        path,
        embeddings=self.embeddings,
        documents=self.documents,
        model_name=self.model_name,
        )


    def load(self, path):
        data = np.load(path, allow_pickle=True)

        self.embeddings = data["embeddings"]
        self.documents = data["documents"].tolist()
        self.model_name = str(data["model_name"])


    def search(
        self,
        query: str,
        top_k: int = 5,
        metric: str = "cosine",
         threshold=None
    ) -> List[Tuple[str, float]]:

        if metric not in SIMILARITY_MAP:
            raise ValueError(f"Unsupported similarity metric: {metric}")
        
        if self.model_name != self.embedder.model_name:
            raise ValueError(
            f"Index was built with model '{self.model_name}', "
            f"but search is using '{self.embedder.model_name}'. "
            "Use the same model."
        )


        query_vec = self.embedder.embed_single(query)

        scores = []
        for doc, vec in zip(self.documents, self.embeddings):
            score = SIMILARITY_MAP[metric](query_vec, vec)
            scores.append((doc, float(score)))

        if threshold is not None and metric in ("cosine", "dot"):
            scores = [(doc, score) for doc, score in scores if score >= threshold]


        reverse = metric in ("cosine", "dot")
        scores.sort(key=lambda x: x[1], reverse=reverse)

        return scores[:top_k]
