from typing import List, Tuple
import numpy as np

from semantic_search.embeddings import EmbeddingGenerator
from semantic_search.similarity import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
)

# Map metric name → function
SIMILARITY_MAP = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_distance,
    "dot": dot_product,
}


def semantic_search(query: str,
    documents: List[str],
    top_k: int = 5,
    metric: str = "cosine",
    model_name: str = "BAAI/bge-small-en-v1.5",
    threshold: float | None = None,) -> List[Tuple[str, float]]:
    """
    Perform semantic search over a list of documents.
    
    Args:
        query: User query string
        documents: List of candidate documents to search
        top_k: How many top results to return
        metric: Similarity metric ("cosine", "euclidean", "dot")
        
    Returns:
        List of tuples (document, score) sorted by score descending
    """

    # Check metric is supported
    if metric not in SIMILARITY_MAP:
        raise ValueError(f"Unsupported similarity metric: {metric}")

    # ✅ Use our wrapper class from Step 1
    embedder = EmbeddingGenerator(model_name)

    # Embed the query
    query_vec = embedder.embed_single(query)

    # Embed all documents (batch)
    doc_vecs = embedder.embed_batch(documents)

    # Calculate similarity scores
    scores = []
    for doc, vec in zip(documents, doc_vecs):
        score = SIMILARITY_MAP[metric](query_vec, vec)
        scores.append((doc, float(score)))

    # Apply threshold (only for similarity metrics)
    if threshold is not None and metric in ("cosine", "dot"):
        scores = [(doc, score) for doc, score in scores if score >= threshold]


    # Sort by score descending for "cosine" and "dot", ascending for "euclidean"
    reverse = metric in ("cosine", "dot")
    scores.sort(key=lambda x: x[1], reverse=reverse)

    # Return top_k results
    return scores[:top_k]
