import numpy as np
from semantic_search.embeddings import EmbeddingGenerator


def test_single_embedding_shape():
    generator = EmbeddingGenerator()
    embedding = generator.embed_single("Hello world")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 384


def test_batch_embedding_shape():
    generator = EmbeddingGenerator()
    texts = ["Hello", "World", "Embeddings are cool"]

    embeddings = generator.embed_batch(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == 384


def test_same_text_same_embedding():
    generator = EmbeddingGenerator()

    emb1 = generator.embed_single("Consistency check")
    emb2 = generator.embed_single("Consistency check")

    assert np.allclose(emb1, emb2)


def test_benchmark_models():
    """
    Ensure multiple embedding models load correctly
    and return non-empty embedding vectors.
    """

    from semantic_search.embeddings import EmbeddingGenerator

    models = [
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
    ]

    text = "Test sentence for benchmarking"

    for model in models:
        embedder = EmbeddingGenerator(model)
        vec = embedder.embed_single(text)

        # Core guarantees
        assert vec is not None
        assert vec.ndim == 1
        assert vec.shape[0] > 0
