import numpy as np
from fastembed import TextEmbedding
from typing import List
from rich.progress import Progress


class EmbeddingGenerator:
    """
    Responsible for converting text into numerical embeddings.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self.dimensions = None

    def _load_model(self):
        """
        Lazily load the embedding model.
        This ensures the model is loaded only once.
        """
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Convert a single text into an embedding vector.
        """
        self._load_model()

        embedding_generator = self._model.embed([text])
        embedding = list(embedding_generator)[0]

        embedding_array = np.array(embedding, dtype=np.float32)

        if self.dimensions is None:
            self.dimensions = embedding_array.shape[0]

        return embedding_array

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into a matrix of embeddings with a Rich progress bar.
        """
        self._load_model()

        # Step 1: Embed all texts in one batch (fast)
        embeddings = list(self._model.embed(texts))
        embedding_matrix = np.array(embeddings, dtype=np.float32)

        # Step 2: Show progress bar for UX (simulated over all texts)
        with Progress() as progress:
            task = progress.add_task("[green]Embedding texts...", total=len(texts))
            for _ in texts:
                progress.update(task, advance=1)

        if self.dimensions is None:
            self.dimensions = embedding_matrix.shape[1]

        return embedding_matrix