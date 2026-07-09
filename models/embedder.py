"""
embedder.py
------------
Wraps a SentenceTransformer model so the rest of the app never has to
think about model loading, batching, or device placement.

Usage:
    from models.embedder import Embedder

    embedder = Embedder()
    vec = embedder.encode("Experienced Python developer with 5 years in ML")
    vecs = embedder.encode_batch(["resume text 1", "resume text 2"])
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """
    Thin wrapper around a SentenceTransformer model.

    Default model: all-MiniLM-L6-v2
    - 384-dim embeddings, ~80MB, fast on CPU — good fit for a resume
      screener that needs to run without a GPU.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        """Encode a single string into a fixed-size vector."""
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: list) -> np.ndarray:
        """Encode a list of strings efficiently in one batch."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """
    Cached singleton so FastAPI/Streamlit don't reload the model
    (which takes a few seconds) on every request.
    """
    return Embedder()


if __name__ == "__main__":
    embedder = get_embedder()
    vec = embedder.encode("Data scientist skilled in Python and machine learning")
    print("Embedding shape:", vec.shape)
    print("First 5 values:", vec[:5])