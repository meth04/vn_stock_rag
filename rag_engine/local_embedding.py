from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import DEVICE, EMBEDDING_MODEL_ID


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)
