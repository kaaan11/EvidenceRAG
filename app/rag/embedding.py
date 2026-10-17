from .runtime import SingleFlightCache, configure_torch_threads, get_runtime
import threading

import numpy as np


_cache = SingleFlightCache(maxsize=2)
_inference_lock = threading.Lock()


def get_embedding_model(model_name="intfloat/multilingual-e5-small", revision=None):
    return get_runtime().run_sync("load", _cache.get, (model_name, revision),
        lambda: _create_embedding_model(model_name, revision))


def _create_embedding_model(model_name, revision):
    # Imported lazily: sentence-transformers pulls in torch, which takes seconds to load.
    from sentence_transformers import SentenceTransformer
    import torch
    configure_torch_threads(torch)

    kwargs = {"revision": revision} if revision is not None else {}
    return SentenceTransformer(model_name, **kwargs)


get_embedding_model.cache_clear = _cache.clear


def embed_texts(texts, model_name="intfloat/multilingual-e5-small"):
    return get_runtime().run_sync("embedding", _embed_texts, texts, model_name)


def _embed_texts(texts, model_name):
    if not texts:
        return np.empty((0, 0), dtype="float32")

    model = get_embedding_model(model_name)
    with _inference_lock:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return np.asarray(embeddings, dtype="float32")
