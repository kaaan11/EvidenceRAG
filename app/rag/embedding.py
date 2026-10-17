from .runtime import SingleFlightCache, configure_torch_threads, get_runtime
from .config import effective_revision, model_manifest
import threading

import numpy as np


_cache = SingleFlightCache(maxsize=2)
_inference_lock = threading.Lock()


def get_embedding_model(model_name="intfloat/multilingual-e5-small", revision=None):
    revision = effective_revision("embedding",model_name,revision)
    return get_runtime().run_sync("load", _cache.get, (model_name, revision),
        lambda: _create_embedding_model(model_name, revision))


def _create_embedding_model(model_name, revision):
    # Imported lazily: sentence-transformers pulls in torch, which takes seconds to load.
    from sentence_transformers import SentenceTransformer
    import torch
    configure_torch_threads(torch)

    model = SentenceTransformer(model_name, revision=revision,
        processor_kwargs={"revision":revision},config_kwargs={"revision":revision})
    record=model_manifest()["embedding"]
    if (model_name,revision)==(record["repository"],record["revision"]):
        if model.get_sentence_embedding_dimension()!=record["embedding_dimension"]:
            raise ValueError("Loaded embedding dimension differs from pinned model metadata")
        if model.max_seq_length!=record["max_sequence_length"]:
            raise ValueError("Loaded embedding sequence limit differs from pinned metadata")
    return model


get_embedding_model.cache_clear = _cache.clear


def embed_texts(texts, model_name="intfloat/multilingual-e5-small", revision=None):
    return get_runtime().run_sync("embedding", _embed_texts, texts, model_name, revision)


def _embed_texts(texts, model_name, revision):
    if not texts:
        return np.empty((0, 0), dtype="float32")

    model = get_embedding_model(model_name,revision=revision)
    with _inference_lock:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    embeddings=np.asarray(embeddings,dtype="float32")
    if embeddings.ndim!=2 or embeddings.shape!=(len(texts),model.get_sentence_embedding_dimension()):
        raise ValueError("Embedding output dimensions differ from the loaded model metadata")
    return embeddings
