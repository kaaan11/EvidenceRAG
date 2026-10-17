from .config import InputLimitError
from .runtime import SingleFlightCache, configure_torch_threads, get_runtime
import threading


class LLMGenerationError(RuntimeError):
    """Raised when the configured local language model cannot answer."""


_cache = SingleFlightCache(maxsize=2)
_inference_lock = threading.Lock()


def get_text_generator(model_name, revision=None):
    return get_runtime().run_sync("load", _cache.get, (model_name, revision),
        lambda: _create_text_generator(model_name, revision))


def _create_text_generator(model_name, revision):
    import torch
    from transformers import pipeline
    configure_torch_threads(torch)

    # Half-precision weights are several times slower on CPUs without native bf16 support.
    dtype = "auto" if torch.cuda.is_available() else torch.float32
    kwargs = {"revision": revision} if revision is not None else {}
    generator = pipeline("text-generation", model=model_name, dtype=dtype, **kwargs)
    # Answers should be reproducible, so drop the model's default sampling settings
    # instead of overriding them per call (which makes transformers warn on every request).
    defaults = generator.model.generation_config
    defaults.do_sample = False
    defaults.temperature = None
    defaults.top_p = None
    defaults.top_k = None
    return generator


get_text_generator.cache_clear = _cache.clear


def validate_prompt_budget(messages, tokenizer, context_window, max_new_tokens):
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    length = len(tokens)
    if length + max_new_tokens > context_window:
        raise InputLimitError(f"Prompt uses {length} tokens plus {max_new_tokens} answer tokens; context limit is {context_window}")
    return length


def _context_window(generator):
    config = getattr(getattr(generator, "model", None), "config", None)
    candidates = [getattr(config, key, None) for key in ["max_position_embeddings", "n_positions"]]
    tokenizer_limit = getattr(getattr(generator, "tokenizer", None), "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 2**31:
        candidates.append(tokenizer_limit)
    candidates = [v for v in candidates if isinstance(v, int) and v > 0]
    if not candidates:
        raise LLMGenerationError("Model/tokenizer context limit is unavailable")
    return min(candidates)


def generate_answer(
    messages,
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=256,
    text_generator=None,
):
    return get_runtime().run_sync("generation", _generate_answer, messages, model_name, max_new_tokens, text_generator)


def _generate_answer(messages, model_name, max_new_tokens, text_generator):
    try:
        from transformers import GenerationConfig

        generator = text_generator or get_text_generator(model_name)
        tokenizer = getattr(generator, "tokenizer", None)
        if tokenizer is None:
            raise LLMGenerationError("Tokenizer is required for the complete prompt budget")
        with _inference_lock:
            validate_prompt_budget(messages, tokenizer, _context_window(generator), max_new_tokens)
            output = generator(
                messages,
                generation_config=GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False),
                return_full_text=False,
                clean_up_tokenization_spaces=False,
            )
        return output[0]["generated_text"].strip()
    except InputLimitError:
        raise
    except Exception as exc:
        raise LLMGenerationError(f"LLM generation failed: {exc}") from exc
