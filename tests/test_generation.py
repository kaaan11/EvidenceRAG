import pytest

from app.rag.generation import LLMGenerationError, generate_answer
from types import SimpleNamespace


def attach_tokenizer(generator):
    generator.tokenizer = SimpleNamespace(model_max_length=1024,
        apply_chat_template=lambda messages, **kwargs: [1, 2, 3])
    return generator


def test_generate_answer_returns_only_new_text_deterministically():
    calls = []

    def fake_generator(messages, **kwargs):
        calls.append((messages, kwargs))
        return [{"generated_text": "  Cevap [Kanıt 1]  "}]

    messages = [{"role": "user", "content": "Soru?"}]
    answer = generate_answer(messages, max_new_tokens=32, text_generator=attach_tokenizer(fake_generator))

    assert answer == "Cevap [Kanıt 1]"
    assert len(calls) == 1
    called_messages, kwargs = calls[0]
    assert called_messages == messages
    assert kwargs["generation_config"].max_new_tokens == 32
    assert kwargs["generation_config"].do_sample is False
    assert kwargs["return_full_text"] is False


def test_generate_answer_wraps_model_errors():
    def broken_generator(messages, **kwargs):
        raise RuntimeError("model unavailable")

    with pytest.raises(LLMGenerationError, match="model unavailable"):
        generate_answer([], text_generator=attach_tokenizer(broken_generator))


@pytest.mark.parametrize("prompt,reserve,limit,allowed", [(3,2,6,True),(4,2,6,True),(5,2,6,False)])
def test_prompt_token_budget_includes_answer_reserve(prompt, reserve, limit, allowed):
    from app.rag.generation import validate_prompt_budget
    from app.rag.config import InputLimitError
    seen = []
    messages = [{"role":"system","content":"System"},{"role":"user","content":"Evidence and full question"}]
    tokenizer = SimpleNamespace(apply_chat_template=lambda value,**kwargs: seen.append((value,kwargs)) or [1]*prompt)
    if allowed:
        assert validate_prompt_budget(messages,tokenizer,limit,reserve)==prompt
    else:
        with pytest.raises(InputLimitError): validate_prompt_budget(messages,tokenizer,limit,reserve)
    assert seen==[(messages,{"tokenize":True,"add_generation_prompt":True})]


def test_prompt_overflow_never_invokes_generator():
    from app.rag.config import InputLimitError
    calls=[]
    def generator(*args,**kwargs): calls.append(args)
    generator.tokenizer=SimpleNamespace(model_max_length=5,apply_chat_template=lambda *args,**kwargs:[1]*4)
    with pytest.raises(InputLimitError): generate_answer([{"role":"user","content":"unchanged"}],max_new_tokens=2,text_generator=generator)
    assert calls==[]


def test_missing_tokenizer_or_context_is_not_a_fake_success():
    with pytest.raises(LLMGenerationError,match="Tokenizer is required"):
        generate_answer([],text_generator=lambda *args,**kwargs: [])


def test_context_limit_is_derived_from_real_generator_metadata():
    from app.rag.generation import _context_window
    generator=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192)),
        tokenizer=SimpleNamespace(model_max_length=4096))
    assert _context_window(generator)==4096
    generator.tokenizer.model_max_length=10**30
    assert _context_window(generator)==8192


def test_generation_mutex_remains_held_until_native_return(monkeypatch):
    import app.rag.generation as module
    from app.rag.runtime import HeavyRuntime
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    runtime=HeavyRuntime(capacity=2,max_waiting=2,wait_seconds=5)
    monkeypatch.setattr(module,"get_runtime",lambda:runtime)
    barrier=threading.Barrier(3); entered=threading.Event(); release=threading.Event(); calls=[]
    def generator(*args,**kwargs):
        calls.append(1); entered.set(); assert release.wait(5)
        return [{"generated_text":"Answer"}]
    attach_tokenizer(generator)
    def run():
        barrier.wait(5)
        return module.generate_answer([],text_generator=generator)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run) for _ in range(2)]; barrier.wait(5)
        try:
            assert entered.wait(5)
            deadline=time.monotonic()+5
            while runtime.snapshot()["active"]<2 and time.monotonic()<deadline: time.sleep(0.005)
            assert runtime.snapshot()["active"]==2 and len(calls)==1
        finally: release.set()
        assert [f.result(5) for f in futures]==["Answer","Answer"]
    assert len(calls)==2 and runtime.snapshot()["active"]==0
