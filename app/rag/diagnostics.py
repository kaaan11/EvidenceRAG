"""Content-free request diagnostics; monotonic durations, never correctness scores."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import logging
from pathlib import Path
import time
import uuid

_current = ContextVar("evidencerag_diagnostics", default=None)
STAGES = ("queue", "model_load", "query_embedding", "retrieval", "generation", "verification")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def static_identity():
    folder = Path(__file__).parent
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.py"))}
    runtime = {}
    for package in ("numpy", "faiss-cpu", "transformers", "sentence-transformers", "fastapi"):
        try: runtime[package] = version(package)
        except PackageNotFoundError: runtime[package] = None
    return {"source_sha256": digest(sources), "prompt_sha256": sources["prompt.py"], "runtime": runtime}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def provenance(settings, store=None):
    identity = {**static_identity(), **settings.model_identity()}
    identity["configured_index_signature"] = identity["index_signature"]
    if store is not None: identity["index_signature"] = store.signature
    identity["index_operation_id"] = getattr(store, "operation_id", None)
    return {**identity, "identity_sha256": digest(identity)}


class Trace:
    def __init__(self, settings, store=None, clock=None):
        self.clock = clock or time.monotonic
        self.started = self.clock()
        self.request_id = uuid.uuid4().hex
        self.reason = None
        self.generation_invoked = False
        self.admission_state = "not_started"
        self.timings = dict.fromkeys(STAGES)
        self.provenance = provenance(settings, store)

    @contextmanager
    def phase(self, name):
        start = self.clock()
        try: yield
        finally:
            elapsed = max(0.0, (self.clock() - start) * 1000)
            self.timings[name] = (self.timings[name] or 0.0) + elapsed

    def snapshot(self):
        return {"schema_version": 1, "reason": self.reason or "request_error",
            "generation_invoked": self.generation_invoked, "request_id": self.request_id,
            "admission_state": self.admission_state,
            "timings_ms": {**self.timings, "total": max(0.0, (self.clock()-self.started)*1000)},
            "provenance": self.provenance}


def current(): return _current.get()


@contextmanager
def trace_scope(settings, store=None):
    trace = current()
    if trace is not None:
        yield trace
        return
    trace = Trace(settings, store)
    token = _current.set(trace)
    try: yield trace
    finally: _current.reset(token)


@contextmanager
def phase(name):
    trace = current()
    if trace is None: yield
    else:
        with trace.phase(name): yield


def mark_reason(reason):
    if current() is not None: current().reason = reason


def mark_admission(state):
    if current() is not None: current().admission_state = state


def invoke_generator(function, *args, **kwargs):
    if current() is not None: current().generation_invoked = True
    with phase("generation"):
        try: return function(*args, **kwargs)
        except Exception:
            mark_reason("model_error")
            raise


class DiagnosticsMiddleware:
    """Enrich only /ask JSON responses, including pre-handler validation errors."""
    def __init__(self, app, pipeline):
        self.app, self.pipeline = app, pipeline

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/ask":
            return await self.app(scope, receive, send)
        pipeline = self.pipeline()
        with trace_scope(pipeline.settings, pipeline.vector_store) as trace:
            start = None
            body = []
            async def enriched(message):
                nonlocal start
                if message["type"] == "http.response.start":
                    start = message
                elif message["type"] == "http.response.body":
                    body.append(message.get("body", b""))
                    if not message.get("more_body", False):
                        data = json.loads(b"".join(body))
                        if trace.reason is None:
                            trace.reason = {200:"answered", 413:"input_limit", 503:"busy"}.get(start["status"], "request_error")
                        data["diagnostics"] = trace.snapshot()
                        encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
                        headers = [(k,v) for k,v in start["headers"] if k.lower() != b"content-length"]
                        await send({**start,"headers":headers+[(b"content-length",str(len(encoded)).encode())]})
                        await send({**message,"body":encoded})
            try:
                await self.app(scope, receive, enriched)
            except asyncio.CancelledError:
                if trace.reason is None: trace.reason = "cancelled"
                raise
            except Exception:
                # Preserve HTTP failure without reflecting exception content/paths.
                trace.reason = trace.reason or "request_error"
                encoded = json.dumps({"detail":"Request processing failed", "diagnostics":trace.snapshot()}).encode()
                await send({"type":"http.response.start","status":500,"headers":[(b"content-type",b"application/json"),(b"content-length",str(len(encoded)).encode())]})
                await send({"type":"http.response.body","body":encoded})
            finally:
                result = trace.snapshot()
                # Deliberately exclude provenance, exceptions, paths and all content.
                logger.info("request_diagnostics %s", json.dumps({k:result[k] for k in
                    ("request_id","reason","generation_invoked","admission_state","timings_ms")},sort_keys=True))
