"""Finite process-local admission; permits belong to actual native work."""
import asyncio
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from contextvars import copy_context
from .diagnostics import phase, mark_admission, mark_reason


class BusyError(RuntimeError):
    pass


class HeavyRuntime:
    def __init__(self, capacity=1, max_waiting=2, per_kind=2, wait_seconds=30):
        if capacity <= 0 or max_waiting < 0 or wait_seconds <= 0:
            raise ValueError("Runtime capacity/wait must be positive and queue nonnegative")
        self.capacity = capacity
        self.max_waiting = max_waiting
        self.per_kind = per_kind
        self.wait_seconds = wait_seconds
        self.active = 0
        self.accepting = True
        self.queue = deque()
        self.condition = threading.Condition()
        self.local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix="evidencerag-heavy")

    def _ticket(self, kind):
        if not self.accepting:
            raise BusyError("Model service is draining")
        if self.active < self.capacity and not self.queue:
            self.active += 1
            return None
        kind_count = sum(k == kind for token, k in self.queue)
        limit = self.per_kind.get(kind, 2) if isinstance(self.per_kind, dict) else self.per_kind
        if len(self.queue) >= self.max_waiting or kind_count >= limit:
            raise BusyError("Heavy-job queue is full; retry later")
        ticket = (object(), kind)
        self.queue.append(ticket)
        return ticket

    def _claim(self, ticket):
        if not self.accepting:
            raise BusyError("Model service is draining")
        if self.active < self.capacity and self.queue and self.queue[0] == ticket:
            self.queue.popleft()
            self.active += 1
            return True
        return False

    def _remove(self, ticket):
        if ticket in self.queue:
            self.queue.remove(ticket)
            self.condition.notify_all()

    def _acquire(self, kind):
        deadline = time.monotonic() + self.wait_seconds
        with self.condition:
            ticket = self._ticket(kind)
            if ticket is None:
                return
            try:
                while not self._claim(ticket):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise BusyError("Heavy-job queue wait expired")
                    self.condition.wait(remaining)
            finally:
                self._remove(ticket)

    async def _acquire_async(self, kind):
        deadline = time.monotonic() + self.wait_seconds
        with self.condition:
            ticket = self._ticket(kind)
        if ticket is None:
            return
        try:
            while True:
                with self.condition:
                    if self._claim(ticket):
                        return
                if time.monotonic() >= deadline:
                    raise BusyError("Heavy-job queue wait expired")
                await asyncio.sleep(min(0.01, self.wait_seconds))
        finally:
            with self.condition:
                self._remove(ticket)

    def _release(self):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def _work(self, function, args, kwargs):
        self.local.owned = True
        try:
            return function(*args, **kwargs)
        finally:
            self.local.owned = False
            self._release()

    def run_sync(self, kind, function, *args, **kwargs):
        if getattr(self.local, "owned", False):
            return function(*args, **kwargs)
        with phase("queue"):
            try: self._acquire(kind)
            except BusyError:
                mark_admission("queue_rejected"); mark_reason("busy")
                raise
        mark_admission("running")
        return self._work(function, args, kwargs)

    async def run_async(self, kind, function, *args, **kwargs):
        with phase("queue"):
            try: await self._acquire_async(kind)
            except asyncio.CancelledError:
                mark_admission("queue_cancelled"); mark_reason("cancelled")
                raise
            except BusyError:
                mark_admission("queue_rejected"); mark_reason("busy")
                raise
        mark_admission("running")
        try:
            context = copy_context()
            future = asyncio.get_running_loop().run_in_executor(self.executor, context.run, self._work, function, args, kwargs)
        except BaseException:
            self._release()
            raise
        cancelled = False
        # Cancellation is not native-thread termination. Retain request-owned files
        # and the worker-owned permit until completion, including repeated cancel.
        while True:
            try:
                result = await asyncio.shield(future)
                break
            except asyncio.CancelledError:
                cancelled = True
                mark_admission("client_cancelled_running"); mark_reason("cancelled")
            except BaseException:
                if cancelled:
                    mark_reason("cancelled")
                    raise asyncio.CancelledError() from None
                raise
        if cancelled:
            mark_reason("cancelled")
            raise asyncio.CancelledError()
        return result

    async def drain(self):
        with self.condition:
            self.accepting = False
            self.condition.notify_all()
        while True:
            with self.condition:
                if self.active == 0:
                    return
            await asyncio.sleep(0.01)

    def snapshot(self):
        with self.condition:
            return {"active": self.active, "waiting": len(self.queue), "accepting": self.accepting}


class SingleFlightCache:
    def __init__(self, maxsize=2):
        self.lock = threading.Lock()
        self.values = OrderedDict()
        self.ready = frozenset()
        self.maxsize = maxsize

    def get(self, key, factory):
        with self.lock:
            if key not in self.values:
                with phase("model_load"):
                    try: value = factory()  # Publish only success; failed initialization can retry.
                    except Exception:
                        mark_reason("model_error")
                        raise
                self.values[key] = value
                if len(self.values) > self.maxsize:
                    self.values.popitem(last=False)
                self.ready = frozenset(self.values)
            else:
                self.values.move_to_end(key)
            return self.values[key]

    def clear(self):
        with self.lock:
            self.values.clear()
            self.ready = frozenset()


_default = None
_default_lock = threading.Lock()


def get_runtime():
    global _default
    with _default_lock:
        if _default is None:
            from .config import Settings
            settings = Settings()
            _default = HeavyRuntime(settings.max_active_heavy_jobs, settings.max_waiting_heavy_jobs,
                {"ask": settings.ask_queue_capacity, "index": settings.index_queue_capacity,
                 "generation": settings.model_queue_capacity, "embedding": settings.model_queue_capacity,
                 "load": settings.model_queue_capacity}, settings.heavy_wait_seconds)
        return _default


_torch_config_lock = threading.Lock()
_torch_configured = False


def configure_torch_threads(torch):
    global _torch_configured
    with _torch_config_lock:
        if not _torch_configured:
            from .config import Settings
            count = Settings().torch_num_threads
            if count is not None:
                torch.set_num_threads(count)
            _torch_configured = True
