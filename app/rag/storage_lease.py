"""OS-owned process writer lease; lockfile existence/PID is not ownership."""
import os
from pathlib import Path
import threading


class WriterLeaseError(RuntimeError):
    pass


_registry = {}
_registry_lock = threading.Lock()


class WriterLease:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.file = None

    @property
    def held(self): return self.file is not None

    def acquire(self):
        with _registry_lock:
            if self.directory in _registry:
                raise WriterLeaseError("Index directory already has a service writer")
            self.directory.mkdir(parents=True, exist_ok=True)
            file = open(self.directory / ".writer.lock", "a+b")
            try:
                if os.fstat(file.fileno()).st_size == 0:
                    file.write(b"\0"); file.flush()
                file.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                file.close()
                raise WriterLeaseError("Index directory already has a service writer") from exc
            self.file = file
            _registry[self.directory] = self  # Strong reference through cancelled lifespan.
        return self

    def require_held(self):
        if not self.held:
            raise WriterLeaseError("Service writer lease is not held; start the service lifespan first")

    def release(self):
        with _registry_lock:
            if self.file is None: return
            self.file.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            finally:
                self.file.close(); self.file = None
                _registry.pop(self.directory, None)

    def drain_and_release(self, runtime):
        return drain_service(runtime, self)


def drain_service(runtime, lease=None):
    # This watcher is submitted before shutdown awaits. Cancelling that await
    # cannot stop the thread or release the OS lock before native work ends.
    with runtime.condition:
        runtime.accepting = False
        runtime.condition.notify_all()
        while runtime.active or runtime.queue:
            runtime.condition.wait(0.05)
    if lease is not None: lease.release()
    with runtime.condition:
        runtime.accepting = True  # Standalone library use remains available.
        runtime.condition.notify_all()
