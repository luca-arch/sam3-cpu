"""Asynchronous I/O worker for overlapping GPU compute with disk writes.

Provides a background thread pool for writing mask images and metadata
to disk while the GPU processes the next prompt or chunk.  This eliminates
I/O-induced GPU idle time between prompts.

Design
------
- Single worker thread (``ThreadPoolExecutor(max_workers=1)``) to
  prevent disk thrashing from concurrent writes.
- Thread-safe submission via ``submit()`` — accepts any callable.
- ``drain()`` blocks until all pending writes complete.
- Stats tracking for metadata (files count, errors, wall time).

Thread safety
-------------
Mask data passed to the worker must already be on CPU (numpy or plain
Python objects).  The caller can safely ``del`` its local reference
after submission — the worker thread holds its own reference via the
closure, keeping the data alive until writing completes.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class AsyncIOWorker:
    """Single-thread background I/O worker.

    Usage::

        worker = AsyncIOWorker()

        # Submit I/O tasks (returns immediately)
        worker.submit(save_masks, result, obj_ids, masks_dir, w, h, n)
        worker.submit(write_json, path, data)

        # ... GPU continues on next prompt ...

        # Wait for all writes before stitching/shutdown
        worker.drain()
        worker.shutdown()
    """

    def __init__(self, max_workers: int = 1):
        self._max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: List[Future] = []
        self._started = False
        self._files_enqueued: int = 0
        self._errors: int = 0
        self._wall_start: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the worker thread pool."""
        if self._started:
            return
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="sam3-io",
        )
        self._started = True
        self._wall_start = time.time()

    def shutdown(self) -> None:
        """Drain pending work and shut down the thread pool."""
        self.drain()
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._started = False

    # ── Submission ───────────────────────────────────────────────────────

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        """Submit an arbitrary callable to the worker thread.

        Returns a ``Future`` that can be inspected later.  Errors are
        collected and reported by ``drain()``.
        """
        if not self._started or self._executor is None:
            raise RuntimeError("AsyncIOWorker not started.  Call start() first.")
        future = self._executor.submit(fn, *args, **kwargs)
        self._futures.append(future)
        self._files_enqueued += 1
        return future

    # ── Synchronisation ──────────────────────────────────────────────────

    def drain(self) -> int:
        """Block until all submitted tasks complete.

        Returns the number of tasks that raised exceptions.
        """
        errors = 0
        for f in self._futures:
            try:
                f.result()
            except Exception as exc:
                errors += 1
                self._errors += 1
                logger.warning("AsyncIO task failed: %s", exc)
        self._futures.clear()
        return errors

    @property
    def pending(self) -> int:
        """Number of tasks still in flight."""
        return sum(1 for f in self._futures if not f.done())

    # ── Stats ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return stats for metadata export."""
        wall = time.time() - self._wall_start if self._wall_start else 0
        return {
            "tasks_submitted": self._files_enqueued,
            "errors": self._errors,
            "pending": self.pending,
            "wall_time_s": round(wall, 3),
        }
