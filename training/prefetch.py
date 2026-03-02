from __future__ import annotations

import queue
import threading
from typing import Iterator


class CPUPrefetchLoader:
    """Threaded CPU-side prefetch wrapper for an existing DataLoader.

    This overlaps main-thread training work with DataLoader iteration by
    queueing a small number of already-collated batches ahead of time.
    """

    _SENTINEL = object()

    def __init__(self, loader, prefetch: int = 2):
        self.loader = loader
        self.prefetch = max(1, int(prefetch))

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator:
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)

        def _producer() -> None:
            try:
                for batch in self.loader:
                    q.put(batch)
            finally:
                q.put(self._SENTINEL)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is self._SENTINEL:
                break
            yield item
