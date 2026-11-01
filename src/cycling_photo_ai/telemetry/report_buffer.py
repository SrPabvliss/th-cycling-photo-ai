"""Bounded, thread-safe buffer of reports waiting to reach yupana."""

from __future__ import annotations

import threading
from collections import deque


class ReportBuffer:
    def __init__(self, max_size: int = 5000) -> None:
        self._max_size = max_size
        self._items: deque[tuple[dict, bool]] = deque()
        self._lock = threading.Lock()
        self._dropped = 0
        self._protected_ids: set[str] = set()

    def add(self, report: dict, protected: bool = False) -> None:
        with self._lock:
            if protected:
                self._protected_ids.add(report["report_id"])
            self._items.append((report, protected))
            self._enforce_bound()

    def take(self, max_items: int) -> list[dict]:
        with self._lock:
            taken: list[dict] = []
            while self._items and len(taken) < max_items:
                taken.append(self._items.popleft()[0])
            return taken

    def return_front(self, reports: list[dict]) -> None:
        with self._lock:
            for report in reversed(reports):
                self._items.appendleft((report, report["report_id"] in self._protected_ids))
            self._enforce_bound()

    @property
    def dropped(self) -> int:
        return self._dropped

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def _enforce_bound(self) -> None:
        while len(self._items) > self._max_size:
            victim = next(
                (i for i, (_, protected) in enumerate(self._items) if not protected), None
            )
            if victim is None:
                return
            del self._items[victim]
            self._dropped += 1
