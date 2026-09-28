"""Lightweight stage-tracing helper shared by the orchestrator and branches.

A ``Stage`` context manager emits paired ``start`` / ``done`` (or ``error``)
events through an ``emit`` callback and times the block, so a streaming UI can
show which method is running live. ``emit`` is any ``callable(dict)``; pass
``noop`` (the default everywhere) to disable tracing with zero overhead.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

EmitFn = Callable[[dict[str, Any]], None]


def noop(_event: dict[str, Any]) -> None:
    pass


class Stage:
    def __init__(self, emit: EmitFn, stage: str, label: str, method: str, group: str):
        self.emit = emit or noop
        self.stage, self.label, self.method, self.group = stage, label, method, group
        self.detail: dict[str, Any] = {}
        # Rich, click-to-inspect blocks: [{kind, title, items}]. Bounded by the caller.
        self.payload: list = []

    def __enter__(self):
        self._t0 = time.time()
        self.emit(
            {
                "type": "stage",
                "stage": self.stage,
                "label": self.label,
                "method": self.method,
                "group": self.group,
                "status": "start",
                "detail": {},
            }
        )
        return self

    def set(self, **detail):
        """Compact key/value summary shown as chips (always visible)."""
        self.detail.update(detail)

    def attach(self, kind: str, title: str, items: Any):
        """Rich payload shown when the user expands the stage.

        ``kind`` ∈ {"chunks", "relations", "entities", "text", "json"}; ``items``
        is the (already bounded) list/dict/str to render.
        """
        if items in (None, [], "", {}):
            return
        self.payload.append({"kind": kind, "title": title, "items": items})

    def __exit__(self, exc_type, exc, tb):
        self.emit(
            {
                "type": "stage",
                "stage": self.stage,
                "label": self.label,
                "method": self.method,
                "group": self.group,
                "status": "error" if exc_type else "done",
                "detail": ({"error": str(exc)} if exc_type else self.detail),
                "payload": ([] if exc_type else self.payload),
                "elapsed_s": round(time.time() - self._t0, 2),
            }
        )
        return False
