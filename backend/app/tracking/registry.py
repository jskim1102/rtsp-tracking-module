"""Tracker adapter registry.

Future names are deliberately reserved so UI/API slots remain stable, but attempting
to instantiate an unimplemented adapter fails explicitly.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from app.tracking.base import TrackerAdapter
from app.tracking.sam3_adapter import Sam3Adapter

_RESERVED = ("sam3", "motip", "storm", "samurai")
_FACTORIES: dict[str, Callable[..., TrackerAdapter]] = {"sam3": Sam3Adapter}


def reserved_trackers() -> list[str]:
    return list(_RESERVED)


def available_trackers() -> list[str]:
    return [name for name in _RESERVED if name in _FACTORIES]


def tracker_capabilities(name: str) -> dict[str, bool]:
    factory = _FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"tracker is not available: {name}")
    return dict(factory.capabilities)


def create_tracker(name: str | None = None, **kwargs: Any) -> TrackerAdapter:
    selected = name or os.getenv("TRACKER_DEFAULT") or "sam3"
    if selected not in _RESERVED:
        raise ValueError(f"unregistered tracker: {selected}")
    factory = _FACTORIES.get(selected)
    if factory is None:
        raise NotImplementedError(f"reserved tracker is not implemented: {selected}")
    return factory(**kwargs)
