"""Stateful tracker adapter public contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[int, int]] | None
    score: float
    is_new: bool


@runtime_checkable
class TrackerSession(Protocol):
    def update(self, frame: np.ndarray, ts: float) -> list[Track]: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class TrackerAdapter(Protocol):
    name: str
    capabilities: Mapping[str, bool]

    def create_session(self, camera_id: str, config: dict) -> TrackerSession: ...
