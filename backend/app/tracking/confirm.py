"""Tracker-independent confirmation, announcement, and expiry bookkeeping."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.tracking.base import Track


@dataclass(frozen=True, slots=True)
class RawTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[int, int]] | None
    score: float


class TrackConfirmer:
    """Apply one confirmation policy consistently across tracker backends."""

    def __init__(
        self,
        confirm_frames: int,
        ttl_frames: int,
        on_expire: Callable[[int], None],
    ) -> None:
        self._confirm_frames = max(1, int(confirm_frames))
        self._ttl_frames = max(1, int(ttl_frames))
        self._on_expire = on_expire
        self._consecutive: dict[int, int] = {}
        self._confirmed: set[int] = set()
        self._announced: set[int] = set()
        self._last_seen_frame: dict[int, int] = {}

    def observe(self, frame_index: int, raw: list[RawTrack]) -> list[Track]:
        current_ids = {item.track_id for item in raw}
        for track_id in current_ids:
            self._last_seen_frame[track_id] = frame_index

        expired_ids = {
            track_id
            for track_id, last_seen in self._last_seen_frame.items()
            if frame_index - last_seen > self._ttl_frames
        }
        for track_id in sorted(expired_ids):
            self._on_expire(track_id)
            self._last_seen_frame.pop(track_id, None)
            self._consecutive.pop(track_id, None)
            self._confirmed.discard(track_id)
            self._announced.discard(track_id)

        for track_id in list(self._consecutive):
            if track_id not in current_ids and track_id not in self._confirmed:
                self._consecutive.pop(track_id, None)

        results: list[Track] = []
        for item in raw:
            if item.track_id not in self._confirmed:
                count = self._consecutive.get(item.track_id, 0) + 1
                self._consecutive[item.track_id] = count
                if count < self._confirm_frames:
                    continue
                self._confirmed.add(item.track_id)
            results.append(
                Track(
                    track_id=item.track_id,
                    bbox=item.bbox,
                    polygon=item.polygon,
                    score=item.score,
                    is_new=item.track_id not in self._announced,
                )
            )
        return results

    def acknowledge_new_tracks(self, ids: list[int]) -> None:
        self._announced.update(
            int(track_id)
            for track_id in ids
            if int(track_id) in self._confirmed
        )

    def reset(self) -> None:
        self._consecutive.clear()
        self._confirmed.clear()
        self._announced.clear()
        self._last_seen_frame.clear()
