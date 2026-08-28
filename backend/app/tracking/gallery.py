"""In-memory first-seen track gallery with ephemeral JPEG crops."""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from app import config
from app.inference.worker import InferenceResult

logger = logging.getLogger("rtsp-tracking.tracking.gallery")
_OWNED_CROP_NAME = re.compile(r"[0-9a-f]{16}--?\d+\.jpg")


@dataclass(slots=True)
class GalleryItem:
    track_id: int
    camera_id: str
    first_seen: float
    thumb_url: str
    active: bool
    thumb_path: Path

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "first_seen": self.first_seen,
            "thumb_url": self.thumb_url,
            "active": self.active,
        }


@dataclass(slots=True)
class GalleryObservation:
    events: list[dict[str, int | float | str]]
    acknowledged_track_ids: list[int]
    created_track_ids: list[int]


class TrackGallery:
    """Process-local gallery; camera IDs form independent track namespaces."""

    def __init__(
        self,
        storage_dir: str | Path | None = None,
        *,
        max_items: int | None = None,
    ) -> None:
        self._temp_dir = None
        if storage_dir is None:
            storage_dir = config.TRACK_GALLERY_DIR or None
            if storage_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory(
                    prefix="rtsp-tracking-tracks-"
                )
                storage_dir = self._temp_dir.name
        self.storage_dir = Path(storage_dir)
        self.max_items = max(
            1,
            int(
                config.TRACK_GALLERY_MAX_ITEMS
                if max_items is None
                else max_items
            ),
        )
        self._items: dict[tuple[str, int], GalleryItem] = {}
        self._lock = threading.RLock()
        if self._temp_dir is None:
            self._sweep_owned_crops()

    @staticmethod
    def _thumb_url(camera_id: str, track_id: int) -> str:
        # track_id is camera-scoped, so the query disambiguates equal IDs while
        # preserving the documented /api/tracks/{track_id}/thumb route.
        return (
            f"/api/tracks/{track_id}/thumb?camera_id="
            f"{quote(camera_id, safe='')}"
        )

    def _thumb_path(self, camera_id: str, track_id: int) -> Path:
        camera_key = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()[:16]
        return self.storage_dir / f"{camera_key}-{track_id}.jpg"

    def _unlink_locked(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.exception("track crop cleanup failed: %s", path)

    def _sweep_owned_crops(self) -> None:
        """Remove orphaned crops from an explicitly persistent gallery path."""
        if not self.storage_dir.exists():
            return
        try:
            candidates = list(self.storage_dir.iterdir())
        except OSError:
            logger.exception("track gallery startup sweep failed: %s", self.storage_dir)
            return
        with self._lock:
            for path in candidates:
                if _OWNED_CROP_NAME.fullmatch(path.name) and not path.is_dir():
                    self._unlink_locked(path)

    def _evict_oldest_locked(self) -> None:
        while len(self._items) >= self.max_items:
            oldest_key, oldest = min(
                self._items.items(),
                key=lambda pair: (
                    pair[1].first_seen,
                    pair[1].camera_id,
                    pair[1].track_id,
                ),
            )
            self._items.pop(oldest_key, None)
            self._unlink_locked(oldest.thumb_path)

    def observe_with_ack(self, result: InferenceResult) -> GalleryObservation:
        """Persist first-seen crops and return the IDs safe to acknowledge."""
        camera_id = result.source_id
        current_ids = {track.track_id for track in result.tracks}
        events: list[dict[str, int | float | str]] = []
        acknowledged_track_ids: list[int] = []
        created_track_ids: list[int] = []
        with self._lock:
            for (item_camera, item_track_id), item in self._items.items():
                if item_camera == camera_id:
                    item.active = item_track_id in current_ids

            for track in result.tracks:
                key = (camera_id, track.track_id)
                existing = self._items.get(key)
                if existing is not None:
                    existing.active = True
                    if track.is_new:
                        acknowledged_track_ids.append(track.track_id)
                    continue
                if not track.is_new:
                    continue
                crop = result.new_track_crops.get(track.track_id)
                if not isinstance(crop, np.ndarray) or crop.size == 0:
                    logger.warning(
                        "new track crop missing: camera=%s track_id=%s",
                        camera_id,
                        track.track_id,
                    )
                    continue
                try:
                    encoded, jpeg = cv2.imencode(".jpg", crop)
                except cv2.error:
                    logger.exception(
                        "new track crop encode failed: camera=%s track_id=%s",
                        camera_id,
                        track.track_id,
                    )
                    continue
                if not encoded:
                    logger.warning(
                        "new track crop encode failed: camera=%s track_id=%s",
                        camera_id,
                        track.track_id,
                    )
                    continue
                thumb_path = self._thumb_path(camera_id, track.track_id)
                try:
                    self.storage_dir.mkdir(parents=True, exist_ok=True)
                    thumb_path.write_bytes(jpeg.tobytes())
                except OSError:
                    logger.exception(
                        "new track crop write failed: camera=%s track_id=%s path=%s",
                        camera_id,
                        track.track_id,
                        thumb_path,
                    )
                    self._unlink_locked(thumb_path)
                    continue
                self._evict_oldest_locked()
                thumb_url = self._thumb_url(camera_id, track.track_id)
                item = GalleryItem(
                    track_id=track.track_id,
                    camera_id=camera_id,
                    first_seen=float(result.timestamp),
                    thumb_url=thumb_url,
                    active=True,
                    thumb_path=thumb_path,
                )
                self._items[key] = item
                acknowledged_track_ids.append(track.track_id)
                created_track_ids.append(track.track_id)
                events.append(
                    {
                        "type": "new_track",
                        "track_id": track.track_id,
                        "camera_id": camera_id,
                        "first_seen": item.first_seen,
                        "thumb_url": thumb_url,
                    }
                )
            retained_track_ids = {
                item_track_id
                for item_camera, item_track_id in self._items
                if item_camera == camera_id
            }
            acknowledged_track_ids = [
                track_id
                for track_id in acknowledged_track_ids
                if track_id in retained_track_ids
            ]
            created_track_ids = [
                track_id
                for track_id in created_track_ids
                if track_id in retained_track_ids
            ]
            events = [
                event
                for event in events
                if int(event["track_id"]) in retained_track_ids
            ]
        return GalleryObservation(
            events=events,
            acknowledged_track_ids=acknowledged_track_ids,
            created_track_ids=created_track_ids,
        )

    def observe(self, result: InferenceResult) -> list[dict[str, int | float | str]]:
        """Compatibility wrapper returning only newly-created WS events."""
        return self.observe_with_ack(result).events

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            items = sorted(
                self._items.values(),
                key=lambda item: (item.first_seen, item.camera_id, item.track_id),
                reverse=True,
            )
            return {
                "active": sum(item.active for item in items),
                "total": len(items),
                "items": [item.as_dict() for item in items],
            }

    def thumbnail(self, track_id: int, camera_id: str | None = None) -> Path | None:
        with self._lock:
            if camera_id is not None:
                item = self._items.get((camera_id, int(track_id)))
                return item.thumb_path if item is not None else None
            matches = [
                item.thumb_path
                for (item_camera, item_track_id), item in self._items.items()
                if item_track_id == int(track_id)
            ]
            return matches[0] if len(matches) == 1 else None

    def mark_camera_inactive(self, camera_id: str) -> None:
        with self._lock:
            for (item_camera, _), item in self._items.items():
                if item_camera == camera_id:
                    item.active = False

    def clear_cameras(self, camera_ids: list[str]) -> None:
        selected = set(camera_ids)
        with self._lock:
            keys = [key for key in self._items if key[0] in selected]
            for key in keys:
                item = self._items.pop(key, None)
                if item is not None:
                    self._unlink_locked(item.thumb_path)

    def discard_tracks(self, camera_id: str, track_ids: list[int]) -> None:
        """Rollback crops created after the camera concurrently disappeared."""
        with self._lock:
            for track_id in track_ids:
                item = self._items.pop((camera_id, int(track_id)), None)
                if item is not None:
                    self._unlink_locked(item.thumb_path)

    def reset(self) -> None:
        with self._lock:
            items = list(self._items.values())
            self._items.clear()
            for item in items:
                self._unlink_locked(item.thumb_path)
