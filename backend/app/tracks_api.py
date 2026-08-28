"""Track gallery and global tracker control REST API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.streaming.manager import manager as stream_manager
from app.tracking.registry import available_trackers, tracker_capabilities


class TrackerUpdate(BaseModel):
    name: str

    model_config = {"extra": "forbid"}


tracks_router = APIRouter(tags=["tracks"])


@tracks_router.get("/api/tracks")
def list_tracks() -> dict[str, object]:
    return stream_manager.get_tracks()


@tracks_router.get("/api/tracks/{track_id}/thumb")
def get_track_thumbnail(
    track_id: int,
    camera_id: str | None = Query(default=None),
):
    path = stream_manager.get_track_thumbnail(track_id, camera_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="track thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")


@tracks_router.post("/api/tracks/reset")
def reset_tracks() -> dict[str, bool]:
    stream_manager.reset_tracking()
    return {"ok": True}


def _tracker_info() -> dict[str, object]:
    name = str(stream_manager.get_inference_config()["model"])
    try:
        capabilities = tracker_capabilities(name)
    except ValueError:
        capabilities = {}
    return {
        "name": name,
        "available": available_trackers(),
        "capabilities": capabilities,
    }


@tracks_router.get("/api/tracker")
def get_tracker() -> dict[str, object]:
    return _tracker_info()


@tracks_router.put("/api/tracker")
def update_tracker(body: TrackerUpdate) -> dict[str, object]:
    if body.name not in available_trackers():
        raise HTTPException(status_code=400, detail="tracker is not available")
    try:
        stream_manager.set_inference_model(body.name)
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _tracker_info()
