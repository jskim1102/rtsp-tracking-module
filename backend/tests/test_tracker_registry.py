from __future__ import annotations

from dataclasses import fields

import pytest

from app.tracking.base import Track
from app.tracking.registry import (
    available_trackers,
    create_tracker,
    reserved_trackers,
    tracker_capabilities,
)


def test_track_fields_match_the_public_adapter_contract():
    assert [field.name for field in fields(Track)] == [
        "track_id",
        "bbox",
        "polygon",
        "score",
        "is_new",
    ]


def test_registry_exposes_all_implemented_trackers_and_capabilities():
    tracker = create_tracker("sam3", backend=object())

    assert tracker.capabilities == {"concept_prompt": True}
    assert available_trackers() == ["sam3", "yolo-botsort", "yolo-bytetrack"]
    assert tracker_capabilities("yolo-botsort") == {"concept_prompt": False}
    assert tracker_capabilities("yolo-bytetrack") == {"concept_prompt": False}


def test_registry_uses_tracker_default_environment_variable(monkeypatch):
    monkeypatch.setenv("TRACKER_DEFAULT", "sam3")

    assert create_tracker(backend=object()).name == "sam3"


def test_phase6_tracker_names_are_reserved_and_storm_is_removed():
    assert reserved_trackers() == [
        "sam3",
        "yolo-botsort",
        "yolo-bytetrack",
        "motip",
        "samurai",
    ]
    for name in ("motip", "samurai"):
        with pytest.raises(NotImplementedError, match=name):
            create_tracker(name)
    with pytest.raises(ValueError, match="unregistered tracker"):
        create_tracker("storm")


def test_unregistered_tracker_name_is_rejected():
    with pytest.raises(ValueError, match="unregistered tracker"):
        create_tracker("unknown")
