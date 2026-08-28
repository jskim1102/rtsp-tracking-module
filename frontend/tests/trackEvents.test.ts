import assert from "node:assert/strict";
import test from "node:test";

import {
  nextTrackPollDelay,
  reduceTrackEvent,
  trackEventInvalidatesPoll,
  type TracksSnapshot,
} from "../src/utils/trackEvents.ts";


const EMPTY: TracksSnapshot = { active: 0, total: 0, items: [] };


test("new_track 이벤트는 polling 전에 즉시 활성 타일을 추가한다", () => {
  const reduced = reduceTrackEvent(EMPTY, {
    type: "new_track",
    track_id: 7,
    camera_id: "ipcam-a",
    first_seen: 10,
    thumb_url: "/api/tracks/7/thumb?camera_id=ipcam-a",
  });

  assert.equal(reduced.data.active, 1);
  assert.equal(reduced.data.total, 1);
  assert.equal(reduced.data.items[0].track_id, 7);
});


test("중복 new_track 이벤트는 같은 카메라 namespace에서 타일을 늘리지 않는다", () => {
  const event = {
    type: "new_track" as const,
    track_id: 7,
    camera_id: "ipcam-a",
    first_seen: 10,
    thumb_url: "/api/tracks/7/thumb?camera_id=ipcam-a",
  };
  const first = reduceTrackEvent(EMPTY, event);
  const second = reduceTrackEvent(first.data, event);

  assert.equal(second.data.total, 1);
  assert.equal(second.data.active, 1);
});


test("tracker_reset은 해당 카메라 타일만 지우고 reset notice를 남긴다", () => {
  const snapshot: TracksSnapshot = {
    active: 2,
    total: 2,
    items: [
      { track_id: 1, camera_id: "ipcam-a", first_seen: 1, thumb_url: "a", active: true },
      { track_id: 1, camera_id: "ipcam-b", first_seen: 1, thumb_url: "b", active: true },
    ],
  };

  const reduced = reduceTrackEvent(
    snapshot,
    { type: "tracker_reset", camera_id: "ipcam-a", reason: "worker_respawn" },
    99,
  );

  assert.equal(reduced.data.total, 1);
  assert.equal(reduced.data.items[0].camera_id, "ipcam-b");
  assert.deepEqual(reduced.resetNotice, {
    camera_id: "ipcam-a",
    reason: "worker_respawn",
    occurred_at: 99,
  });
});


test("poll generation은 tracker_reset만 무효화하고 new_track은 건드리지 않는다", () => {
  assert.equal(trackEventInvalidatesPoll({
    type: "new_track",
    track_id: 7,
    camera_id: "ipcam-a",
    first_seen: 10,
    thumb_url: "/thumb",
  }), false);
  assert.equal(trackEventInvalidatesPoll({
    type: "tracker_reset",
    camera_id: "ipcam-a",
    reason: "worker_respawn",
  }), true);
});


test("generation 불일치 poll은 1초를 기다리지 않고 즉시 재시도한다", () => {
  assert.equal(nextTrackPollDelay(3, 4), 0);
  assert.equal(nextTrackPollDelay(4, 4), 1000);
});
