import type { TrackLifecycleEvent } from "../hooks/useTracksWs";

export interface TrackItem {
  track_id: number;
  camera_id: string;
  first_seen: number;
  thumb_url: string;
  active: boolean;
}

export interface TracksSnapshot {
  active: number;
  total: number;
  items: TrackItem[];
}

export interface TrackerResetNotice {
  camera_id: string;
  reason: string;
  occurred_at: number;
}

export function trackEventInvalidatesPoll(event: TrackLifecycleEvent): boolean {
  return event.type === "tracker_reset";
}

export function nextTrackPollDelay(
  startedGeneration: number,
  currentGeneration: number,
): number {
  return startedGeneration === currentGeneration ? 1000 : 0;
}

export function reduceTrackEvent(
  snapshot: TracksSnapshot,
  event: TrackLifecycleEvent,
  occurredAt = Date.now() / 1000,
): { data: TracksSnapshot; resetNotice: TrackerResetNotice | null } {
  if (event.type === "tracker_reset") {
    const items = snapshot.items.filter(
      (item) => item.camera_id !== event.camera_id,
    );
    return {
      data: {
        active: items.filter((item) => item.active).length,
        total: items.length,
        items,
      },
      resetNotice: {
        camera_id: event.camera_id,
        reason: event.reason,
        occurred_at: occurredAt,
      },
    };
  }

  const nextItem: TrackItem = {
    track_id: event.track_id,
    camera_id: event.camera_id,
    first_seen: event.first_seen,
    thumb_url: event.thumb_url,
    active: true,
  };
  const existingIndex = snapshot.items.findIndex(
    (item) =>
      item.camera_id === event.camera_id && item.track_id === event.track_id,
  );
  const items = [...snapshot.items];
  if (existingIndex >= 0) {
    items[existingIndex] = { ...items[existingIndex], ...nextItem };
  } else {
    items.unshift(nextItem);
  }
  return {
    data: {
      active: items.filter((item) => item.active).length,
      total: items.length,
      items,
    },
    resetNotice: null,
  };
}
