import { useCallback, useEffect, useRef, useState } from "react";
import { apiBase } from "./useApi";
import type { TrackLifecycleEvent } from "./useTracksWs";
import {
  nextTrackPollDelay,
  reduceTrackEvent,
  trackEventInvalidatesPoll,
  type TrackerResetNotice,
  type TracksSnapshot,
} from "../utils/trackEvents";

export type {
  TrackItem,
  TrackerResetNotice,
  TracksSnapshot,
} from "../utils/trackEvents";

interface UseTracksApiOptions {
  initialData?: TracksSnapshot;
  initialResetNotice?: TrackerResetNotice | null;
}

const EMPTY_TRACKS: TracksSnapshot = { active: 0, total: 0, items: [] };

export function useTracksApi(options: UseTracksApiOptions = {}) {
  const mockMode = options.initialData !== undefined;
  const [data, setData] = useState<TracksSnapshot>(
    options.initialData ?? EMPTY_TRACKS,
  );
  const [resetNotice, setResetNotice] = useState<TrackerResetNotice | null>(
    options.initialResetNotice ?? null,
  );
  const [loading, setLoading] = useState(!mockMode);
  const [error, setError] = useState("");
  const eventVersionRef = useRef(0);

  useEffect(() => {
    if (mockMode) return;
    let cancelled = false;
    let timer: number | null = null;
    const poll = async () => {
      const eventVersion = eventVersionRef.current;
      try {
        const response = await fetch(`${apiBase()}/api/tracks`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const snapshot = (await response.json()) as TracksSnapshot;
        if (!cancelled && eventVersion === eventVersionRef.current) {
          setData(snapshot);
          setError("");
        }
      } catch (reason) {
        if (!cancelled && eventVersion === eventVersionRef.current) {
          setError(`track 목록을 불러오지 못했습니다: ${reason}`);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
          timer = window.setTimeout(
            poll,
            nextTrackPollDelay(eventVersion, eventVersionRef.current),
          );
        }
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [mockMode]);

  const reset = useCallback(async () => {
    if (!mockMode) {
      const response = await fetch(`${apiBase()}/api/tracks/reset`, {
        method: "POST",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
    }
    eventVersionRef.current += 1;
    setData(EMPTY_TRACKS);
    setResetNotice(null);
    setError("");
  }, [mockMode]);

  const applyEvent = useCallback((event: TrackLifecycleEvent) => {
    if (trackEventInvalidatesPoll(event)) {
      eventVersionRef.current += 1;
    }
    if (event.type === "tracker_reset") {
      setResetNotice({
        camera_id: event.camera_id,
        reason: event.reason,
        occurred_at: Date.now() / 1000,
      });
    }
    setData((previous) => reduceTrackEvent(previous, event).data);
  }, []);

  return {
    data,
    error,
    loading,
    applyEvent,
    reset,
    resetNotice,
    setData,
    setResetNotice,
  };
}
