import { useEffect, useRef, useState } from "react";
import { apiBase } from "./useApi";
import type { Track } from "../components/TrackingOverlay";

const RECONNECT_DELAY = 2000;
const TRACK_FPS_WINDOW_MS = 2000;

export interface NewTrackEvent {
  type: "new_track";
  track_id: number;
  camera_id: string;
  first_seen: number;
  thumb_url: string;
}

export interface TrackerResetEvent {
  type: "tracker_reset";
  camera_id: string;
  reason: string;
}

export type TrackLifecycleEvent = NewTrackEvent | TrackerResetEvent;

export interface TrackStream {
  items: Track[];
  frameW: number;
  frameH: number;
  trackFps: number;
}

export function useTracksWs(
  streamKey: string,
  active: boolean,
  onEvent?: (event: TrackLifecycleEvent) => void,
): TrackStream {
  const [items, setItems] = useState<Track[]>([]);
  const [frame, setFrame] = useState({ w: 0, h: 0 });
  const [trackFps, setTrackFps] = useState(0);
  const arrivalsRef = useRef<number[]>([]);
  const onEventRef = useRef(onEvent);

  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  useEffect(() => {
    if (!active) {
      setItems([]);
      setFrame({ w: 0, h: 0 });
      arrivalsRef.current = [];
      setTrackFps(0);
      return;
    }

    let unmounted = false;
    let reconnectTimer: number | null = null;
    let ws: WebSocket | null = null;
    const fpsTimer = window.setInterval(() => {
      if (unmounted) return;
      const cutoff = performance.now() - TRACK_FPS_WINDOW_MS;
      const arrivals = arrivalsRef.current.filter((time) => time >= cutoff);
      arrivalsRef.current = arrivals;
      setTrackFps(arrivals.length / (TRACK_FPS_WINDOW_MS / 1000));
    }, 1000);
    const wsUrl = `${apiBase().replace(/^http/, "ws")}/api/ipcams/${streamKey}/ws`;

    const connect = () => {
      if (unmounted) return;
      ws = new WebSocket(wsUrl);
      ws.onmessage = (message: MessageEvent) => {
        if (unmounted || typeof message.data !== "string") return;
        try {
          const payload = JSON.parse(message.data) as Record<string, unknown>;
          if (payload.type === "tracks") {
            arrivalsRef.current.push(performance.now());
            setItems((payload.items ?? []) as Track[]);
            const nextFrame = payload.frame as
              | { w?: number; h?: number }
              | undefined;
            setFrame({ w: nextFrame?.w ?? 0, h: nextFrame?.h ?? 0 });
            return;
          }
          if (payload.type === "new_track") {
            onEventRef.current?.(payload as unknown as NewTrackEvent);
            return;
          }
          if (payload.type === "tracker_reset") {
            setItems([]);
            arrivalsRef.current = [];
            setTrackFps(0);
            onEventRef.current?.(payload as unknown as TrackerResetEvent);
          }
        } catch {
          // malformed text payloads are isolated from the reconnect lifecycle.
        }
      };
      ws.onclose = () => {
        if (!unmounted) {
          reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY);
        }
      };
      ws.onerror = () => ws?.close();
    };

    connect();
    return () => {
      unmounted = true;
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
      window.clearInterval(fpsTimer);
      arrivalsRef.current = [];
      setTrackFps(0);
      ws?.close();
    };
  }, [active, streamKey]);

  return { items, frameW: frame.w, frameH: frame.h, trackFps };
}
