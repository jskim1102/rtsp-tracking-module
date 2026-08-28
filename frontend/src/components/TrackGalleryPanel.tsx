import { useEffect, useMemo, useRef, useState } from "react";
import { apiBase } from "../hooks/useApi";
import {
  useTracksApi,
  type TrackerResetNotice,
  type TracksSnapshot,
} from "../hooks/useTracksApi";
import type { TrackLifecycleEvent } from "../hooks/useTracksWs";
import { trackColor } from "../utils/colors";

interface CameraLabel {
  name: string;
  stream_key: string;
}

interface Props {
  cams?: readonly CameraLabel[];
  initialData?: TracksSnapshot;
  initialResetNotice?: TrackerResetNotice | null;
  resetEvent?: TrackerResetNotice | null;
  trackEvents?: readonly TrackEventEnvelope[];
}

export interface TrackEventEnvelope {
  sequence: number;
  event: TrackLifecycleEvent;
}

function formatTime(epochSeconds: number): string {
  return new Intl.DateTimeFormat("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date(epochSeconds * 1000));
}

function thumbnailSrc(url: string): string {
  if (/^(?:https?:|data:|blob:)/.test(url)) return url;
  return `${apiBase()}${url.startsWith("/") ? url : `/${url}`}`;
}

export default function TrackGalleryPanel({
  cams = [],
  initialData,
  initialResetNotice,
  resetEvent,
  trackEvents = [],
}: Props) {
  const {
    data,
    error,
    loading,
    reset,
    resetNotice,
    applyEvent,
    setData,
    setResetNotice,
  } = useTracksApi({ initialData, initialResetNotice });
  const [resetting, setResetting] = useState(false);
  const [resetError, setResetError] = useState("");
  const processedEventRef = useRef(0);

  useEffect(() => {
    if (!resetEvent) return;
    setData({ active: 0, total: 0, items: [] });
    setResetNotice(resetEvent);
  }, [resetEvent, setData, setResetNotice]);

  useEffect(() => {
    for (const envelope of trackEvents) {
      if (envelope.sequence <= processedEventRef.current) continue;
      applyEvent(envelope.event);
      processedEventRef.current = envelope.sequence;
    }
  }, [applyEvent, trackEvents]);

  const cameraNames = useMemo(
    () => new Map(cams.flatMap((cam) => {
      const sourceId = cam.stream_key.startsWith("ipcam-")
        ? cam.stream_key
        : `ipcam-${cam.stream_key}`;
      return [
        [cam.stream_key, cam.name] as const,
        [sourceId, cam.name] as const,
      ];
    })),
    [cams],
  );
  const items = useMemo(
    () => [...data.items].sort((left, right) => right.first_seen - left.first_seen),
    [data.items],
  );

  const handleReset = async () => {
    if (!window.confirm("전체 track ID 와 갤러리를 리셋합니다. 계속할까요?")) {
      return;
    }
    setResetting(true);
    setResetError("");
    try {
      await reset();
    } catch (reason) {
      setResetError(`추적 리셋에 실패했습니다: ${reason}`);
    } finally {
      setResetting(false);
    }
  };

  return (
    <section className="track-gallery">
      <div className="track-gallery-head">
        <h2>ID 갤러리</h2>
        <div className="track-gallery-actions">
          <div className="track-counters">
            <span>
              <span className="track-counter-label">활성</span>
              <span className="track-counter-value">{data.active}</span>
            </span>
            <span>
              <span className="track-counter-label">누적</span>
              <span className="track-counter-value">{data.total}</span>
            </span>
          </div>
          <button
            className="danger"
            disabled={resetting}
            onClick={() => void handleReset()}
          >
            {resetting ? "리셋 중..." : "추적 리셋"}
          </button>
        </div>
      </div>

      {resetNotice && (
        <p className="tracker-reset-note" aria-live="polite">
          {formatTime(resetNotice.occurred_at)}{" "}
          {cameraNames.get(resetNotice.camera_id) ?? resetNotice.camera_id} tracker 리셋
          {" — "}{resetNotice.reason}
          {" · 기존 ID 연속성이 끊겨 새 ID 로 재발급됩니다."}
        </p>
      )}

      {(error || resetError) && (
        <p className="form-error">{resetError || error}</p>
      )}

      {items.length === 0 ? (
        <p className="track-empty">
          {loading ? "track 로딩 중..." : "아직 등장한 track 이 없습니다"}
        </p>
      ) : (
        <div className="track-tiles">
          {items.map((item) => (
            <article
              className="track-tile"
              key={`${item.camera_id}-${item.track_id}`}
            >
              <div className="track-tile-thumb">
                <img
                  src={thumbnailSrc(item.thumb_url)}
                  alt={`track #${item.track_id} 첫 등장 crop`}
                />
              </div>
              <div className="track-tile-meta">
                <div className="track-tile-id">
                  <span className="track-tile-id-label">
                    <span
                      className="track-tile-swatch"
                      style={{ background: trackColor(item.track_id) }}
                      aria-hidden="true"
                    />
                    #{item.track_id}
                  </span>
                  <span
                    className={`track-badge ${
                      item.active ? "track-badge-on" : "track-badge-off"
                    }`}
                  >
                    {item.active ? "● 활성" : "○ 이탈"}
                  </span>
                </div>
                <span className="track-tile-cam">
                  {cameraNames.get(item.camera_id) ?? item.camera_id}
                </span>
                <time className="track-tile-time" dateTime={new Date(item.first_seen * 1000).toISOString()}>
                  {formatTime(item.first_seen)}
                </time>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
