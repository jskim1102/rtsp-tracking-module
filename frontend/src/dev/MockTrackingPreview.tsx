import { useEffect, useRef, useState } from "react";
import TrackGalleryPanel from "../components/TrackGalleryPanel";
import TrackerSwitchBar, {
  type TrackerInfo,
} from "../components/TrackerSwitchBar";
import TrackingOverlay, {
  type Track,
} from "../components/TrackingOverlay";
import type {
  TrackerResetNotice,
  TracksSnapshot,
} from "../hooks/useTracksApi";

const VIDEO_WIDTH = 1280;
const VIDEO_HEIGHT = 720;
const FRAME_WIDTH = 640;
const FRAME_HEIGHT = 360;
const MOCK_NOW = 1_787_000_000;

function ellipse(
  centerX: number,
  centerY: number,
  radiusX: number,
  radiusY: number,
  points = 24,
): [number, number][] {
  return Array.from({ length: points }, (_, index) => {
    const angle = (index / points) * Math.PI * 2;
    return [
      Math.round(centerX + Math.cos(angle) * radiusX),
      Math.round(centerY + Math.sin(angle) * radiusY),
    ];
  });
}

const TRACKS: Track[] = [
  {
    track_id: 6,
    bbox: [55, 30, 235, 335],
    polygon: ellipse(145, 182, 82, 145),
    score: 0.94,
    is_new: false,
  },
  {
    track_id: 7,
    bbox: [245, 45, 455, 330],
    polygon: ellipse(350, 185, 96, 136),
    score: 0.91,
    is_new: true,
  },
  {
    track_id: 8,
    bbox: [505, 195, 571, 345],
    polygon: null,
    score: 0.87,
    is_new: false,
  },
];

const MOCK_CAMS = [
  { stream_key: "ipcam-living", name: "거실 카메라" },
  { stream_key: "ipcam-entry", name: "현관 카메라" },
] as const;

function mockThumb(trackId: number, wide = false): string {
  const width = wide ? 140 : 84;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="160" viewBox="0 0 ${width} 160"><rect width="100%" height="100%" fill="#111827"/><circle cx="${width / 2}" cy="45" r="18" fill="#9ca3af"/><rect x="${width / 2 - 20}" y="68" width="40" height="76" rx="16" fill="#9ca3af"/><text x="8" y="152" fill="#f9fafb" font-family="sans-serif" font-size="12">#${trackId}</text></svg>`;
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

const MOCK_GALLERY: TracksSnapshot = {
  active: 3,
  total: 8,
  items: Array.from({ length: 8 }, (_, index) => {
    const trackId = 8 - index;
    return {
      track_id: trackId,
      camera_id: trackId % 2 === 0 ? "ipcam-living" : "ipcam-entry",
      first_seen: MOCK_NOW - index * 157,
      thumb_url: mockThumb(trackId, trackId === 4 || trackId === 7),
      active: trackId >= 6,
    };
  }),
};

const MOCK_TRACKER: TrackerInfo = {
  name: "sam3",
  available: ["sam3"],
};

function drawTestPattern(ctx: CanvasRenderingContext2D) {
  const gradient = ctx.createLinearGradient(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
  gradient.addColorStop(0, "#111827");
  gradient.addColorStop(1, "#334155");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);

  ctx.strokeStyle = "rgba(255,255,255,0.16)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= VIDEO_WIDTH; x += 160) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, VIDEO_HEIGHT);
    ctx.stroke();
  }
  for (let y = 0; y <= VIDEO_HEIGHT; y += 90) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(VIDEO_WIDTH, y);
    ctx.stroke();
  }

  for (const track of TRACKS) {
    if (track.polygon) {
      ctx.beginPath();
      ctx.moveTo(track.polygon[0][0] * 2, track.polygon[0][1] * 2);
      for (let index = 1; index < track.polygon.length; index += 1) {
        ctx.lineTo(track.polygon[index][0] * 2, track.polygon[index][1] * 2);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.72)";
      ctx.lineWidth = 3;
      ctx.stroke();
      continue;
    }
    const [x1, y1, x2, y2] = track.bbox;
    ctx.strokeStyle = "rgba(255,255,255,0.72)";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1 * 2, y1 * 2, (x2 - x1) * 2, (y2 - y1) * 2);
  }

  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.font = "24px sans-serif";
  ctx.fillText("video 1280×720 · tracker frame 640×360 · expected scale 2×", 24, 40);
}

export default function MockTrackingPreview() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [previewError, setPreviewError] = useState("");
  const [trackerReset, setTrackerReset] = useState<TrackerResetNotice | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const source = document.createElement("canvas");
    source.width = VIDEO_WIDTH;
    source.height = VIDEO_HEIGHT;
    const ctx = source.getContext("2d");
    if (!ctx) {
      setPreviewError("테스트 canvas를 만들 수 없습니다.");
      return;
    }
    drawTestPattern(ctx);

    const stream = source.captureStream(10);
    video.srcObject = stream;

    return () => {
      stream.getTracks().forEach((track) => track.stop());
      video.srcObject = null;
    };
  }, []);

  return (
    <main className="app">
      <TrackerSwitchBar
        initialInfo={MOCK_TRACKER}
        switchTracker={async (name) => ({ ...MOCK_TRACKER, name })}
        onChanged={(name) => {
          setTrackerReset({
            camera_id: "ipcam-entry",
            reason: `tracker_switch (${name})`,
            occurred_at: Date.now() / 1000,
          });
        }}
      />

      <header className="page-head">
        <div>
          <h1>RTSP Tracking Mock</h1>
          <p className="subtitle">백엔드 없이 overlay·ID 갤러리·tracker 선택을 검수합니다.</p>
        </div>
      </header>

      {previewError && <p className="form-error">{previewError}</p>}
      <div style={styles.viewport}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={styles.video}
        />
        <TrackingOverlay
          videoRef={videoRef}
          tracks={TRACKS}
          frameW={FRAME_WIDTH}
          frameH={FRAME_HEIGHT}
        />
      </div>
      <p style={styles.hint}>
        #6·#7은 polygon mask, #8은 polygon=None bbox 폴백입니다. 흰 실루엣과 외곽선이 맞물리면 2× SEAM 스케일이 정확합니다.
      </p>

      <TrackGalleryPanel
        cams={MOCK_CAMS}
        initialData={MOCK_GALLERY}
        initialResetNotice={{
          camera_id: "ipcam-entry",
          reason: "worker_respawn",
          occurred_at: MOCK_NOW - 300,
        }}
        resetEvent={trackerReset}
      />
    </main>
  );
}

const styles: Record<string, React.CSSProperties> = {
  viewport: {
    position: "relative",
    width: "100%",
    aspectRatio: "16 / 9",
    overflow: "hidden",
    background: "#111827",
  },
  video: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    objectFit: "contain",
  },
  hint: { margin: "8px 0 0", color: "#6b7280" },
};
