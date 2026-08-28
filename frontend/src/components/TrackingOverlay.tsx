import { useEffect, useRef, useState } from "react";
import { trackColor } from "../utils/colors";

/**
 * WHEP `<video>` 위에 track mask/bbox 좌표를 그리는 hold-last canvas.
 *
 * 영상과 tracker 캡처가 다른 경로이므로 `videoNatural / frame` 비율로
 * 좌표를 맞춘다. 새 결과가 오기 전까지 현재 canvas 픽셀을 그대로 유지한다.
 */

const MASK_FILL_ALPHA = 0.35;

export interface Track {
  track_id: number;
  bbox: [number, number, number, number];
  polygon: [number, number][] | null;
  score: number;
  is_new: boolean;
}

interface Props {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  tracks: Track[];
  frameW: number;
  frameH: number;
}

function traceMask(
  ctx: CanvasRenderingContext2D,
  polygon: [number, number][],
  sx: number,
  sy: number,
): boolean {
  if (polygon.length < 3) return false;
  ctx.beginPath();
  ctx.moveTo(polygon[0][0] * sx, polygon[0][1] * sy);
  for (let index = 1; index < polygon.length; index += 1) {
    ctx.lineTo(polygon[index][0] * sx, polygon[index][1] * sy);
  }
  ctx.closePath();
  return true;
}

export default function TrackingOverlay({
  videoRef,
  tracks,
  frameW,
  frameH,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const update = () => {
      if (video.videoWidth && video.videoHeight) {
        setSize((previous) => {
          if (previous?.w === video.videoWidth && previous?.h === video.videoHeight) {
            return previous;
          }
          return { w: video.videoWidth, h: video.videoHeight };
        });
      }
    };
    video.addEventListener("loadedmetadata", update);
    video.addEventListener("resize", update);
    update();
    return () => {
      video.removeEventListener("loadedmetadata", update);
      video.removeEventListener("resize", update);
    };
  }, [videoRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !size) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (tracks.length === 0) return;

    const sx = frameW > 0 ? size.w / frameW : 1;
    const sy = frameH > 0 ? size.h / frameH : 1;
    const scale = Math.max(1, Math.min(size.w, size.h) / 600);
    const lineWidth = Math.max(1.5, 2 * scale);
    const fontPx = Math.max(11, Math.round(13 * scale));
    ctx.font = `${fontPx}px sans-serif`;
    ctx.textBaseline = "top";

    // 1. polygon mask 채움. bbox-only tracker는 채움없이 2단계에서 stroke만 그린다.
    ctx.globalAlpha = MASK_FILL_ALPHA;
    for (const track of tracks) {
      if (!track.polygon || !traceMask(ctx, track.polygon, sx, sy)) continue;
      ctx.fillStyle = trackColor(track.track_id);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // 2. polygon 외곽선 또는 nullable polygon의 bbox 폴백.
    ctx.lineWidth = lineWidth;
    for (const track of tracks) {
      ctx.strokeStyle = trackColor(track.track_id);
      if (track.polygon && traceMask(ctx, track.polygon, sx, sy)) {
        ctx.stroke();
        continue;
      }
      const [rawX1, rawY1, rawX2, rawY2] = track.bbox;
      const x1 = rawX1 * sx;
      const y1 = rawY1 * sy;
      ctx.strokeRect(x1, y1, (rawX2 - rawX1) * sx, (rawY2 - rawY1) * sy);
    }

    // 3. ID 라벨은 다른 track 채움에 가리지 않도록 마지막에 그린다.
    for (const track of tracks) {
      const [rawX1, rawY1] = track.bbox;
      const x1 = rawX1 * sx;
      const y1 = rawY1 * sy;
      const color = trackColor(track.track_id);
      const label = `#${track.track_id}`;
      const padX = 4 * scale;
      const padY = 2 * scale;
      const textWidth = ctx.measureText(label).width;
      const labelHeight = fontPx + padY * 2;
      const labelY = Math.max(0, y1 - labelHeight);

      ctx.fillStyle = color;
      ctx.fillRect(x1, labelY, textWidth + padX * 2, labelHeight);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + padX, labelY + padY);
    }
  }, [frameH, frameW, size, tracks]);

  if (!size) return null;

  return (
    <canvas
      ref={canvasRef}
      width={size.w}
      height={size.h}
      style={canvasStyle}
    />
  );
}

const canvasStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "contain",
  pointerEvents: "none",
};
