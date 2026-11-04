import { useEffect, useMemo, useState } from "react";
import { apiBase } from "../hooks/useApi";

export interface TrackerInfo {
  name: string;
  available: string[];
}

interface TrackerSlot {
  name: string;
  label: string;
}

interface Props {
  initialInfo?: TrackerInfo;
  onChanged?: (name: string) => void;
  switchTracker?: (name: string) => Promise<TrackerInfo>;
}

export const TRACKER_SLOTS: readonly TrackerSlot[] = [
  { name: "sam3", label: "SAM3" },
  { name: "yolo-botsort", label: "YOLO+BoT-SORT" },
  { name: "yolo-bytetrack", label: "YOLO+ByteTrack" },
  { name: "motip", label: "MOTIP" },
  { name: "samurai", label: "SAMURAI" },
];

export default function TrackerSwitchBar({
  initialInfo,
  onChanged,
  switchTracker,
}: Props) {
  const [info, setInfo] = useState<TrackerInfo | null>(initialInfo ?? null);
  const [loading, setLoading] = useState(initialInfo === undefined);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (initialInfo) return;

    let cancelled = false;
    setLoading(true);
    fetch(`${apiBase()}/api/tracker`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return (await response.json()) as TrackerInfo;
      })
      .then((next) => {
        if (cancelled) return;
        setInfo(next);
        setError("");
      })
      .catch((reason) => {
        if (!cancelled) {
          setError(`tracker 정보를 불러오지 못했습니다: ${reason}`);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [initialInfo]);

  const available = useMemo(() => new Set(info?.available ?? []), [info]);

  const selectTracker = async (name: string, label: string) => {
    if (!available.has(name) || info?.name === name || loading || switching) return;
    if (!window.confirm(
      `tracker 를 ${label} 으로 전환합니다. 모든 카메라 세션이 재시작되고 track ID 연속성이 끊깁니다. 계속할까요?`,
    )) {
      return;
    }

    setSwitching(true);
    setError("");
    try {
      let next: TrackerInfo;
      if (switchTracker) {
        next = await switchTracker(name);
      } else {
        const response = await fetch(`${apiBase()}/api/tracker`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        next = (await response.json()) as TrackerInfo;
      }
      setInfo(next);
      onChanged?.(next.name);
    } catch (reason) {
      setError(`tracker 변경에 실패했습니다: ${reason}`);
    } finally {
      setSwitching(false);
    }
  };

  return (
    <>
      <div className="tracker-switch" aria-label="Tracker 선택">
        {TRACKER_SLOTS.map((slot) => {
          const implemented = available.has(slot.name);
          return (
            <button
              key={slot.name}
              type="button"
              className={`tracker-switch-item${info?.name === slot.name ? " on" : ""}`}
              disabled={!implemented || loading || switching}
              title={implemented ? undefined : "미구현 — registry 예약 슬롯"}
              aria-pressed={info?.name === slot.name}
              onClick={() => void selectTracker(slot.name, slot.label)}
            >
              {slot.label}
            </button>
          );
        })}
      </div>
      {error && <p className="form-error tracker-switch-error">{error}</p>}
    </>
  );
}
