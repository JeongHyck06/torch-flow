// 하단 Run 패널. Figma 02 Attach mode 아래쪽.
//
// 탭은 알약이 아니라 언더라인이다. 곡선은 SVG 폴리라인 하나로 그린다.
// 차트 라이브러리는 스트리밍이 붙을 때 필요해지고 지금은 아니다.

import { useCallback, useEffect, useState } from "react";
import { fetchCurve, fetchLogs, fetchRuns } from "../api";
import type { CurveSeries, RunInfo } from "../api";
import { useStore } from "../store";

type Tab = "curves" | "manifest" | "logs";

const COLORS = ["var(--dtype-f32)", "#4a86c9", "#7c8898", "#2a558d", "#5e93d1"];

export function RunPanel() {
  const [tab, setTab] = useState<Tab>("curves");
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [key, setKey] = useState<string>("");
  const [series, setSeries] = useState<CurveSeries[]>([]);
  const [logs, setLogs] = useState<{ text: string; stream: string; node_id?: string }[]>([]);
  const open = useStore((state) => state.runPanel);
  const toggle = useStore((state) => state.toggleRunPanel);

  const refresh = useCallback(async () => {
    const found = await fetchRuns();
    setRuns(found);
    const keys = [...new Set(found.flatMap((run) => run.keys))];
    const chosen = key && keys.includes(key) ? key : keys[0] ?? "";
    setKey(chosen);
    if (chosen) setSeries(await fetchCurve(chosen, found.map((run) => run.id)));
    if (tab === "logs") setLogs(await fetchLogs());
  }, [key, tab]);

  useEffect(() => {
    if (!open) return;
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => window.clearInterval(timer);
  }, [open, refresh]);

  if (!open) {
    return (
      <button className="runpanel__handle" onClick={toggle}>
        Run 곡선 {runs.length ? `(${runs.length})` : ""}
      </button>
    );
  }

  const active = runs.find((run) => run.id === (runs[0]?.id ?? ""));
  const keys = [...new Set(runs.flatMap((run) => run.keys))];

  return (
    <section className="runpanel" aria-label="Run 패널">
      <div className="runpanel__tabs">
        {(["curves", "manifest", "logs"] as Tab[]).map((name) => (
          <button
            key={name}
            className={`runpanel__tab${tab === name ? " runpanel__tab--on" : ""}`}
            onClick={() => setTab(name)}
          >
            {name === "curves" ? "Run 곡선" : name === "manifest" ? "run manifest" : "로그"}
          </button>
        ))}
        <div className="runpanel__spacer" />
        {tab === "curves" && keys.length > 1 && (
          <select className="runpanel__select mono" value={key}
                  onChange={(event) => setKey(event.target.value)}>
            {keys.map((name) => <option key={name} value={name}>{name}</option>)}
          </select>
        )}
        <button className="runpanel__close" onClick={toggle} aria-label="닫기">닫기</button>
      </div>

      <div className="runpanel__body">
        {tab === "curves" && (
          series.length === 0 || series.every((one) => one.points.length === 0) ? (
            <p className="mono muted">
              아직 기록이 없습니다. 학습 스크립트에서 tf.log(step, loss=...)를 부르면 여기에 쌓입니다
            </p>
          ) : (
            <Curves series={series} label={key} />
          )
        )}

        {tab === "manifest" && (
          active?.manifest && Object.keys(active.manifest).length ? (
            <pre className="mono runpanel__json">
              {JSON.stringify(active.manifest, null, 2)}
            </pre>
          ) : <p className="mono muted">manifest가 아직 없습니다</p>
        )}

        {tab === "logs" && (
          logs.length ? (
            <pre className="mono runpanel__json">
              {logs.map((line) =>
                `${line.node_id ? line.node_id + "  " : ""}${line.text}`).join("\n")}
            </pre>
          ) : <p className="mono muted">로그가 비어 있습니다</p>
        )}
      </div>
    </section>
  );
}

function Curves({ series, label }: { series: CurveSeries[]; label: string }) {
  const points = series.flatMap((one) => one.points);
  if (!points.length) return null;

  const width = 900;
  const height = 150;
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x: number) => ((x - x0) / Math.max(x1 - x0, 1)) * width;
  const sy = (y: number) => height - ((y - y0) / Math.max(y1 - y0, 1e-9)) * height;

  return (
    <div className="curves">
      <div className="curves__legend mono">
        <span>{label}</span>
        {series.map((one, index) => (
          <span key={one.run} className="curves__key">
            <span className="curves__swatch" style={{ background: COLORS[index % COLORS.length] }} />
            {one.run}
          </span>
        ))}
        <span className="muted">{y0.toFixed(4)} .. {y1.toFixed(4)}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none"
           className="curves__plot" role="img" aria-label={`${label} 곡선`}>
        {series.map((one, index) => (
          <polyline
            key={one.run}
            fill="none"
            stroke={COLORS[index % COLORS.length]}
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
            points={one.points.map((point) => `${sx(point[0])},${sy(point[1])}`).join(" ")}
          />
        ))}
      </svg>
    </div>
  );
}
