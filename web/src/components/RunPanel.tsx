// 하단 Run 패널. Figma 02 Attach mode 아래쪽.
//
// 탭은 알약이 아니라 언더라인이다. 곡선은 SVG 폴리라인 하나로 그린다.
// 차트 라이브러리는 스트리밍이 붙을 때 필요해지고 지금은 아니다.

import { useCallback, useEffect, useState } from "react";
import { fetchCurve, fetchLogs, fetchRuns } from "../api";
import type { CurveSeries, LogLine, RunInfo } from "../api";
import { setGraphData } from "../api";
import { BlockDetail } from "./BlockDetail";
import { Console } from "./Console";
import { DataCard } from "./DataCard";
import { TestPanel } from "./TestPanel";
import { TrainLog } from "./TrainLog";
import { Trainer } from "./Trainer";
import { SYNTHETIC, useStore } from "../store";
import type { RunTab } from "../store";

const TAB_LABELS: Record<RunTab, string> = {
  curves: "곡선", test: "테스트", stdout: "학습 출력", data: "데이터", block: "블록 상세",
  manifest: "실행 설정 기록", logs: "커널 로그", console: "콘솔",
};
// 앞 다섯은 학습 결과, 뒤 셋은 연구용. 구분선 뒤로 흐리게 두어 처음 보는 사람이 안 헤매게 한다.
const ADVANCED_TABS = new Set<RunTab>(["manifest", "logs", "console"]);
// 곡선 이름. 원어를 괄호에 남긴다 - 검색하거나 논문을 볼 때 필요한 이름이다.
const CURVE_LABELS: Record<string, string> = {
  loss: "손실 (loss)", acc: "정확도 (acc)", val_loss: "검증 손실 (val_loss)",
  val_acc: "검증 정확도 (val_acc)", lr: "학습률 (lr)", grad_norm: "기울기 크기 (grad_norm)",
};

const COLORS = ["var(--dtype-f32)", "#4a86c9", "#7c8898", "#2a558d", "#5e93d1"];
// 색은 한 계열(214°)뿐이라 run이 넷이면 구분이 안 된다 - 선 모양으로 한 번 더 가른다.
const DASHES = ["", "6 3", "2 3", "8 3 2 3", "1 3"];

/** ``run-20260908-011654-cbe5`` -> ``09-08 01:16 · cbe5``. 범례에 id를 그대로 쓰면 못 가른다. */
export function runLabelShort(runId: string): string {
  const match = /^run-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})\d{2}-(\w+)$/.exec(runId);
  return match ? `${match[2]}-${match[3]} ${match[4]}:${match[5]} · ${match[6]}` : runId;
}

const DEFAULT_HEIGHT = 232;
const MIN_HEIGHT = 160;

export function RunPanel() {
  const tab = useStore((state) => state.runTab);
  const setTab = useStore((state) => state.setRunTab);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [key, setKey] = useState<string>("");
  const [series, setSeries] = useState<CurveSeries[]>([]);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const open = useStore((state) => state.runPanel);
  const toggle = useStore((state) => state.toggleRunPanel);
  const dataset = useStore((state) => state.dataset);
  const recipe = useStore((state) => state.recipe);
  const setData = useStore((state) => state.setData);
  const openData = useStore((state) => state.openData);
  const [applied, setApplied] = useState<string | null>(null);
  // 위쪽 경계를 끌어 높이를 바꾼다. 곡선을 크게 보려는 것이라 브라우저에 기억해 둔다.
  const [height, setHeight] = useState(() => {
    try { return Number(localStorage.getItem("torchflow.runpanel.height")) || DEFAULT_HEIGHT; }
    catch { return DEFAULT_HEIGHT; }
  });
  const resize = (next: number) => {
    const clamped = Math.max(MIN_HEIGHT, Math.min(next, window.innerHeight - 160));
    setHeight(clamped);
    try { localStorage.setItem("torchflow.runpanel.height", String(clamped)); } catch { /* 저장 못 해도 동작한다 */ }
  };
  const onGrip = (event: React.PointerEvent<HTMLDivElement>) => {
    const startY = event.clientY;
    const startHeight = height;
    const grip = event.currentTarget;
    grip.setPointerCapture(event.pointerId);
    const move = (moving: PointerEvent) => resize(startHeight + (startY - moving.clientY));
    const stop = () => {
      grip.removeEventListener("pointermove", move);
      grip.removeEventListener("pointerup", stop);
    };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", stop);
  };
  // 테스트 탭은 표와 샘플 격자가 커서 기본 높이로는 잘린다. 처음 열 때 한 번 키운다.
  useEffect(() => {
    if (tab === "test" && height < 480) resize(Math.min(560, Math.round(window.innerHeight * 0.6)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  // 레시피를 그래프에 붙인다. hub가 Input을 set_ports op로 맞추고 브로드캐스트한다.
  const apply = async () => {
    setApplied("…");
    const result = await setGraphData(dataset, recipe);
    if (result.error) { setApplied(result.error); return; }
    setData(dataset, result.spec?.recipe ?? recipe);
    setApplied(`적용됨 · Input [B, ${(result.spec?.shape ?? []).join(", ")}]`);
  };

  const refresh = useCallback(async () => {
    const found = await fetchRuns();
    setRuns(found);
    const keys = [...new Set(found.flatMap((run) => run.keys))];
    // 기본은 loss다 - 사람이 먼저 보는 곡선이 알파벳 순으로 정해지면 안 된다.
    const chosen = key && keys.includes(key) ? key
      : keys.find((name) => name === "loss") ?? keys[0] ?? "";
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
        결과 보기{runs.length ? ` (${runs.length})` : ""}
      </button>
    );
  }

  const active = runs.find((run) => run.id === (runs[0]?.id ?? ""));
  const keys = [...new Set(runs.flatMap((run) => run.keys))];

  return (
    <section className="runpanel" aria-label="Run 패널" style={{ height }}>
      <div
        className="runpanel__grip" role="separator" aria-label="패널 높이"
        title="끌어서 높이 조절 · 더블클릭으로 크게/원래대로"
        onPointerDown={onGrip}
        onDoubleClick={() => resize(height > DEFAULT_HEIGHT ? DEFAULT_HEIGHT : Math.round(window.innerHeight * 0.6))}
      />
      <div className="runpanel__tabs">
        {(Object.keys(TAB_LABELS) as RunTab[]).map((name) => (
          <button
            key={name}
            className={`runpanel__tab${tab === name ? " runpanel__tab--on" : ""}`
              + `${ADVANCED_TABS.has(name) ? " runpanel__tab--dim" : ""}`
              + `${name === "manifest" ? " runpanel__tab--divided" : ""}`}
            onClick={() => setTab(name)}
          >
            {TAB_LABELS[name]}
          </button>
        ))}
        <div className="runpanel__spacer" />
        {tab === "curves" && keys.length > 1 && (
          <select className="runpanel__select mono" value={key}
                  onChange={(event) => setKey(event.target.value)}>
            {keys.map((name) =>
              <option key={name} value={name}>{CURVE_LABELS[name] ?? name}</option>)}
          </select>
        )}
        <button className="runpanel__close" onClick={toggle} aria-label="닫기">닫기</button>
      </div>

      <div className="runpanel__body">
        {/* 함수 identity가 렌더마다 바뀌면 Trainer의 effect가 매 렌더 다시 돌아 요청이 꼬리를 문다. */}
        {tab === "curves" && <Trainer onChange={refresh} />}
        {tab === "curves" && (
          series.length === 0 || series.every((one) => one.points.length === 0) ? (
            <p className="mono muted">
              아직 기록이 없습니다 · 단계 표시줄의 4 학습으로 학습을 시작하거나, 스크립트에서 tf.log(step, loss=...)를 부르면 여기에 쌓입니다
            </p>
          ) : (
            <Curves series={series} label={CURVE_LABELS[key] ?? key} />
          )
        )}

        {tab === "block" && <BlockDetail />}

        {tab === "data" && (
          SYNTHETIC.has(dataset) ? (
            <div className="trainer">
              <button className="trainer__run" onClick={openData}>Input에 데이터 불러오기</button>
              <span className="mono muted">지금은 합성 과제로 학습합니다</span>
            </div>
          ) : (
            <>
              <div className="trainer">
                <button className="trainer__run" onClick={() => void apply()}>그래프에 적용</button>
                <span className="mono trainer__progress">{dataset}</span>
                {applied && <span className="mono trainer__note">{applied}</span>}
              </div>
              <DataCard name={dataset} recipe={recipe} onRecipe={(next) => setData(dataset, next)} />
            </>
          )
        )}

        {tab === "stdout" && <TrainLog />}

        {tab === "test" && <TestPanel />}

        {tab === "manifest" && (
          active?.manifest && Object.keys(active.manifest).length ? (
            <pre className="mono runpanel__json">
              {JSON.stringify(active.manifest, null, 2)}
            </pre>
          ) : <p className="mono muted">실행 설정 기록이 아직 없습니다</p>
        )}

        {tab === "logs" && (
          logs.length ? (
            <pre className="mono runpanel__json">
              {logs.map((line) =>
                `${line.node_id ? line.node_id + "  " : ""}${line.text}`).join("\n")}
            </pre>
          ) : <p className="mono muted">커널 로그가 비어 있습니다 · 블록 코드의 print와 모양 계산 오류가 여기에 쌓입니다</p>
        )}

        {tab === "console" && <Console />}
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
          <span key={one.run} className="curves__key" title={one.run}>
            <svg className="curves__swatch" width="18" height="4" aria-hidden>
              <line x1="0" y1="2" x2="18" y2="2" stroke={COLORS[index % COLORS.length]}
                    strokeWidth="2" strokeDasharray={DASHES[index % DASHES.length] || undefined} />
            </svg>
            {runLabelShort(one.run)}
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
            strokeDasharray={DASHES[index % DASHES.length] || undefined}
            vectorEffect="non-scaling-stroke"
            points={one.points.map((point) => `${sx(point[0])},${sy(point[1])}`).join(" ")}
          />
        ))}
      </svg>
    </div>
  );
}
