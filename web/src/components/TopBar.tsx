// 상단 바 - Figma Screens 기준. 크롬은 무채색이고 색은 그래프에서만 나온다.

import { useEffect, useState } from "react";
import { closeGraph, estimateMemory, runProbe, saveGraph } from "../api";
import { applyEdit } from "../edit";
import { op } from "../graph/ops";
import { useStore } from "../store";
import { formatCount } from "../theme";

export function TopBar() {
  const scopes = useStore((state) => state.scopes);
  const popToScope = useStore((state) => state.popToScope);
  const connected = useStore((state) => state.connected);
  const kernelAlive = useStore((state) => state.kernelAlive);
  const totals = useStore((state) => state.totals);
  const setTotals = useStore((state) => state.setTotals);
  const graph = useStore((state) => state.graph);
  const attached = useStore((state) => state.attached);
  const probing = useStore((state) => state.probing);
  const probeObjective = useStore((state) => state.probeObjective);
  const gradOverlay = useStore((state) => state.gradOverlay);
  const tab = useStore((state) => state.tab);
  const setTab = useStore((state) => state.setTab);
  const toggleGradOverlay = useStore((state) => state.toggleGradOverlay);
  const setProbing = useStore((state) => state.setProbing);
  const token = useStore((state) => state.token);
  const dirty = useStore((state) => state.dirty);
  const setDirty = useStore((state) => state.setDirty);
  const [saving, setSaving] = useState<string | null>(null);

  const home = async () => {
    if (dirty && !window.confirm("저장하지 않은 편집이 있습니다. 첫 화면으로 나갈까요?")) return;
    await closeGraph();
    useStore.getState().closeGraph();
  };

  const name = graph?.graph.name ?? "";
  const rename = (next: string) => {
    const trimmed = next.trim();
    if (!trimmed || trimmed === name) return;
    // 역 op가 이전 이름을 알아야 되돌릴 수 있다(§8.2.3).
    void applyEdit(op("rename", { name: trimmed, previous: name }));
  };

  const save = async () => {
    setSaving("…");
    const result = await saveGraph();
    setSaving(result.error ? result.error : null);
    if (!result.error) setDirty(false);
  };

  const probe = async () => {
    setProbing(true);
    try {
      await runProbe();
    } finally {
      setProbing(false);
    }
  };

  useEffect(() => {
    if (!graph) return;
    let cancelled = false;
    estimateMemory(totals.batch)
      .then((estimate) => {
        if (!cancelled && estimate.ok) setTotals({ band: estimate.band_gb as [number, number] });
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [graph, totals.batch, setTotals]);

  return (
    <header className="topbar">
      <button className="wordmark wordmark--home" onClick={home} title="첫 화면으로">
        <span className="wordmark__dot" aria-hidden />
        torchflow
        <span className="wordmark__version">0.0.1</span>
      </button>

      <input
        className="graphname"
        defaultValue={name}
        key={name}
        aria-label="그래프 이름"
        spellCheck={false}
        onBlur={(event) => rename(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") (event.target as HTMLInputElement).blur();
          if (event.key === "Escape") (event.target as HTMLInputElement).value = name;
        }}
      />

      <span className="savestate">
        <button className="savestate__button" onClick={save} disabled={saving === "…"}>
          저장
        </button>
        <span className="savestate__label">
          {saving && saving !== "…" ? saving : dirty ? "저장 안 됨" : "저장됨"}
        </span>
      </span>

      <nav className="crumbs" aria-label="그래프 경로">
        {(scopes.length > 1 ? scopes : []).map((scope, index) => (
          <span key={scope.callPath || scope.name}>
            {index > 0 && <span className="crumbs__sep">›</span>}
            <button
              className="crumbs__item"
              onClick={() => popToScope(index)}
              aria-current={index === scopes.length - 1 ? "page" : undefined}
            >
              {scope.label}
            </button>
          </span>
        ))}
      </nav>

      <nav className="tabs" aria-label="화면">
        <button className={`tabs__item${tab === "model" ? " tabs__item--on" : ""}`}
                onClick={() => setTab("model")}>Model</button>
        <button className={`tabs__item${tab === "code" ? " tabs__item--on" : ""}`}
                onClick={() => setTab("code")}>Code</button>
        <button className="tabs__item" disabled title="최소 L2 워커는 M7입니다">Experiment</button>
        <button className="tabs__item" disabled title="run 비교는 v1입니다">Runs</button>
      </nav>

      <div className="probe">
        <button
          className="probe__run"
          onClick={probe}
          disabled={probing || attached}
          title={attached
            ? "Attach 모드에서는 사용자 프로세스가 probe를 돌립니다 (sess.probe())"
            : "L1 probe 1회 — 목적함수의 조상 폐쇄를 forward+backward"}
        >
          {probing ? "Probe…" : "Probe"}
        </button>
        {probeObjective && <span className="probe__obj">obj: {probeObjective}</span>}
        <label className="probe__toggle">
          <input type="checkbox" checked={gradOverlay} onChange={toggleGradOverlay} />
          Grad-Flow
        </label>
        {attached && <span className="badge badge--attach">attached</span>}
      </div>

      {/* 논문용 그림(§6.4). 토큰이 쿼리로 지나가므로 링크 하나면 받아진다. */}
      <div className="export">
        <span className="export__label">그림</span>
        {(["svg", "pdf"] as const).map((format) => (
          <a
            key={format}
            className="export__link"
            href={`/api/export?format=${format}&token=${encodeURIComponent(token)}`}
            title="흑백 안전 · IR 해시 포함 (NeurIPS 5.5 in)"
          >
            {format.toUpperCase()}
          </a>
        ))}
      </div>

      <div className="totals">
        <span className="totals__item">Σ {formatCount(totals.params)} params</span>
        {totals.band && (
          <span className="totals__item" title="cuDNN workspace·할당자 파편화를 포함한 밴드">
            est. {totals.band[0].toFixed(2)}–{totals.band[1].toFixed(2)} GB @B={totals.batch}
          </span>
        )}
        <label className="totals__batch">
          B
          <input
            type="range" min={1} max={256} step={1} value={totals.batch}
            onChange={(event) => setTotals({ batch: Number(event.target.value) })}
            aria-label="추정 배치 크기"
          />
        </label>
      </div>

      <div className="status">
        <span className={`dot ${connected ? "dot--on" : "dot--off"}`} />
        hub
        <span className={`dot ${kernelAlive ? "dot--on" : "dot--off"}`} />
        L0
      </div>
    </header>
  );
}
