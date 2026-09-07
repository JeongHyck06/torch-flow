// 블록 상세 탭 (기획서 §6.3). 선택한 블록이 마지막 probe에서 무엇을 했는지 그린다.
//
// 그림은 커널이 만든다(입력·출력 채널 격자, 필터, 가중치 히트맵, 전후 분포). 여기서는
// 받은 패널을 종류대로 늘어놓을 뿐이다. probe가 다시 돌면(노드 상태가 바뀌면) 새로 받는다.

import { useEffect, useState } from "react";

import { fetchDetail } from "../api";
import type { BlockDetailInfo, DetailPanel } from "../api";
import { useStore } from "../store";
import { formatCount, formatShape } from "../theme";

export function BlockDetail() {
  const selected = useStore((state) => state.selected);
  const scopes = useStore((state) => state.scopes);
  const nodeStates = useStore((state) => state.nodeStates);
  const callPath = scopes[scopes.length - 1]?.callPath ?? "";
  const key = selected ? (callPath ? `${callPath}/${selected}` : selected) : "";
  const state = key ? nodeStates[key] : undefined;
  const [detail, setDetail] = useState<BlockDetailInfo | null>(null);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    if (!key) { setDetail(null); return; }
    let cancelled = false;
    fetchDetail(key).then((body) => {
      if (cancelled) return;
      setDetail(body.ok ? body : null);
      setNote(body.ok ? null : body.error ?? "읽지 못했습니다");
    }).catch((reason) => { if (!cancelled) setNote(String(reason)); });
    return () => { cancelled = true; };
  }, [key, state]);

  if (!selected) {
    return <p className="mono muted">캔버스나 모듈 트리에서 블록을 고르면 그 블록이 무엇을 했는지 보입니다</p>;
  }
  if (note) return <p className="mono muted">{note}</p>;
  if (!detail) return <p className="mono muted">읽는 중</p>;

  return (
    <div className="blockdetail">
      <div className="blockdetail__head">
        <h2>{detail.label} <span className="mono muted">{detail.kind.replace(/^torch\.nn\.|^torchflow\./, "")}</span></h2>
        <p>{detail.explain}</p>
        <p className="mono muted">
          {detail.input_shape ? `${formatShape(detail.input_shape)} → ` : ""}{formatShape(detail.output_shape ?? undefined)}
          {detail.params ? ` · ${formatCount(detail.params)} params` : " · 파라미터 없음"}
        </p>
      </div>
      <div className="blockdetail__panels">
        {detail.panels.map((panel, index) => <Panel key={index} panel={panel} />)}
      </div>
    </div>
  );
}

function Panel({ panel }: { panel: DetailPanel }) {
  if (panel.type === "grid" || panel.type === "heatmap") {
    return (
      <figure className="blockdetail__panel">
        <figcaption>{panel.title}</figcaption>
        <img src={`data:image/png;base64,${panel.png}`} alt={panel.title}
             style={{ width: panel.type === "heatmap" ? 160 : undefined }} />
        {panel.note && <span className="mono muted">{panel.note}</span>}
      </figure>
    );
  }
  if (panel.type === "hist") return <Hist panel={panel} />;
  if (panel.type === "bars") return <Bars panel={panel} />;
  return <p className="mono muted">{panel.title}</p>;
}

/** 전(회색)·후(파랑) 분포를 같은 축에 겹친다. 무엇이 잘리고 무엇이 퍼졌는지 보인다. */
function Hist({ panel }: { panel: DetailPanel }) {
  const width = 260;
  const height = 80;
  const series = [panel.before, panel.after].filter((one): one is number[] => Array.isArray(one));
  const peak = Math.max(1, ...series.flat());
  const path = (bins: number[]) => {
    const step = width / bins.length;
    return bins.map((count, index) =>
      `${index === 0 ? "M" : "L"}${index * step},${height - (count / peak) * height}`).join(" ")
      + ` L${width},${height} L0,${height} Z`;
  };
  const [lo, hi] = panel.range ?? [0, 1];
  return (
    <figure className="blockdetail__panel">
      <figcaption>{panel.title}</figcaption>
      <svg className="blockdetail__hist" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none"
           role="img" aria-label={panel.title}>
        {panel.before && <path d={path(panel.before)} fill="var(--border-default)" />}
        {panel.after && <path d={path(panel.after)} fill="var(--dtype-f32)" fillOpacity={0.75} />}
      </svg>
      <span className="mono muted">{lo.toFixed(2)} .. {hi.toFixed(2)}{panel.before ? " · 회색 전, 파랑 후" : ""}</span>
    </figure>
  );
}

function Bars({ panel }: { panel: DetailPanel }) {
  const values = panel.values ?? [];
  const before = panel.before ?? [];
  const scale = Math.max(1e-6, ...values.map(Math.abs), ...before.map(Math.abs));
  return (
    <figure className="blockdetail__panel">
      <figcaption>{panel.title}</figcaption>
      <div className="blockdetail__bars">
        {values.map((value, index) => (
          <div key={index} className="blockdetail__bar mono">
            <span className="muted">{panel.labels?.[index] ?? index}</span>
            <span className="blockdetail__track">
              {before[index] !== undefined && (
                <span className="blockdetail__fill blockdetail__fill--before"
                      style={{ width: `${(Math.abs(before[index]) / scale) * 100}%` }} />
              )}
              <span className="blockdetail__fill" style={{ width: `${(Math.abs(value) / scale) * 100}%` }} />
            </span>
            <span>{value.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </figure>
  );
}
