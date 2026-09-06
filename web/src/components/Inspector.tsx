// Inspector - 박스 없는 `라벨 …… 값` 조밀한 행 (Figma Screens).
// 파라미터 편집 폼은 M5(IR 편집 op). 지금은 읽기 전용이다.

import { useEffect, useState } from "react";

import { fetchLogs } from "../api";
import type { LogLine } from "../api";
import { currentScope, useStore } from "../store";
import { formatRatio, formatShape } from "../theme";

export function Inspector() {
  const graph = useStore((state) => state.graph);
  const scopes = useStore((state) => state.scopes);
  const selected = useStore((state) => state.selected);
  const states = useStore((state) => state.nodeStates);
  const scope = currentScope({ graph, scopes });
  const stateKey = selected
    ? (scopes[scopes.length - 1].callPath
        ? `${scopes[scopes.length - 1].callPath}/${selected}` : selected)
    : "";
  const [logs, setLogs] = useState<LogLine[]>([]);

  // 이 노드가 찍은 stdout/stderr(§5.6.1). 하단 로그 탭과 같은 테이블을 읽는다.
  useEffect(() => {
    if (!stateKey) return setLogs([]);
    let cancelled = false;
    fetchLogs(stateKey).then((lines) => { if (!cancelled) setLogs(lines); })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [stateKey, states[stateKey]?.seq]);

  if (!scope || !selected) {
    return (
      <aside className="inspector" aria-label="Inspector">
        <h3>단축키</h3>
        <dl className="rows">
          <div><dt>노드 이동</dt><dd>← →</dd></div>
          <div><dt>선택</dt><dd>Enter</dd></div>
          <div><dt>컴포지트 진입</dt><dd>]</dd></div>
          <div><dt>나가기</dt><dd>Esc</dd></div>
          <div><dt>전체 보기</dt><dd>f</dd></div>
          <div><dt>위치 조정</dt><dd>드래그</dd></div>
        </dl>
      </aside>
    );
  }

  const node = scope.nodes?.find((candidate) => candidate.id === selected);
  const instance = node?.call ? scope.instances?.[node.call] : undefined;
  const state = states[stateKey];
  const args = { ...(instance?.args ?? {}), ...(node?.args ?? {}) };
  const badges = state?.badges ?? {};

  return (
    <aside className="inspector" aria-label="Inspector">
      <h2>{node?.label}</h2>
      <p className="inspector__type">{(node?.type ?? instance?.type ?? "").split("@")[0]}</p>

      <h3>Module</h3>
      <dl className="rows">
        <div>
          <dt>출력</dt>
          <dd>{formatShape(state?.spec?.shape as (string | number)[] | undefined)}</dd>
        </div>
        {state?.spec?.dtype ? <div><dt>dtype</dt><dd>{String(state.spec.dtype)}</dd></div> : null}
        {badges.measured ? (
          <div>
            <dt>실측 (L1)</dt>
            <dd>{formatShape((badges.measured as { shape?: (string | number)[] }).shape)}</dd>
          </div>
        ) : null}
        {badges.elapsed_ms !== undefined ? (
          <div>
            <dt>{badges.cache_hit ? "캐시" : "실행"}</dt>
            <dd>{badges.cache_hit ? "hit" : `${Number(badges.elapsed_ms).toFixed(1)} ms`}</dd>
          </div>
        ) : null}
      </dl>

      {Object.keys(args).length > 0 && (
        <>
          <h3>Parameters</h3>
          <dl className="rows">
            {Object.entries(args).map(([key, value]) => (
              <div key={key}><dt>{key}</dt><dd>{renderValue(value)}</dd></div>
            ))}
          </dl>
        </>
      )}

      {badges.grad_ratio !== undefined && (
        <>
          <h3>Gradient</h3>
          <dl className="rows">
            <div><dt>‖g‖</dt><dd>{Number(badges.grad_norm).toExponential(2)}</dd></div>
            <div>
              <dt>‖g‖/‖w‖</dt>
              <dd className={badges.grad_warn ? "warn" : undefined}>
                {formatRatio(badges.grad_ratio as number)}
              </dd>
            </div>
            {badges.grad_warn ? (
              <div><dt>진단</dt><dd className="warn">~ {String(badges.grad_warn)}</dd></div>
            ) : null}
          </dl>
          {badges.probe_objective ? (
            <p className="mono muted">{String(badges.probe_objective)}</p>
          ) : null}
        </>
      )}

      {Array.isArray(badges.histogram) && (
        <>
          <h3>활성값 분포</h3>
          <Histogram bins={badges.histogram as number[]} />
        </>
      )}

      {logs.length > 0 && (
        <>
          <h3>출력</h3>
          <pre className="mono inspector__stdout">
            {logs.map((line) => line.text).join("\n")}
          </pre>
        </>
      )}

      {state?.error && (
        <>
          <h3>진단 · {String(state.error.kind)}</h3>
          <p className="mono inspector__error">{String(state.error.message)}</p>
        </>
      )}
    </aside>
  );
}

/** 히스토그램 스파크라인. 차트 라이브러리 없이 SVG 막대 하나면 충분하다. */
function Histogram({ bins }: { bins: number[] }) {
  const peak = Math.max(...bins, 1);
  const width = 284;
  const height = 52;
  const step = width / bins.length;
  return (
    <svg width={width} height={height} role="img" aria-label={`${bins.length}빈 히스토그램`}>
      {bins.map((count, index) => (
        <rect
          key={index}
          x={index * step}
          y={height - (count / peak) * height}
          width={Math.max(step - 0.5, 0.5)}
          height={(count / peak) * height}
          fill="var(--dtype-f32)"
        />
      ))}
    </svg>
  );
}

function renderValue(value: unknown): string {
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 1 && entries[0][0].startsWith("$")) {
      return `${entries[0][0]}: ${String(entries[0][1])}`;
    }
    return JSON.stringify(value);
  }
  return String(value);
}
