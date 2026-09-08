// Inspector - 박스 없는 `라벨 …… 값` 조밀한 행 (Figma Screens).
//
// 파라미터는 그 자리에서 고친다: 값을 바꾸면 `set_param` op가 나가고 서버가
// 승인한 그래프를 다시 읽는다(§8.2.2). `$hp`/`$p`/`$expr` 참조는 읽기 전용으로
// 둔다 - 참조를 푸는 UI는 승격(`promote_hp`)과 함께 와야 한다.

import { useEffect, useRef, useState } from "react";

import { fetchLogs, fetchRegistry } from "../api";
import type { LogLine } from "../api";
import { applyEdit } from "../edit";
import { op } from "../graph/ops";
import type { Block } from "../graph/ops";
import { SYNTHETIC, currentScope, useStore } from "../store";
import { formatRatio, formatShape } from "../theme";

export function Inspector() {
  const graph = useStore((state) => state.graph);
  const scopes = useStore((state) => state.scopes);
  const selected = useStore((state) => state.selected);
  const states = useStore((state) => state.nodeStates);
  const dataset = useStore((state) => state.dataset);
  const openData = useStore((state) => state.openData);
  const scope = currentScope({ graph, scopes });
  const stateKey = selected
    ? (scopes[scopes.length - 1].callPath
        ? `${scopes[scopes.length - 1].callPath}/${selected}` : selected)
    : "";
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [blocks, setBlocks] = useState<Block[]>([]);
  useEffect(() => { fetchRegistry().then(setBlocks).catch(() => undefined); }, []);

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
          <div><dt>연결 끊기</dt><dd>선 클릭 후 Delete</dd></div>
        </dl>
      </aside>
    );
  }

  const node = scope.nodes?.find((candidate) => candidate.id === selected);
  const instance = node?.call ? scope.instances?.[node.call] : undefined;
  const state = states[stateKey];
  const badges = state?.badges ?? {};
  const composite = scopes[scopes.length - 1].name === "$graph"
    ? null : scopes[scopes.length - 1].name;
  const kind = (node?.type ?? instance?.type ?? "").split("@")[0];
  // 묶음 블록의 인자 스키마는 레지스트리가 아니라 그래프 안의 컴포지트 정의에 있다.
  const schema: Record<string, { type: string; default?: unknown; choices?: unknown[] }> =
    kind.startsWith("composite:")
      ? ((graph?.composites?.[kind.slice("composite:".length)]?.params ?? {}) as
          Record<string, { type: string; default?: unknown; choices?: unknown[] }>)
      : (blocks.find((block) => block.type.split("@")[0] === kind)?.params ?? {});
  // 인자는 두 곳에 있다: 모듈 생성 인자는 인스턴스에, 호출 인자는 노드에.
  const owner = instance
    ? { instance: node?.call as string, args: instance.args ?? {} }
    : { node: selected, args: node?.args ?? {} };
  const fields = { ...Object.fromEntries(Object.keys(schema).map((name) => [name, undefined])),
                   ...owner.args };
  // 앞 블록의 출력 shape. 크기가 안 맞는 인자를 버튼 하나로 맞추는 데 쓴다.
  const callPath = scopes[scopes.length - 1].callPath;
  const incoming = (scope.edges ?? []).find(([, dst]) => String(dst).split(".")[0] === selected);
  const upstreamId = incoming ? String(incoming[0]).split(".")[0] : null;
  const upstream = upstreamId
    ? (states[callPath ? `${callPath}/${upstreamId}` : upstreamId]?.spec?.shape as (string | number)[] | undefined)
    : undefined;
  const compositePort = kind.startsWith("composite:")
    ? (graph?.composites?.[kind.slice("composite:".length)]?.ports?.in?.[0]?.shape as (string | number)[] | undefined)
    : undefined;
  const suggestion = suggestFix(kind, fields, upstream, compositePort);
  const showFix = suggestion && (state?.error || fields[suggestion.path] === undefined);

  return (
    <aside className="inspector" aria-label="Inspector">
      {/* 선택이 바뀌면 내용이 새로 떠오른다(key). 등록부 조회는 aside 바깥이라 다시 돌지 않는다. */}
      <div className="inspector__body" key={selected}>
      <h2>{node?.label}</h2>
      <p className="inspector__type">{(node?.type ?? instance?.type ?? "").split("@")[0]}</p>

      {kind !== "torchflow.Train" && (
        <>
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
        </>
      )}

      {kind === "torchflow.Input" && (
        <>
          <h3>데이터</h3>
          <dl className="rows">
            <div>
              <dt>데이터셋</dt>
              <dd>{SYNTHETIC.has(dataset) ? "없음 · 합성 과제" : dataset}</dd>
            </div>
          </dl>
          <button className="ghost inspector__action" onClick={openData}>
            데이터 불러오기
          </button>
        </>
      )}

      {kind === "torchflow.Input" && node?.ports_out?.length ? (
        <>
          <h3>입력 규격</h3>
          <dl className="rows">
            {node.ports_out.map((port, index) => (
              <div key={port.name}>
                <dt>{port.name}</dt>
                <dd>
                  <input
                    className="mono field field--wide"
                    defaultValue={(port.shape ?? []).join(", ")}
                    placeholder="B, 3, 32, 32"
                    spellCheck={false}
                    aria-label={`${port.name} shape`}
                    onKeyDown={(event) => {
                      if (event.key !== "Enter") return;
                      (event.target as HTMLInputElement).blur();
                    }}
                    onBlur={(event) => {
                      const shape = parseShape(event.target.value);
                      if (!shape) return;
                      const ports = node.ports_out!.map((entry, at) =>
                        at === index ? { ...entry, shape } : entry);
                      void applyEdit(op("set_ports", {
                        ...(composite ? { composite } : {}),
                        node: selected, ports_out: ports,
                      }));
                    }}
                  />
                </dd>
              </div>
            ))}
          </dl>
          <p className="mono muted">B는 배치 심볼입니다. 숫자와 심볼을 쉼표로 씁니다</p>
        </>
      ) : null}

      {Object.keys(fields).length > 0 && (
        <>
          <h3>Parameters</h3>
          <dl className="rows">
            {Object.entries(fields).map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd>
                  <ParamField
                    name={key}
                    value={value}
                    schema={schema[key]}
                    onCommit={(next) => void applyEdit(op("set_param", {
                      ...(composite ? { composite } : {}),
                      ...(owner.instance ? { instance: owner.instance } : { node: owner.node }),
                      path: key, value: next,
                    }))}
                  />
                </dd>
              </div>
            ))}
          </dl>
        </>
      )}

      {kind === "torchflow.Train" && (
        <p className="mono muted">
          값을 고치면 다음 실행에 반영됩니다 · 시작은 단계 표시줄의 4 실행 ·
          steps는 배치 수, batch는 한 번에 보는 샘플 수, lr은 한 걸음의 크기입니다
        </p>
      )}

      {showFix && suggestion && (
        <div className="inspector__fix">
          <p className="mono muted">{suggestion.why}</p>
          <button className="ghost" onClick={() => void applyEdit(op("set_param", {
            ...(composite ? { composite } : {}),
            ...(owner.instance ? { instance: owner.instance } : { node: owner.node }),
            path: suggestion.path, value: suggestion.value,
          }))}>
            {suggestion.path}를 {suggestion.value}로 맞추기
          </button>
        </div>
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
          {suggestion && (
            <p className="inspector__error">
              {fields[suggestion.path] === undefined
                ? `${suggestion.path}가 비어 있습니다 - 위의 맞추기 버튼이 앞 블록 출력에서 채웁니다`
                : `앞 블록의 출력과 ${suggestion.path} 값이 맞지 않습니다`}
            </p>
          )}
          <p className="mono inspector__error">{String(state.error.message)}</p>
        </>
      )}
      </div>
    </aside>
  );
}

/**
 * 값 하나를 고치는 칸. 타입은 레지스트리 스키마에서 오고, 없으면 값에서 짐작한다.
 *
 * 커밋 시점은 blur와 Enter다 - 글자 하나마다 op를 보내면 journal이 타자 기록이 된다.
 * 슬라이더와 pointer-up 코얼레싱(§5.5.1)은 아직 없다.
 */
/** 앞 블록 출력에서 바로 정해지는 인자. 사람이 곱셈으로 셀 필요가 없는 것들이다. */
const CHANNEL_ARG: Record<string, string> = {
  "torch.nn.Conv1d": "in_channels", "torch.nn.Conv2d": "in_channels", "torch.nn.Conv3d": "in_channels",
  "torch.nn.ConvTranspose2d": "in_channels",
  "torch.nn.BatchNorm1d": "num_features", "torch.nn.BatchNorm2d": "num_features",
  "torch.nn.BatchNorm3d": "num_features", "torch.nn.InstanceNorm2d": "num_features",
  "torch.nn.GroupNorm": "num_channels",
};

function suggestFix(kind: string, args: Record<string, unknown>,
                    upstream: (string | number)[] | undefined,
                    compositePort?: (string | number)[]):
    { path: string; value: number; why: string } | null {
  if (!upstream || !upstream.length) return null;
  const shape = `[${upstream.join(", ")}]`;
  const last = upstream[upstream.length - 1];
  if (compositePort) {
    // 묶음 블록의 입력 포트 [B, in_ch, H, W]에서 심볼 자리가 곧 인자 이름이다.
    for (let index = 1; index < compositePort.length; index += 1) {
      const symbol = compositePort[index];
      const value = upstream[index];
      if (typeof symbol === "string" && symbol !== "B" && typeof value === "number"
          && symbol in args && args[symbol] !== value && /^[a-z_]+$/i.test(symbol)) {
        return { path: symbol, value, why: `앞 블록 출력 ${shape}에서 ${symbol} 자리가 ${value}입니다` };
      }
    }
    return null;
  }
  if (kind === "torch.nn.Linear" && typeof last === "number" && args.in_features !== last) {
    return { path: "in_features", value: last, why: `앞 블록 출력 ${shape}의 마지막 차원이 ${last}입니다` };
  }
  if (kind === "torch.nn.LayerNorm" && typeof last === "number" && args.normalized_shape !== last) {
    return { path: "normalized_shape", value: last, why: `앞 블록 출력 ${shape}의 마지막 차원이 ${last}입니다` };
  }
  const arg = CHANNEL_ARG[kind];
  const channels = upstream[1];
  if (arg && typeof channels === "number" && args[arg] !== channels) {
    return { path: arg, value: channels, why: `앞 블록 출력 ${shape}의 채널 수가 ${channels}입니다` };
  }
  return null;
}

function ParamField({ name, value, schema, onCommit }: {
  name: string;
  value: unknown;
  schema?: { type: string; default?: unknown; choices?: unknown[] };
  onCommit: (next: unknown) => void;
}) {
  const reference = refOf(value);
  const kind = schema?.type ?? typeOf(value);
  const [draft, setDraft] = useState(() => text(value));
  const committed = useRef(text(value));

  useEffect(() => { setDraft(text(value)); committed.current = text(value); }, [value]);

  if (reference) {
    // 참조는 값이 아니라 다른 곳을 가리킨다. 여기서 고치면 참조가 끊긴다.
    return <span className="mono" title="하이퍼파라미터 참조 - 상단에서 값을 바꿉니다">{reference}</span>;
  }

  if (kind === "bool") {
    // 인자를 안 적었으면 torch 기본값이 돈다 - 체크박스도 그 값을 보여야 한다. 전에는 bias를
    // 안 적은 Conv2d가 꺼진 것처럼 보였다.
    const unset = value === undefined || value === null;
    return (
      <input
        type="checkbox" checked={unset ? schema?.default === true : value === true} aria-label={name}
        className={unset ? "field--default" : undefined}
        title={unset ? `기본값 ${String(schema?.default ?? false)}` : undefined}
        onChange={(event) => onCommit(event.target.checked)}
      />
    );
  }

  if (Array.isArray(schema?.choices) && schema.choices.length) {
    return (
      <select
        className="mono field" aria-label={name}
        value={String(value ?? schema.default ?? "")}
        onChange={(event) => onCommit(event.target.value)}
      >
        {schema.choices.map((choice) => (
          <option key={String(choice)} value={String(choice)}>{String(choice)}</option>
        ))}
      </select>
    );
  }

  const commit = () => {
    if (draft === committed.current) return;
    committed.current = draft;
    onCommit(parse(draft, kind));
  };

  return (
    <input
      className="mono field"
      value={draft}
      placeholder={schema?.default !== undefined ? String(schema.default) : "값 없음"}
      spellCheck={false}
      aria-label={name}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") { commit(); (event.target as HTMLInputElement).blur(); }
        if (event.key === "Escape") setDraft(committed.current);
      }}
    />
  );
}

/** ``"B, 3, 32, 32"`` -> ``["B", 3, 32, 32]``. 숫자는 숫자로, 나머지는 심볼로 남긴다. */
function parseShape(text: string): (string | number)[] | null {
  const parts = text.split(",").map((piece) => piece.trim()).filter(Boolean);
  if (!parts.length) return null;
  return parts.map((piece) => (/^-?\d+$/.test(piece) ? Number(piece) : piece));
}

function refOf(value: unknown): string | null {
  if (!value || typeof value !== "object") return null;
  const entries = Object.entries(value as Record<string, unknown>);
  if (entries.length === 1 && entries[0][0].startsWith("$")) {
    return `${entries[0][0]}: ${entries[0][1]}`;
  }
  return null;
}

function typeOf(value: unknown): string {
  if (typeof value === "boolean") return "bool";
  if (typeof value === "number") return Number.isInteger(value) ? "int" : "float";
  return "str";
}

function text(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/** 빈 칸은 null - 서버가 인자를 지우고 블록의 기본값으로 돌아간다. */
function parse(draft: string, kind: string): unknown {
  const trimmed = draft.trim();
  if (!trimmed) return null;
  if (kind === "int" || kind === "float") {
    const parsed = Number(trimmed);
    return Number.isNaN(parsed) ? trimmed : parsed;
  }
  if (trimmed === "true" || trimmed === "false") return trimmed === "true";
  try {
    return JSON.parse(trimmed);
  } catch {
    return trimmed;
  }
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
