// IR 스코프 하나 -> React Flow 노드/엣지.

import type { Edge, Node as FlowNode } from "@xyflow/react";
import type { Composite, Graph, NodeState } from "../types.gen";
import type { Lod, NodeCardData } from "../components/NodeCard";
import type { NodeStateName } from "../theme";
import { DTYPE_COLOR, LOD_WIDTH, categoryOf, edgeWidth, gradColor, gradRange } from "../theme";
import { layeredLayout } from "./layout";

// LOD별 노드 높이 어림값(flow px). 겹침 판정에만 쓴다.
const NODE_HEIGHT: Record<Lod, number> = { far: 44, mid: 84, near: 128, focus: 128 };
const OVERLAP_GAP = 24;

/**
 * 사용자가 옮긴 노드(layout.json)와 자동 배치 노드는 서로를 모른다 - 실제로 Input 위에
 * Output이 얹혀 더블클릭이 막혔다. 고정 노드는 두고, 자동 배치 노드가 이미 놓인 것과 겹치면
 * 그 아래로 밀어낸다.
 */
function resolveOverlaps(nodes: FlowNode[], fixed: Set<string>, lod: Lod): void {
  const width = LOD_WIDTH[lod];
  const height = NODE_HEIGHT[lod];
  const placed = nodes.filter((node) => fixed.has(node.id));
  for (const node of nodes) {
    if (fixed.has(node.id)) continue;
    for (let guard = 0; guard < 24; guard += 1) {
      const hit = placed.find((other) =>
        Math.abs(other.position.x - node.position.x) < width + OVERLAP_GAP
        && Math.abs(other.position.y - node.position.y) < height + OVERLAP_GAP);
      if (!hit) break;
      node.position = { x: node.position.x, y: hit.position.y + height + OVERLAP_GAP };
    }
    placed.push(node);
  }
}

export function lodOf(zoom: number): Lod {
  // 0.55 아래에서는 12px 본문이 화면에서 7px도 안 된다 - 제목만 남기고 글자를 줌에 반비례해 키운다(CSS --zoom).
  if (zoom < 0.55) return "far";
  if (zoom < 0.9) return "mid";
  if (zoom < 1.6) return "near";
  return "focus";
}

/** 이 노드가 들어갈 수 있는 컴포지트인가 - 브레드크럼 진입 대상(§2.2). */
export function enterableComposite(scope: Graph | Composite, nodeId: string): string | null {
  const node = scope.nodes?.find((candidate) => candidate.id === nodeId);
  if (!node?.call) return null;
  const instance = scope.instances?.[node.call];
  if (!instance) return null;
  for (const candidate of [instance.type, instance.body]) {
    if (typeof candidate === "string" && candidate.startsWith("composite:")) {
      return candidate.slice("composite:".length);
    }
  }
  // Switch는 활성 변형으로 들어간다.
  const active = typeof instance.active === "string" ? instance.active : null;
  const variant = active ? instance.variants?.[active] : undefined;
  const variantType = variant?.type;
  if (typeof variantType === "string" && variantType.startsWith("composite:")) {
    return variantType.slice("composite:".length);
  }
  return null;
}

/** Inspector와 노드 본문에 같은 표기를 쓴다: 참조는 `$hp: dim` 그대로. */
function renderArg(value: unknown): string {
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 1 && entries[0][0].startsWith("$")) return String(entries[0][1]);
    return JSON.stringify(value);
  }
  return String(value);
}

function formatShapeLabel(shape: (string | number)[] | undefined): string {
  return shape?.length ? `[${shape.join(", ")}]` : "?";
}

export function toFlow(
  scope: Graph | Composite,
  states: Record<string, NodeState>,
  options: {
    lod: Lod; selected: string | null; focused: string | null; callPath: string;
    gradOverlay?: boolean;
    positions?: Record<string, { x: number; y: number }>;
    /** 그래프에 붙은 데이터 이름. 합성 과제면 빈 문자열. */
    dataset?: string;
    /** 마지막 run 한 줄. Train 노드 배지에 걸린다. */
    run?: string;
  },
): { nodes: FlowNode[]; edges: Edge[] } {
  const irNodes = scope.nodes ?? [];
  const irEdges = (scope.edges ?? []) as [string, string][];
  const positions = layeredLayout(irNodes, irEdges);

  const keyOf = (id: string) => (options.callPath ? `${options.callPath}/${id}` : id);
  const stateOf = (id: string) => states[keyOf(id)];

  /**
   * 접힌 컴포지트는 안쪽을 대신 말해야 한다(§6.1 "내부 최악 상태").
   * 자식 노드의 상태 키는 부모 키를 접두어로 갖는다.
   */
  const foldedOf = (id: string) => {
    const base = keyOf(id);
    let worst: number | undefined;
    let warn: string | undefined;
    for (const [key, value] of Object.entries(states)) {
      // Repeat의 회차는 `#0`으로, 컴포지트 진입은 `/`로 이어진다.
      if (!key.startsWith(`${base}/`) && !key.startsWith(`${base}#`)) continue;
      const norm = value.badges?.grad_norm as number | undefined;
      if (typeof norm === "number" && (worst === undefined || norm > worst)) worst = norm;
      warn = warn ?? (value.badges?.grad_warn as string | undefined);
    }
    return { gradNorm: worst, gradWarn: warn };
  };

  const norms = irNodes
    .map((node) => stateOf(node.id)?.badges?.grad_norm as number | undefined)
    .filter((value): value is number => typeof value === "number");
  const range = gradRange(norms);

  const nodes: FlowNode[] = irNodes.map((node) => {
    const state = stateOf(node.id);
    const own = state?.badges?.grad_norm as number | undefined;
    const folded = own === undefined ? foldedOf(node.id) : undefined;
    const gradNorm = own ?? folded?.gradNorm;
    const instance = node.call ? scope.instances?.[node.call] : undefined;
    const typeLabel = (node.type ?? instance?.type ?? "").split("@")[0];
    const data: NodeCardData = {
      label: node.label,
      category: categoryOf(node.type ?? instance?.type, Boolean(node.call)),
      typeLabel: typeLabel.replace(/^torch\.nn\.|^torchflow\.|^composite:/, ""),
      params: Object.entries(instance?.args ?? node.args ?? {})
        .slice(0, 3)
        .map(([key, value]) => `${key}=${renderArg(value)}`)
        .join("  "),
      paramCount: state?.badges?.params as number | undefined,
      state: (state?.state as NodeStateName) ?? "idle",
      shape: state?.spec?.shape as (string | number)[] | undefined,
      dtype: state?.spec?.dtype as string | undefined,
      elapsedMs: state?.badges?.elapsed_ms as number | undefined,
      cacheHit: state?.badges?.cache_hit as boolean | undefined,
      errorMessage: state?.error?.message as string | undefined,
      gradNorm,
      gradRatio: state?.badges?.grad_ratio as number | undefined,
      gradWarn: (state?.badges?.grad_warn ?? state?.badges?.warn ?? folded?.gradWarn) as
        string | undefined,
      folded: own === undefined && gradNorm !== undefined,
      gradColor: options.gradOverlay && gradNorm ? gradColor(gradNorm, range) : undefined,
      histogram: state?.badges?.histogram as number[] | undefined,
      feature: state?.badges?.feature as
        { size: number; pixels: string; channels: number; shown: number } | undefined,
      enterable: enterableComposite(scope, node.id) !== null,
      // Input에는 붙은 데이터 이름이 걸린다. 없으면 빈 문자열 - 카드가 "데이터 없음"으로 읽는다.
      dataset: typeLabel === "torchflow.Input" ? options.dataset ?? "" : undefined,
      run: typeLabel === "torchflow.Train" ? options.run ?? "" : undefined,
      lod: options.lod,
      selected: options.selected === node.id,
      focused: options.focused === node.id,
    };
    return {
      id: node.id,
      type: "tf",
      // 사용자가 옮긴 좌표가 있으면 그것이 자동 배치를 이긴다.
      position: options.positions?.[keyOf(node.id)] ?? positions[node.id],
      data: data as unknown as Record<string, unknown>,
      selectable: true,
    };
  });
  resolveOverlaps(nodes, new Set(irNodes.filter((node) => options.positions?.[keyOf(node.id)])
    .map((node) => node.id)), options.lod);

  const edges: Edge[] = irEdges
    .filter(([src, dst]) => !src.startsWith("$in") && !dst.startsWith("$out"))
    .map(([src, dst]) => {
      const [source, sourcePort] = src.split(".");
      const [target] = dst.split(".");
      const spec = stateOf(source)?.spec as
        { shape?: (string | number)[]; dtype?: string } | undefined;
      return {
        id: `${src}->${dst}`,
        source,
        target,
        // 끊을 때 disconnect op에 그대로 실린다.
        data: { src, dst },
        // 굵기 = log(원소 수), 색 = dtype, 라벨 = shape (§6.1).
        style: {
          strokeWidth: edgeWidth(spec?.shape),
          stroke: DTYPE_COLOR[spec?.dtype ?? ""] ?? "var(--border-default)",
        },
        label: options.lod === "far" ? undefined : formatShapeLabel(spec?.shape),
        labelStyle: { fontSize: 10, fill: "var(--text-muted)" },
        labelBgStyle: { fill: "var(--surface-canvas)", fillOpacity: 0.9 },
        ariaLabel: `${sourcePort} → ${target}`,
      } satisfies Edge;
    });

  // 경계에 붙지 않는 노드도 스코프 안에서는 보여야 한다(컴포지트 $in/$out).
  return { nodes, edges };
}
