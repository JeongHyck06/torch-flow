// 위상 층 배치. layout.json이 아직 없을 때의 기본 좌표를 만든다.
//
// ELK 자동 정렬(`Cmd+L`)은 M5다. 여기서는 의존성 없이 층만 나눈다 - 노드가
// 왼쪽에서 오른쪽으로 흐르고, 같은 층은 세로로 쌓인다.

import type { Node } from "../types.gen";

export interface Position { x: number; y: number }

const COLUMN = 260;
const ROW = 104;

// 깊은 모델(ResNet-18은 20층, DiT는 ×28)은 한 줄로 펴면 5000 px가 넘어 첫 화면이
// 읽히지 않는다. 층을 밴드로 접어 넣는다. 사용자가 옮긴 좌표(layout.json)가
// 생기면 그것이 우선한다 - M5.
const MAX_COLUMNS = 10;
const BAND_GAP = 220;

/** 노드별 층 깊이 = 가장 긴 선행 경로 길이. 위상 순서의 정본이다. */
export function computeLayers(nodes: Node[], edges: [string, string][]): Map<string, number> {
  const ids = new Set(nodes.map((node) => node.id));
  const indegree = new Map<string, number>(nodes.map((node) => [node.id, 0]));
  const successors = new Map<string, string[]>(nodes.map((node) => [node.id, []]));

  for (const [src, dst] of edges) {
    const from = src.split(".")[0];
    const to = dst.split(".")[0];
    if (!ids.has(from) || !ids.has(to)) continue;
    successors.get(from)!.push(to);
    indegree.set(to, indegree.get(to)! + 1);
  }

  // 층 = 가장 긴 선행 경로 길이. 사이클이 있어도 멈추지 않는다.
  const layer = new Map<string, number>();
  const queue = nodes.filter((node) => indegree.get(node.id) === 0).map((node) => node.id);
  queue.forEach((id) => layer.set(id, 0));

  while (queue.length) {
    const current = queue.shift()!;
    for (const next of successors.get(current) ?? []) {
      layer.set(next, Math.max(layer.get(next) ?? 0, (layer.get(current) ?? 0) + 1));
      indegree.set(next, indegree.get(next)! - 1);
      if (indegree.get(next) === 0) queue.push(next);
    }
  }

  return layer;
}

export function layeredLayout(
  nodes: Node[], edges: [string, string][],
): Record<string, Position> {
  const layer = computeLayers(nodes, edges);
  const rows = new Map<number, number>();
  const positions: Record<string, Position> = {};
  for (const node of nodes) {
    const depth = layer.get(node.id) ?? 0;
    const band = Math.floor(depth / MAX_COLUMNS);
    const column = depth % MAX_COLUMNS;
    const row = rows.get(depth) ?? 0;
    rows.set(depth, row + 1);
    positions[node.id] = { x: column * COLUMN, y: band * BAND_GAP + row * ROW };
  }
  return positions;
}

/** 위상 순 노드 id — 키보드 탐색이 이 순서를 따른다(§2.2). */
export function topologicalIds(nodes: Node[], edges: [string, string][]): string[] {
  const layer = computeLayers(nodes, edges);
  return [...nodes]
    .sort((a, b) =>
      (layer.get(a.id) ?? 0) - (layer.get(b.id) ?? 0) || a.label.localeCompare(b.label))
    .map((node) => node.id);
}
