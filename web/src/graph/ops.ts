// 편집 op를 만들고 뒤집는다 (기획서 §8.2.2, §8.2.3).
//
// 실행 취소는 서버 히스토리를 되감지 않는다. 클라이언트가 **역 op를 새 op로**
// 보내고 서버는 그것을 평범한 편집으로 적용한다. 그래서 여기서 만드는 역 op가
// 정확해야 하고, 그 정확성은 편집 직전 그래프를 보고 계산하는 데서 온다.

import type { Composite, Graph, Instance, Node } from "../types.gen";

export interface Op {
  client_id: string;
  tmp_seq: number;
  kind: string;
  payload: Record<string, unknown>;
  ops?: Op[];
  inverse_of?: number;
}

export interface Block {
  type: string;
  label: string;
  category: string;
  params: Record<string, { type: string; required?: boolean; default?: unknown }>;
  ports: { in: string[]; out: string[] };
  doc?: string;
  source: "builtin" | "reflection" | string;
}

const CLIENT_ID = Math.random().toString(36).slice(2, 10);
let counter = 0;

/** 이 탭의 식별자. 브로드캐스트에서 내 편집을 가려내는 데 쓴다(§8.2.2). */
export function clientId(): string {
  return CLIENT_ID;
}

/** ULID를 흉내 낸 26자 id. 시간 앞자리 덕에 정렬하면 만든 순서가 된다. */
export function newId(): string {
  const time = Date.now().toString(32).toUpperCase().padStart(10, "0");
  const random = Math.random().toString(32).slice(2, 12).toUpperCase();
  return `${time}${random}`.slice(0, 16);
}

export function op(kind: string, payload: Record<string, unknown>, ops?: Op[]): Op {
  counter += 1;
  return { client_id: CLIENT_ID, tmp_seq: counter, kind, payload, ...(ops ? { ops } : {}) };
}

/** 스코프 안에서 겹치지 않는 라벨. 라벨은 생성 코드에서 속성 이름이 된다(§7.2). */
export function uniqueLabel(scope: Graph | Composite, base: string): string {
  const taken = new Set((scope.nodes ?? []).map((node) => node.label));
  const stem = base.replace(/[^A-Za-z0-9_]/g, "_").toLowerCase();
  if (!taken.has(stem)) return stem;
  for (let index = 2; ; index += 1) {
    if (!taken.has(`${stem}${index}`)) return `${stem}${index}`;
  }
}

/**
 * 팔레트에서 고른 블록 하나를 `add_node` op로.
 *
 * `torch.nn.*`은 살아 있는 모듈이라 인스턴스를 함께 만들고, 내장 텐서 연산과
 * 구조 노드는 `type`만 있는 순수 호출이다.
 */
export function addBlockOp(
  scope: Graph | Composite, block: Block, composite: string | null,
): { op: Op; nodeId: string } {
  const nodeId = newId();
  const label = uniqueLabel(scope, block.label);
  const payload: Record<string, unknown> = composite ? { composite } : {};

  if (block.source === "reflection") {
    payload.instance = { id: newId(), label, type: block.type, args: defaultArgs(block) };
    payload.node = { id: nodeId, label, method: "forward" };
  } else {
    payload.node = { id: nodeId, label, type: block.type, ...inputPorts(block) };
  }
  return { op: op("add_node", payload), nodeId };
}

/** 기본값이 있는 파라미터만 채운다. 필수인데 빈 것은 L0가 노드에 귀속해 알려준다(§5.6). */
function defaultArgs(block: Block): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(block.params)
      .filter(([, schema]) => schema.default !== undefined)
      .map(([name, schema]) => [name, schema.default]),
  );
}

/** Input 노드는 출력 포트가 곧 그래프의 입력 규격이다. 빈 채로 두면 L0가 멈춘다. */
function inputPorts(block: Block): Record<string, unknown> {
  if (block.type !== "torchflow.Input") return {};
  return { ports_out: [{ name: "x", type: "Tensor", shape: ["B", 3, 32, 32], dtype: "float32" }] };
}

/** 노드 하나를 지우는 op. 지워진 엣지는 역 op(:func:`inverseOf`)가 되살린다. */
export function removeNodeOp(nodeId: string, composite: string | null): Op {
  return op("remove_node", { ...(composite ? { composite } : {}), node: nodeId });
}

/**
 * op의 역. 편집 **직전** 그래프를 받아 계산한다.
 *
 * 되돌릴 수 없는 op(아직 그래프에 적용되지 않는 kind)는 `null`이고, 그런 op는
 * 실행 취소 스택에 들어가지 않는다 - 스택에 넣고 아무 일도 안 하는 것보다 낫다.
 */
export function inverseOf(scope: Graph | Composite, current: Op): Op | null {
  const payload = current.payload;
  const composite = (payload.composite as string | undefined) ?? null;
  const scoped = composite ? { composite } : {};

  switch (current.kind) {
    case "add_node": {
      const node = payload.node as { id: string };
      return op("remove_node", { ...scoped, node: node.id });
    }
    case "remove_node": {
      const nodeId = payload.node as string;
      const node = (scope.nodes ?? []).find((candidate) => candidate.id === nodeId);
      if (!node) return null;
      const instance = node.call ? scope.instances?.[node.call] : undefined;
      const restore = op("add_node", {
        ...scoped,
        node: strip(node),
        ...(instance ? { instance: { id: node.call, ...strip(instance) } } : {}),
      });
      // 엣지도 같이 돌아와야 원래 그래프다 - 하나의 원자 op로 묶는다(§8.2.3).
      const edges = (scope.edges ?? []).filter(
        ([src, dst]) => endpointNode(src) === nodeId || endpointNode(dst) === nodeId);
      if (!edges.length) return restore;
      return op("batch", {}, [
        restore,
        ...edges.map(([src, dst]) => op("connect", { ...scoped, src, dst })),
      ]);
    }
    case "connect":
      return op("disconnect", { ...scoped, src: payload.src, dst: payload.dst });
    case "disconnect":
      return op("connect", { ...scoped, src: payload.src, dst: payload.dst });
    case "set_param": {
      const instance = scope.instances?.[payload.instance as string];
      if (!instance) return null;
      return op("set_param", {
        ...scoped, instance: payload.instance, path: payload.path,
        value: (instance.args ?? {})[payload.path as string] ?? null,
      });
    }
    case "set_ports": {
      const node = (scope.nodes ?? []).find((entry) => entry.id === payload.node);
      if (!node) return null;
      return op("set_ports", { ...scoped, node: payload.node, ports_out: node.ports_out ?? [] });
    }
    case "set_switch_active": {
      const instance = scope.instances?.[payload.instance as string];
      if (!instance) return null;
      return op("set_switch_active", {
        ...scoped, instance: payload.instance, active: instance.active,
      });
    }
    default:
      return null;
  }
}

function endpointNode(endpoint: string): string {
  return endpoint.split(".")[0];
}

/** 서버가 준 모델에서 기본값 키를 걷어낸다 - 왕복해도 같은 IR이어야 한다. */
function strip<T extends Node | Instance>(model: T): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(model).filter(([, value]) => value !== null && value !== undefined));
}
