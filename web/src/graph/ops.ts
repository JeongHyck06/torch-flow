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
  source: "builtin" | "reflection" | "composite" | string;
  /** 묶음 블록이면 그래프에 없을 때 먼저 정의해 넣을 몸체. */
  composite?: unknown;
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
  } else if (block.source === "composite") {
    // 템플릿의 block1처럼: 인스턴스는 composite:이름, 노드는 그 호출이고 출력 포트 이름을 든다.
    payload.instance = { id: newId(), label, type: block.type, args: defaultArgs(block) };
    payload.node = { id: nodeId, label, method: "forward",
                     ports_out: block.ports.out.map((name) => ({ name, type: "Tensor" })) };
  } else {
    // 내장 블록의 기본값도 노드에 적는다 - Train의 lr·steps가 보이는 값이어야 고칠 수 있다.
    const args = defaultArgs(block);
    payload.node = { id: nodeId, label, type: block.type, ...inputPorts(block),
                     ...(Object.keys(args).length ? { args } : {}) };
  }
  return { op: op("add_node", payload), nodeId };
}

/** 기본값이 있는 파라미터만 채운다. 필수인데 빈 것은 L0가 노드에 귀속해 알려준다(§5.6). */
function defaultArgs(block: Block): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(block.params)
      .filter(([, schema]) => schema.default !== undefined && schema.default !== null)
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
 * 고른 블록을 다른 블록으로 갈아 끼운다(팔레트 Replace, §4.2). 배선은 그대로 남는다.
 *
 * 포트 이름은 되도록 쓰던 것을 그대로 쓰고, 새 블록에 그런 이름이 없으면 첫 입력으로 보낸다 -
 * Conv2d(`input`)를 묶음 블록(`x`)으로 바꿔도 앞뒤가 끊기지 않는다.
 */
export function replaceOp(
  scope: Graph | Composite, nodeId: string, block: Block, composite: string | null,
): { op: Op; nodeId: string } | { error: string } {
  const old = (scope.nodes ?? []).find((node) => node.id === nodeId);
  if (!old) return { error: "바꿀 블록이 없습니다" };
  if (NOT_GROUPABLE.has((old.type ?? "").split("@")[0])) {
    return { error: `${old.label} 블록은 바꿀 수 없습니다` };
  }
  const edges = (scope.edges ?? []) as [string, string][];
  const incoming = edges.filter(([, dst]) => endpointNode(dst) === nodeId);
  const outgoing = edges.filter(([src]) => endpointNode(src) === nodeId);
  if (incoming.length && !block.ports.in.length) {
    return { error: `${block.label}에는 입력이 없어 앞 블록과 이을 수 없습니다` };
  }
  if (outgoing.length && !block.ports.out.length) {
    return { error: `${block.label}에는 출력이 없어 뒤 블록과 이을 수 없습니다` };
  }

  const { op: add, nodeId: fresh } = addBlockOp(scope, block, composite);
  const scoped = composite ? { composite } : {};
  const outPort = block.ports.out[0] ?? "output";
  const inner: Op[] = [
    add,
    ...incoming.map(([src, dst]) => {
      const port = dst.split(".")[1];
      const target = block.ports.in.includes(port) ? port : block.ports.in[0];
      return op("connect", { ...scoped, src, dst: `${fresh}.${target}` });
    }),
    ...outgoing.map(([, dst]) => op("connect", { ...scoped, src: `${fresh}.${outPort}`, dst })),
    op("remove_node", { ...scoped, node: nodeId }),
  ];
  return { op: op("batch", {}, inner), nodeId: fresh };
}

/** 그래프 경계와 학습 설정. 안으로 넣으면 그래프가 입력도 출력도 잃는다. */
const NOT_GROUPABLE = new Set(["torchflow.Input", "torchflow.Output", "torchflow.Train"]);

/**
 * 고른 블록들을 묶음 블록 하나로 승격한다(`Cmd+G`, 기획서 §4.4.3).
 *
 * 안쪽으로 들어가는 선은 `$in.x`가 되고 밖으로 나가는 선은 `$out.output`이 된다 -
 * 컴포지트의 입출력 포트는 그렇게 **원래 있던 선에서** 정해진다. 사람이 포트를
 * 설계하지 않아도 되고, 묶기 전후로 바깥에서 본 배선이 같다.
 *
 * 한 덩어리 batch로 보낸다. 실행 취소가 통째로 한 번에 되돌려야 하기 때문이다.
 */
export function groupOp(
  scope: Graph | Composite, ids: string[], composite: string | null, definedNames: string[],
): { op: Op; nodeId: string; name: string } | { error: string } {
  const inside = new Set(ids);
  const chosen = (scope.nodes ?? []).filter((node) => inside.has(node.id));
  if (chosen.length < 2) {
    return { error: "블록 두 개 이상을 고르세요 - Cmd 를 누른 채 클릭하거나 Shift 로 끕니다" };
  }
  const blocked = chosen.find((node) => NOT_GROUPABLE.has((node.type ?? "").split("@")[0]));
  if (blocked) return { error: `${blocked.label} 블록은 묶음 안에 넣을 수 없습니다` };

  const internal: [string, string][] = [];
  const inbound: [string, string][] = [];
  const outbound: [string, string][] = [];
  for (const edge of (scope.edges ?? []) as [string, string][]) {
    const from = inside.has(endpointNode(edge[0]));
    const to = inside.has(endpointNode(edge[1]));
    if (from && to) internal.push(edge);
    else if (to) inbound.push(edge);
    else if (from) outbound.push(edge);
  }

  // 같은 곳에서 오는 선은 포트 하나를 나눠 쓴다 - 스킵 연결이 입력을 둘로 늘리지 않는다.
  const inPort = new Map<string, string>();
  for (const [src] of inbound) if (!inPort.has(src)) inPort.set(src, portName("x", inPort.size));
  const outPort = new Map<string, string>();
  for (const [src] of outbound) {
    if (!outPort.has(src)) outPort.set(src, portName("output", outPort.size));
  }

  const body = {
    ports: {
      in: [...inPort.values()].map((name) => ({ name, type: "Tensor" })),
      out: [...outPort.values()].map((name) => ({ name, type: "Tensor" })),
    },
    instances: Object.fromEntries(
      chosen.filter((node) => node.call)
        .map((node) => [node.call as string, strip(scope.instances![node.call as string])])),
    nodes: chosen.map(strip),
    edges: [
      ...internal,
      ...inbound.map(([src, dst]) => [`$in.${inPort.get(src)}`, dst]),
      ...outbound.map(([src]) => [src, `$out.${outPort.get(src)}`]),
    ],
  };

  const name = uniqueName(definedNames);
  const nodeId = newId();
  const label = uniqueLabel(scope, name);
  const scoped = composite ? { composite } : {};
  const inner: Op[] = [
    op("define_composite", { name, body }),
    ...chosen.map((node) => op("remove_node", { ...scoped, node: node.id })),
    op("add_node", {
      ...scoped,
      instance: { id: newId(), label, type: `composite:${name}`, args: {} },
      node: { id: nodeId, label, method: "forward",
              ports_out: [...outPort.values()].map((port) => ({ name: port, type: "Tensor" })) },
    }),
    // 바깥 배선을 새 노드에 다시 잇는다. 안쪽으로 가던 선은 remove_node가 이미 지웠다.
    ...inbound.map(([src]) => op("connect", { ...scoped, src, dst: `${nodeId}.${inPort.get(src)}` })),
    ...outbound.map(([src, dst]) =>
      op("connect", { ...scoped, src: `${nodeId}.${outPort.get(src)}`, dst })),
  ];
  return { op: op("batch", {}, inner), nodeId, name };
}

function portName(base: string, index: number): string {
  return index === 0 ? base : `${base}${index + 1}`;
}

function uniqueName(taken: string[]): string {
  const names = new Set(taken);
  if (!names.has("Group")) return "Group";
  for (let index = 2; ; index += 1) if (!names.has(`Group${index}`)) return `Group${index}`;
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
    case "rename":
      // 스코프가 아니라 그래프 전체 이름이다. 이전 이름은 호출한 쪽이 실어 준다.
      return payload.previous === undefined
        ? null : op("rename", { name: payload.previous });
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
    // 인스턴스를 통째로 바꾸는 편집(Switch 만들기, 변형 더하기·빼기)의 역은
    // **이전 인스턴스를 실은 같은 op**다.
    case "set_instance": {
      const instance = scope.instances?.[payload.instance as string];
      if (!instance) return null;
      return op("set_instance", { ...scoped, instance: payload.instance, body: strip(instance) });
    }
    case "promote_hp": {
      const target = payload.instance
        ? scope.instances?.[payload.instance as string]
        : (scope.nodes ?? []).find((node) => node.id === payload.node);
      if (!target) return null;
      return op("demote_hp", {
        ...scoped, instance: payload.instance, node: payload.node, path: payload.path,
        name: payload.name, value: (target.args ?? {})[payload.path as string],
      });
    }
    case "demote_hp":
      return op("promote_hp", {
        ...scoped, instance: payload.instance, node: payload.node,
        path: payload.path, name: payload.name, value: payload.value,
      });
    case "save_variant":
      return op("remove_variant", { name: payload.name });
    case "remove_variant":
      return payload.values === undefined
        ? null : op("save_variant", { name: payload.name, values: payload.values });
    // 정의는 팔레트가 그래프에 없을 때만 보내므로 역은 지우기다. 부르는 노드가 남아 있으면
    // 서버가 거부한다 - 실행 취소는 노드부터 되돌아가므로 순서가 맞는다.
    // 몸체를 같이 실어야 이 역 op의 역(다시 실행)이 정의를 되살릴 수 있다.
    case "define_composite":
      return op("remove_composite", { name: payload.name, body: payload.body });
    case "remove_composite":
      return payload.body === undefined
        ? null : op("define_composite", { name: payload.name, body: payload.body });
    // 묶음 하나를 통째로 되돌린다. 안쪽 역 op는 전부 **묶기 직전** 그래프에서 계산하고
    // 순서만 뒤집는다 - 마지막에 넣은 것부터 빼야 서버의 참조 검사(컴포지트를 부르는
    // 노드가 남아 있으면 지울 수 없다)를 통과한다.
    case "batch": {
      const inner = (current.ops ?? []).map((one) => inverseOf(scope, one));
      if (inner.some((one) => one === null)) return null;
      const flat = (inner as Op[]).reverse().flatMap(flatten);
      return flat.length === 1 ? flat[0] : op("batch", {}, flat);
    }
    default:
      return null;
  }
}

function endpointNode(endpoint: string): string {
  return endpoint.split(".")[0];
}

/** 중첩 batch를 편다. 서버의 batch는 한 겹만 풀어 적용하므로 겹쳐 보내면 조용히 사라진다. */
export function flatten(one: Op): Op[] {
  return one.kind === "batch" ? (one.ops ?? []).flatMap(flatten) : [one];
}

/** 서버가 준 모델에서 기본값 키를 걷어낸다 - 왕복해도 같은 IR이어야 한다. */
function strip<T extends Node | Instance>(model: T): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(model).filter(([, value]) => value !== null && value !== undefined));
}
