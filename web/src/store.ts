// 클라이언트 상태 (Zustand).
//
// 서버가 권위를 갖는다(§8.2.2): op는 낙관적으로 먼저 그리고 서버가 seq를 붙여
// 승인한다. 지금은 뷰어라 편집 op가 없으므로 NodeState 수신과 스코프·선택만
// 여기서 관리한다. 낙관 적용 큐는 M5에서 이 자리에 붙는다.

import { create } from "zustand";
import type { ModuleGraph, NodeState } from "./types.gen";

export interface Scope {
  name: string;        // "$graph" 또는 컴포지트 이름
  label: string;       // 브레드크럼에 보일 이름
  // 여기까지 내려오며 거친 호출 노드 id 체인. 같은 컴포지트가 여러 번
  // 인스턴스화되면(ResNet의 BasicBlock ×8) 이 경로만이 둘을 가른다.
  callPath: string;
}

interface State {
  token: string;
  connected: boolean;
  seq: number;
  graph: ModuleGraph | null;
  nodeStates: Record<string, NodeState>;
  scopes: Scope[];               // 브레드크럼 스택
  selected: string | null;
  focused: string | null;        // 키보드 포커스
  totals: { params: number; band: [number, number] | null; batch: number };
  kernelAlive: boolean;
  attached: boolean;
  probing: boolean;
  probeObjective: string;
  gradOverlay: boolean;
  /** 노드 키 → 좌표. layout.json이 정본이고 IR은 좌표를 모른다(§10.1). */
  positions: Record<string, { x: number; y: number }>;

  setGraph: (graph: ModuleGraph, seq: number) => void;
  applyNodeState: (state: NodeState) => void;
  applyNodeStates: (states: Record<string, NodeState>) => void;
  setConnected: (connected: boolean) => void;
  setKernel: (alive: boolean) => void;
  setAttached: (attached: boolean) => void;
  setProbing: (probing: boolean) => void;
  setProbeObjective: (objective: string) => void;
  toggleGradOverlay: () => void;
  setPosition: (key: string, position: { x: number; y: number }) => void;
  setPositions: (positions: Record<string, { x: number; y: number }>) => void;
  setTotals: (totals: Partial<State["totals"]>) => void;
  enterScope: (scope: Scope) => void;
  popToScope: (index: number) => void;
  select: (id: string | null) => void;
  focus: (id: string | null) => void;
}

/** 상태 키 = 호출 경로 + 노드 id. 컴포지트가 여러 번 인스턴스화돼도 섞이지 않는다. */
export function nodeStateKey(state: { node: string; path?: string }): string {
  return state.path ? `${state.path}/${state.node}` : state.node;
}

export const useStore = create<State>((set) => ({
  token: new URLSearchParams(location.search).get("token") ?? "",
  connected: false,
  seq: 0,
  graph: null,
  nodeStates: {},
  scopes: [{ name: "$graph", label: "Net", callPath: "" }],
  selected: null,
  focused: null,
  totals: { params: 0, band: null, batch: 64 },
  kernelAlive: false,
  attached: false,
  probing: false,
  probeObjective: "",
  gradOverlay: true,
  positions: {},

  setGraph: (graph, seq) => set({ graph, seq }),
  applyNodeState: (state) =>
    // L1은 L0가 채운 spec 위에 grad 배지를 얹는다 - 축이 다르므로 덮어쓰지 않고 합친다.
    set((prev) => {
      const key = nodeStateKey(state);
      const previous = prev.nodeStates[key];
      // L0(심볼 shape)와 L1(실측 B) 은 서로 다른 축이다(§5.3). L1이 도착해도
      // 캔버스에 붙는 shape는 L0의 것을 유지하고, 실측치는 배지로만 남긴다 -       // 아니면 [B, 3, 32, 32]가 probe 한 번에 [4, 3, 32, 32]로 굳어 버린다.
      const fromL1 = state.axis === "L1fwd" || state.axis === "L1bwd";
      const merged = previous
        ? { ...previous, ...state,
            axis: previous.axis ?? state.axis,
            spec: fromL1 ? (previous.spec ?? state.spec) : (state.spec ?? previous.spec),
            badges: {
              ...(previous.badges ?? {}),
              ...(state.badges ?? {}),
              ...(fromL1 && state.spec ? { measured: state.spec } : {}),
            } }
        : state;
      return { nodeStates: { ...prev.nodeStates, [key]: merged } };
    }),
  applyNodeStates: (states) =>
    set((prev) => ({ nodeStates: { ...prev.nodeStates, ...states } })),
  setConnected: (connected) => set({ connected }),
  setKernel: (kernelAlive) => set({ kernelAlive }),
  setAttached: (attached) => set({ attached }),
  setProbing: (probing) => set({ probing }),
  setProbeObjective: (probeObjective) => set({ probeObjective }),
  toggleGradOverlay: () => set((prev) => ({ gradOverlay: !prev.gradOverlay })),
  setPosition: (key, position) =>
    set((prev) => ({ positions: { ...prev.positions, [key]: position } })),
  setPositions: (positions) => set((prev) => ({ positions: { ...prev.positions, ...positions } })),
  setTotals: (totals) => set((prev) => ({ totals: { ...prev.totals, ...totals } })),
  enterScope: (scope) =>
    set((prev) =>
      prev.scopes.some((existing) => existing.callPath === scope.callPath)
        ? prev
        : { scopes: [...prev.scopes, scope], selected: null, focused: null }),
  popToScope: (index) =>
    set((prev) => ({ scopes: prev.scopes.slice(0, index + 1), selected: null, focused: null })),
  select: (selected) => set({ selected }),
  focus: (focused) => set({ focused }),
}));

/** 현재 스코프의 nodes/edges/instances. 컴포지트에 들어가면 그 안을 본다. */
export function currentScope(state: Pick<State, "graph" | "scopes">) {
  const graph = state.graph;
  if (!graph) return null;
  const current = state.scopes[state.scopes.length - 1];
  if (current.name === "$graph") return graph.graph;
  return graph.composites?.[current.name] ?? graph.graph;
}
