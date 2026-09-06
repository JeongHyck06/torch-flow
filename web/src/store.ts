// 클라이언트 상태 (Zustand).
//
// 서버가 권위를 갖는다(§8.2.2): 편집 op를 보내고 서버가 seq를 붙여 승인한 그래프를
// 다시 읽는다. 실행 취소 스택은 클라이언트별이며 **역 op**를 담는다(§8.2.3) -
// 다른 클라이언트의 op는 여기 들어오지 않는다.

import { create } from "zustand";
import type { ModuleGraph, NodeState } from "./types.gen";
import type { Recipe } from "./api";
import type { Op } from "./graph/ops";

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
  /** 노드 키 -> 좌표. layout.json이 정본이고 IR은 좌표를 모른다(§10.1). */
  positions: Record<string, { x: number; y: number }>;
  runPanel: boolean;
  /** 상단 탭. Experiment와 Runs는 M7 전까지 비어 있다. */
  tab: "model" | "code";
  /** 역 op 스택. 스택에 든 op를 그대로 보내면 되돌아간다(§8.2.3). */
  undoStack: Op[];
  redoStack: Op[];
  dirty: boolean;                // 마지막 저장 이후 편집이 있었나
  /** 팔레트를 띄운 캔버스 좌표. null이면 닫혀 있다(§4.2). */
  paletteAt: { x: number; y: number } | null;
  /** 이 그래프가 배우는 데이터와 정제 설정. 그래프의 experiment.data가 정본이다. */
  dataset: string;
  recipe: Recipe | null;

  setGraph: (graph: ModuleGraph, seq: number) => void;
  openGraph: (graph: ModuleGraph, seq: number) => void;
  closeGraph: () => void;
  applyNodeState: (state: NodeState) => void;
  applyNodeStates: (states: Record<string, NodeState>) => void;
  setConnected: (connected: boolean) => void;
  setKernel: (alive: boolean) => void;
  setAttached: (attached: boolean) => void;
  setProbing: (probing: boolean) => void;
  setProbeObjective: (objective: string) => void;
  toggleGradOverlay: () => void;
  toggleRunPanel: () => void;
  setTab: (tab: State["tab"]) => void;
  setPosition: (key: string, position: { x: number; y: number }) => void;
  setPositions: (positions: Record<string, { x: number; y: number }>) => void;
  setTotals: (totals: Partial<State["totals"]>) => void;
  pushUndo: (inverse: Op, clearRedo?: boolean) => void;
  takeUndo: () => Op | undefined;
  pushRedo: (inverse: Op) => void;
  takeRedo: () => Op | undefined;
  setDirty: (dirty: boolean) => void;
  openPalette: (at: { x: number; y: number }) => void;
  setData: (dataset: string, recipe: Recipe | null) => void;
  closePalette: () => void;
  enterScope: (scope: Scope) => void;
  popToScope: (index: number) => void;
  select: (id: string | null) => void;
  focus: (id: string | null) => void;
}

/** 상태 키 = 호출 경로 + 노드 id. 컴포지트가 여러 번 인스턴스화돼도 섞이지 않는다. */
export function nodeStateKey(state: { node: string; path?: string }): string {
  return state.path ? `${state.path}/${state.node}` : state.node;
}

export const useStore = create<State>((set, get) => ({
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
  runPanel: false,
  tab: "model",
  undoStack: [],
  redoStack: [],
  dirty: false,
  paletteAt: null,
  dataset: "teacher",
  recipe: null,

  openGraph: (graph, seq) =>
    // **다른** 그래프를 연다. 이전 그래프에 딸린 것은 전부 비운다 - 노드 상태,
    // 브레드크럼, 선택, 편집 히스토리, 총계. setGraph는 편집 왕복마다 불리므로
    // 거기서 비우면 매 편집마다 배지가 사라진다. 그래서 둘을 나눈다.
    set({ graph, seq, nodeStates: {}, selected: null, focused: null,
          scopes: [{ name: "$graph", label: graph.graph.name || "Net", callPath: "" }],
          undoStack: [], redoStack: [], dirty: false, paletteAt: null,
          totals: { params: 0, band: null, batch: 64 },
          ...dataOf(graph) }),
  setGraph: (graph, seq) =>
    // 브레드크럼의 뿌리는 그래프 이름이다.
    set((prev) => ({
      graph, seq,
      scopes: [{ name: "$graph", label: graph.graph.name || "Net", callPath: "" },
               ...prev.scopes.slice(1)],
    })),
  // 첫 화면으로 돌아간다. 편집 히스토리와 노드 상태는 그래프에 딸린 것이라 함께 비운다.
  closeGraph: () =>
    set({ graph: null, seq: 0, nodeStates: {}, selected: null, focused: null,
          scopes: [{ name: "$graph", label: "Net", callPath: "" }],
          undoStack: [], redoStack: [], dirty: false, paletteAt: null,
          totals: { params: 0, band: null, batch: 64 } }),
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
  toggleRunPanel: () => set((prev) => ({ runPanel: !prev.runPanel })),
  setTab: (tab) => set({ tab }),
  setPosition: (key, position) =>
    set((prev) => ({ positions: { ...prev.positions, [key]: position } })),
  setPositions: (positions) => set((prev) => ({ positions: { ...prev.positions, ...positions } })),
  setTotals: (totals) => set((prev) => ({ totals: { ...prev.totals, ...totals } })),
  // 새 편집은 redo를 무효로 만든다 - 다른 갈래로 갔으므로 앞으로 갈 곳이 없다.
  // 다시 실행(redo)이 밀어 넣을 때는 남은 redo 스택을 지우지 않는다.
  pushUndo: (inverse, clearRedo = true) =>
    set((prev) => ({ undoStack: [...prev.undoStack, inverse],
                     redoStack: clearRedo ? [] : prev.redoStack, dirty: true })),
  takeUndo: () => {
    const stack = get().undoStack;
    if (!stack.length) return undefined;
    set({ undoStack: stack.slice(0, -1) });
    return stack[stack.length - 1];
  },
  pushRedo: (inverse) => set((prev) => ({ redoStack: [...prev.redoStack, inverse], dirty: true })),
  takeRedo: () => {
    const stack = get().redoStack;
    if (!stack.length) return undefined;
    set({ redoStack: stack.slice(0, -1) });
    return stack[stack.length - 1];
  },
  setDirty: (dirty) => set({ dirty }),
  openPalette: (paletteAt) => set({ paletteAt }),
  setData: (dataset, recipe) => set({ dataset, recipe }),
  closePalette: () => set({ paletteAt: null }),
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

/** 그래프에 붙은 데이터. 없으면 합성 과제다. */
function dataOf(graph: ModuleGraph): { dataset: string; recipe: Recipe | null } {
  const data = (graph.experiment as { data?: { name?: string; recipe?: Recipe } } | undefined)?.data;
  return { dataset: data?.name ?? "teacher", recipe: data?.recipe ?? null };
}

/** 현재 스코프의 nodes/edges/instances. 컴포지트에 들어가면 그 안을 본다. */
export function currentScope(state: Pick<State, "graph" | "scopes">) {
  const graph = state.graph;
  if (!graph) return null;
  const current = state.scopes[state.scopes.length - 1];
  if (current.name === "$graph") return graph.graph;
  return graph.composites?.[current.name] ?? graph.graph;
}
