// hub와의 통신. WebSocket 하나 + REST 몇 개(§8.2.2).

import type { ModuleGraph, NodeState, ToBrowser } from "./types.gen";
import { clientId } from "./graph/ops";
import type { Block, Op } from "./graph/ops";
import { useStore } from "./store";

function authHeaders(): HeadersInit {
  return { Authorization: `token ${useStore.getState().token}` };
}

export async function fetchGraph(): Promise<{ seq: number; graph: ModuleGraph }> {
  const response = await fetch("/api/graph", { headers: authHeaders() });
  if (!response.ok) throw new Error(`graph: ${response.status}`);
  return response.json();
}

export async function fetchHealth(): Promise<{
  kernel: { alive: boolean }; attached?: boolean; graph_open?: boolean;
}> {
  const response = await fetch("/api/health", { headers: authHeaders() });
  return response.json();
}

export async function runShapes(): Promise<{
  node_states: NodeState[]; total_params: number; rerun_ratio: number;
}> {
  const response = await fetch("/api/shapes", { method: "POST", headers: authHeaders() });
  if (!response.ok) throw new Error(`shapes: ${response.status}`);
  return response.json();
}

export interface Template {
  id: string; name: string; recipe: string; metric: string;
  path: string; available: boolean;
}

export interface Candidate {
  name: string; bases: string[]; suggestion: string; doc: string;
  params: { name: string; annotation: string | null; default: unknown }[];
}

export interface RecentGraph {
  path: string; name: string; file: string; modified: number; where: string;
}

export interface StartInfo {
  graph_open: boolean;
  state_dir: string;
  torch_version: string | null;
  devices: { name: string; label: string }[];
  templates: Template[];
  recent: RecentGraph[];
}

export async function fetchStart(): Promise<StartInfo> {
  const response = await fetch("/api/start", { headers: authHeaders() });
  if (!response.ok) throw new Error(`start: ${response.status}`);
  return response.json();
}

export async function fetchRegistry(): Promise<Block[]> {
  const response = await fetch("/api/registry", { headers: authHeaders() });
  if (!response.ok) return [];
  return (await response.json()).blocks ?? [];
}

/** 빈 그래프에서 시작한다. 첫 화면의 진입점 하나(§2.2). */
/** ``dataset``을 주면 Input/Output이 그 데이터 규격으로 깔린 채 열린다. */
export async function newGraph(name = "untitled", dataset?: string, recipe?: Recipe | null):
    Promise<{ ok?: boolean; error?: string }> {
  const response = await fetch("/api/new", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(dataset ? { name, dataset, recipe: recipe ?? {} } : { name }),
  });
  return response.json();
}

/** 그래프를 닫고 첫 화면으로. 커널은 살아 있다. */
export async function closeGraph(): Promise<{ ok?: boolean }> {
  const response = await fetch("/api/close", { method: "POST", headers: authHeaders() });
  return response.json();
}

export async function saveGraph(path?: string):
    Promise<{ ok?: boolean; path?: string; problems?: string[]; error?: string }> {
  const response = await fetch("/api/save", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(path ? { path } : {}),
  });
  return response.json();
}

/**
 * 편집 op 하나를 권위 그래프에 보낸다.
 *
 * 서버가 seq를 붙여 승인하고 같은 응답에 L0 결과를 실어 준다. `ponytail: 낙관
 * 적용 대신 승인 후 재조회. 편집 규칙을 클라이언트에도 복제하는 것보다 왕복
 * 한 번이 싸다 - 로컬 hub에서 한 자릿수 ms다.`
 */
export async function sendOp(operation: Op):
    Promise<{ seq: number; node_states: NodeState[]; error?: string }> {
  const response = await fetch("/api/ops", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(operation),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? `op: ${response.status}`);
  return body;
}

export async function openGraph(path: string): Promise<{ ok?: boolean; error?: string }> {
  const response = await fetch("/api/open", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  return response.json();
}

/** 드롭된 소스를 hub가 ast로 읽는다. 실행하지 않는다. */
export async function inspectSource(source: string):
    Promise<{ candidates?: Candidate[]; error?: string }> {
  const response = await fetch("/api/inspect", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ source }),
  });
  return response.json();
}

/** 커널이 모델을 만들고 한 번 돌려 그래프로 편다. */
export async function importSource(request: {
  filename: string; source: string; factory: string; example: string;
}): Promise<{ ok?: boolean; error?: string; stage?: string }> {
  const response = await fetch("/api/import", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return response.json();
}

export interface RunInfo {
  id: string; kind: string; name: string | null; status: string;
  created: number; updated: number; keys: string[]; manifest: Record<string, unknown>;
}

export interface CurveSeries { run: string; points: [number, number][] }

export async function fetchRuns(): Promise<RunInfo[]> {
  const response = await fetch("/api/runs", { headers: authHeaders() });
  if (!response.ok) return [];
  return (await response.json()).runs ?? [];
}

export async function fetchCurve(key: string, runs: string[]): Promise<CurveSeries[]> {
  const query = new URLSearchParams({ key, runs: runs.join(",") });
  const response = await fetch(`/api/runs/curve?${query}`, { headers: authHeaders() });
  if (!response.ok) return [];
  return (await response.json()).series ?? [];
}

export interface LogLine { text: string; stream: string; node_id: string | null }

/** 노드를 주면 그 노드가 찍은 것만 (Inspector 출력 섹션). */
export async function fetchLogs(node = ""): Promise<LogLine[]> {
  const query = node ? `?node=${encodeURIComponent(node)}` : "";
  const response = await fetch(`/api/logs${query}`, { headers: authHeaders() });
  if (!response.ok) return [];
  return (await response.json()).logs ?? [];
}

export interface EvalResult {
  ok: boolean; text?: string; error?: string;
  spec?: { shape: (string | number)[]; dtype: string; device: string };
}

/** Debug Console - 선택 노드의 마지막 probe 값을 표현식으로 조회한다 (§5.6.1). */
export async function evalExpression(expr: string, node: string): Promise<EvalResult> {
  const response = await fetch("/api/eval", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ expr, node }),
  });
  return response.json();
}

export async function fetchCode():
    Promise<{ code?: string; lines?: number; ir_sha256?: string; error?: string }> {
  const response = await fetch("/api/code", { headers: authHeaders() });
  return response.json();
}

export async function fetchLayout(): Promise<Record<string, { x: number; y: number }>> {
  const response = await fetch("/api/layout", { headers: authHeaders() });
  if (!response.ok) return {};
  return (await response.json()).positions ?? {};
}

/** 노드 하나의 좌표를 저장한다. 그래프 의미가 아니므로 재실행을 유발하지 않는다. */
export async function saveLayout(key: string, position: { x: number; y: number }) {
  await fetch("/api/layout", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ positions: { [key]: position } }),
  });
}

export interface TrainRun {
  run_id: string; state: string; step: number; total: number;
  device: string; alive: boolean; error: { message?: string } | null;
  reason?: string; nan_step?: number;
  // reported run은 hparam이 동결이라 학습 중 편집이 fork가 된다(§5.7.2).
  kind?: string; smoke?: boolean; parent_run?: string; forked_from?: string;
  // 스케줄러가 있으면 lr은 base이고 실제 lr은 base x schedule(§5.7.1).
  scheduler?: string | null; lr?: number; base_lr?: number;
  restart_required?: string; message?: string; ok?: boolean;
}

export interface DatasetInfo {
  name: string; label: string; shape: number[]; classes: number; available: boolean;
  /** builtin은 내려받는 것, user는 data/ 아래 폴더(이미지 폴더·CSV·npy). */
  source: "builtin" | "user"; kind: string; size_mb?: number; count?: number;
  class_names?: string[];
}

/** 사람이 정제한 설정. 키는 datasets.recipe_defaults가 정한다. */
export type Recipe = Record<string, unknown>;

export interface DatasetColumn { name: string; kind: string; missing: number; uniques?: string[] | null }
export interface EffectiveSpec extends DatasetInfo {
  recipe: Recipe; split?: { train: number; val: number };
  class_counts?: Record<string, number>; problem?: string;
}
export interface DatasetPreview {
  base: DatasetInfo & { columns?: DatasetColumn[]; label_column?: string };
  spec: EffectiveSpec;
  preview: {
    columns?: DatasetColumn[]; header?: string[]; rows?: string[][];
    thumbnails?: { png: string; labels: string[]; tile: number; per_class: number } | null;
  };
  error?: string;
}

export async function previewDataset(name: string, recipe: Recipe | null): Promise<DatasetPreview> {
  const response = await fetch(`/api/datasets/${encodeURIComponent(name)}/preview`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ recipe: recipe ?? {} }),
  });
  return response.json();
}

/** 그래프에 데이터와 레시피를 붙이고 Input 규격을 맞춘다. */
export async function setGraphData(name: string, recipe: Recipe | null):
    Promise<{ ok?: boolean; error?: string; spec?: EffectiveSpec }> {
  const response = await fetch("/api/data", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ name, recipe: recipe ?? {} }),
  });
  return response.json();
}

/** 끌어다 놓은 파일 하나를 data/<name>/<path>로 올린다. 본문이 곧 파일이다. */
export async function uploadDatasetFile(name: string, path: string, file: File):
    Promise<{ ok?: boolean; error?: string }> {
  const response = await fetch(
    `/api/datasets/${encodeURIComponent(name)}/files?path=${encodeURIComponent(path)}`,
    { method: "PUT", headers: authHeaders(), body: file });
  return response.json();
}

/** 다른 곳의 폴더를 data/ 에 링크로 등록한다. */
export async function addDatasetFolder(path: string): Promise<DatasetInfo & { error?: string }> {
  const response = await fetch("/api/datasets", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  return response.json();
}

export async function fetchDatasets(): Promise<DatasetInfo[]> {
  const response = await fetch("/api/datasets", { headers: authHeaders() });
  if (!response.ok) return [];
  return (await response.json()).datasets ?? [];
}

/** 다운로드는 사람이 누른 버튼에서만 시작한다(§3.1). */
export async function downloadDataset(name: string): Promise<{ ok?: boolean; error?: string }> {
  const response = await fetch(`/api/datasets/${encodeURIComponent(name)}/download`, {
    method: "POST", headers: authHeaders(),
  });
  return response.json();
}

export interface TrainOptions {
  dataset?: string; recipe?: Recipe;
  steps?: number; batch?: number; lr?: number; optimizer?: string; smoke?: boolean;
  scheduler?: string; warmup_steps?: number;
}

/** 학습을 시작한다. 워커는 hub와 분리된 세션에서 돈다(§5.5.3). */
export interface PortSpec { name: string; type: string; shape: (string | number)[]; dtype: string }

export async function startTraining(options: TrainOptions):
    Promise<TrainRun & { error?: string; fix?: { node: string; ports_out: PortSpec[] } }> {
  const response = await fetch("/api/train", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(options),
  });
  return response.json();
}

export async function controlTraining(
  runId: string, cmd: string, extra: Record<string, unknown> = {},
): Promise<TrainRun> {
  const response = await fetch(`/api/train/${runId}`, {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ cmd, ...extra }),
  });
  return response.json();
}

/** 워커 프로세스의 표준출력. 커서를 돌려주므로 늘어난 만큼만 이어 붙인다. */
export async function fetchTrainStdout(runId: string, offset: number):
    Promise<{ text: string; offset: number }> {
  const response = await fetch(`/api/train/${runId}/stdout?offset=${offset}`,
                               { headers: authHeaders() });
  if (!response.ok) return { text: "", offset };
  return response.json();
}

export async function fetchTraining(): Promise<TrainRun[]> {
  const response = await fetch("/api/train", { headers: authHeaders() });
  if (!response.ok) return [];
  return (await response.json()).runs ?? [];
}

export interface DetailPanel {
  type: "grid" | "heatmap" | "hist" | "bars" | "text"; title: string;
  png?: string; cols?: number; rows?: number; note?: string;
  range?: [number, number]; before?: number[] | null; after?: number[];
  labels?: string[]; values?: number[];
}
export interface BlockDetailInfo {
  ok: boolean; error?: string; node: string; label: string; kind: string; params: number;
  input_shape?: number[] | null; output_shape?: number[] | null; explain: string; panels: DetailPanel[];
}

/** 블록 상세(§6.3). L1 커널의 마지막 probe 값으로 그린다. */
export async function fetchDetail(node: string): Promise<BlockDetailInfo> {
  const response = await fetch("/api/detail", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ node }),
  });
  return response.json();
}

export async function runProbe(batch = 4) {
  const response = await fetch("/api/probe", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ batch, objective: "random_target_ce", collect: ["grad_norm", "grad_ratio", "hist"] }),
  });
  return response.json();
}

export async function estimateMemory(batch: number) {
  const response = await fetch(`/api/memory?batch=${batch}`, {
    method: "POST", headers: authHeaders(),
  });
  return response.json();
}

async function refreshGraph(): Promise<void> {
  const { graph, seq } = await fetchGraph();
  useStore.getState().setGraph(graph, seq);
}

/** WebSocket 연결. 끊기면 다시 붙고 `Resync`로 놓친 op를 메운다. */
export function connect(): () => void {
  const { token } = useStore.getState();
  let socket: WebSocket | null = null;
  let retry: number | undefined;
  let closed = false;

  const open = () => {
    if (closed) return;
    socket = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?token=${token}`);

    socket.onopen = () => useStore.getState().setConnected(true);
    socket.onclose = () => {
      useStore.getState().setConnected(false);
      if (!closed) retry = window.setTimeout(open, 1000);
    };
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data) as ToBrowser;
      const store = useStore.getState();
      switch (message.type) {
        case "Resync": {
          const states = (message.snapshot?.node_states ?? {}) as Record<string, NodeState>;
          store.applyNodeStates(states);
          break;
        }
        case "NodeState":
          store.applyNodeState(message);
          if (message.badges?.probe_objective) {
            store.setProbeObjective(String(message.badges.probe_objective));
          }
          break;
        case "OpBroadcast": {
          // 다른 클라이언트의 편집. 내 op는 이미 응답으로 반영했으므로 건너뛴다.
          const author = (message.op as { client_id?: string } | undefined)?.client_id;
          if (author !== clientId()) void refreshGraph();
          break;
        }
        case "KernelStatus":
          store.setKernel(message.alive ?? false);
          break;
      }
    };
  };

  open();
  return () => {
    closed = true;
    window.clearTimeout(retry);
    socket?.close();
  };
}
