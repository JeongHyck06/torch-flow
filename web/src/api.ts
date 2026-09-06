// hub와의 통신. WebSocket 하나 + REST 몇 개(§8.2.2).

import type { ModuleGraph, NodeState, ToBrowser } from "./types.gen";
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

export interface StartInfo {
  graph_open: boolean;
  state_dir: string;
  torch_version: string | null;
  devices: { name: string; note: string | null }[];
  templates: Template[];
}

export async function fetchStart(): Promise<StartInfo> {
  const response = await fetch("/api/start", { headers: authHeaders() });
  if (!response.ok) throw new Error(`start: ${response.status}`);
  return response.json();
}

export async function openGraph(path: string): Promise<{ ok?: boolean; error?: string }> {
  const response = await fetch("/api/open", {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
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
