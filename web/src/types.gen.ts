// 생성 파일 - 손으로 고치지 말 것.

// torchflow typegen (src/torchflow/typegen.py) 이 Pydantic 모델에서 뽑아낸다.


// IR (기획서 §17.1)

export interface Port {
  name: string;
  type?: string;
  shape?: (string | number)[];
  dtype?: string;
  [key: string]: unknown;
}

export interface HParam {
  type: "int" | "float" | "bool" | "str" | "enum";
  default?: unknown;
  choices?: unknown[];
  space?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Instance {
  label: string;
  type: string;
  args?: Record<string, unknown>;
  active?: unknown;
  variants?: Record<string, Record<string, unknown>>;
  body?: string;
  count?: unknown;
  mode?: "sequential" | "parallel" | "shared_weights";
  reduce?: unknown;
  bind?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Node {
  label: string;
  id: string;
  type?: string;
  call?: string;
  method?: string;
  args?: Record<string, unknown>;
  ports_out?: Port[];
  enabled?: unknown;
  [key: string]: unknown;
}

export interface Probe {
  label: string;
  id: string;
  type: string;
  on_edge?: [string, string];
  on?: string;
  auto_policy?: string;
  [key: string]: unknown;
}

export interface CodeCell {
  kind: "CellModule" | "CellFunction" | "CellStep" | "CellHook" | "CellData";
  file: string;
  source?: string;
  ports?: Record<string, Port[]>;
  shape_fn?: string;
  export_compatible?: boolean;
  children_ports?: string[];
  sha256?: string;
  [key: string]: unknown;
}

export interface Composite {
  doc?: string;
  params?: Record<string, HParam>;
  ports?: Record<string, Port[]>;
  instances?: Record<string, Instance>;
  nodes?: Node[];
  edges?: [string, string][];
  user_code?: Record<string, string>;
  [key: string]: unknown;
}

export interface Graph {
  name: string;
  instances?: Record<string, Instance>;
  nodes?: Node[];
  edges?: [string, string][];
  probes?: Probe[];
  init?: Record<string, unknown>;
  probe_objective?: string;
  user_code?: Record<string, string>;
  [key: string]: unknown;
}

export interface ModuleGraph {
  schema_version?: string;
  meta?: Record<string, unknown>;
  hparams?: Record<string, HParam>;
  composites?: Record<string, Composite>;
  graph: Graph;
  code_cells?: Record<string, CodeCell>;
  variant_sets?: Record<string, Record<string, unknown>>;
  experiment?: Record<string, unknown>;
  [key: string]: unknown;
}


// 프로토콜 (기획서 §8.2)

export interface NodeRequest {
  node_id: string;
  version: number;
  inputs?: string[];
  [key: string]: unknown;
}

export interface TensorSpec {
  shape?: (string | number)[];
  dtype?: string;
  device?: string;
  [key: string]: unknown;
}

export interface Op {
  type?: "Op";
  client_id: string;
  tmp_seq: number;
  kind: "add_node" | "remove_node" | "set_param" | "set_ports" | "rename" | "connect" | "disconnect" | "move" | "set_code" | "set_switch_active" | "promote_hp" | "save_variant" | "add_probe" | "remove_probe" | "batch";
  payload?: Record<string, unknown>;
  inverse_of?: number;
  ops?: Record<string, unknown>[];
  [key: string]: unknown;
}

export interface OpAck {
  type?: "OpAck";
  tmp_seq: number;
  seq: number;
  [key: string]: unknown;
}

export interface OpBroadcast {
  type?: "OpBroadcast";
  seq: number;
  op: Record<string, unknown>;
  [key: string]: unknown;
}

export interface NodeState {
  type?: "NodeState";
  seq?: number;
  node: string;
  path?: string;
  axis?: "L0" | "L1fwd" | "L1bwd" | "L2";
  state?: string;
  spec?: Record<string, unknown>;
  error?: Record<string, unknown>;
  badges?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface KernelStatus {
  type?: "KernelStatus";
  level?: "L0" | "L1" | "L2";
  alive?: boolean;
  responsive?: boolean;
  device?: string;
  busy?: Record<string, unknown>;
  mem?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Resync {
  type?: "Resync";
  from_seq?: number;
  to_seq?: number;
  ops?: Record<string, unknown>[];
  snapshot?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface Error {
  type?: "Error";
  req_id: string;
  kind: "shape" | "exception" | "oom" | "export" | "kernel" | "timeout" | "input" | "args";
  node_id?: string;
  message?: string;
  mapping?: Record<string, unknown>;
  traceback?: string;
  [key: string]: unknown;
}

export interface Log {
  type?: "Log";
  req_id?: string;
  level_name?: string;
  node_id?: string;
  text?: string;
  [key: string]: unknown;
}

export interface MemoryEstimate {
  type?: "MemoryEstimate";
  req_id: string;
  ok?: boolean;
  batch?: number;
  optimizer?: string;
  breakdown_bytes?: Record<string, number>;
  total_bytes?: number;
  band_gb?: number[];
  error?: Record<string, unknown>;
  [key: string]: unknown;
}


export type ToBrowser = OpAck | OpBroadcast | NodeState | KernelStatus | Resync | Error | Log;

