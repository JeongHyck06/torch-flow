// "지금 어느 단계인가"를 한 곳에서 센다. 단계 표시줄(Stages)과 Train 노드 배지가 같이 쓴다.

import type { TrainRun } from "./api";
import { SYNTHETIC } from "./store";
import { formatCount } from "./theme";
import type { ModuleGraph, Node, NodeState } from "./types.gen";

export type StageState = "done" | "active" | "pending" | "error";

export interface Stage {
  key: "data" | "model" | "train" | "run";
  name: string;
  state: StageState;
  detail: string;
  hint: string;
}

export const RUNNING = new Set(["running", "starting"]);
const STRUCTURAL = new Set(["torchflow.Input", "torchflow.Output", "torchflow.Train"]);

export function kindOf(node: Node): string {
  return (node.type ?? "").split("@")[0];
}

/** 마지막 run을 한 줄로. 없으면 빈 문자열. */
export function runLabel(runs: TrainRun[]): string {
  const run = runs[runs.length - 1];
  if (!run) return "";
  const step = `step ${run.step.toLocaleString()} / ${run.total.toLocaleString()}`;
  if (RUNNING.has(run.state)) return step;
  if (run.state === "paused") return `일시정지 · ${step}`;
  if (run.state === "done") return `done · step ${run.step.toLocaleString()}`;
  return `${run.state} · step ${run.step.toLocaleString()}`;
}

export function computeStages(input: {
  graph: ModuleGraph | null;
  nodeStates: Record<string, NodeState>;
  dataset: string;
  runs: TrainRun[];
  params: number;
}): Stage[] {
  const nodes = input.graph?.graph.nodes ?? [];
  const hasInput = nodes.some((node) => kindOf(node) === "torchflow.Input");
  const hasOutput = nodes.some((node) => kindOf(node) === "torchflow.Output");
  const train = nodes.find((node) => kindOf(node) === "torchflow.Train");
  const errors = nodes.filter((node) => input.nodeStates[node.id]?.state === "error").length;
  const modules = nodes.filter((node) => !STRUCTURAL.has(kindOf(node))).length;
  const run = input.runs[input.runs.length - 1];

  const data: Stage = !SYNTHETIC.has(input.dataset)
    ? { key: "data", name: "데이터", state: "done", detail: input.dataset,
        hint: "Input을 더블클릭하면 바꿀 수 있습니다" }
    : hasInput
      ? { key: "data", name: "데이터", state: "pending", detail: "Input에 불러오기",
          hint: "Input 블록을 더블클릭해 데이터를 넣습니다" }
      : { key: "data", name: "데이터", state: "pending", detail: "Input 블록 없음",
          hint: "Tab으로 팔레트를 열어 Input을 놓습니다" };

  let model: Stage;
  if (!modules) {
    model = { key: "model", name: "모델", state: "pending", detail: "블록 없음",
              hint: "Tab으로 팔레트를 열어 블록을 놓고 잇습니다" };
  } else if (errors) {
    model = { key: "model", name: "모델", state: "error", detail: `오류 ${errors}`,
              hint: "오류 블록을 눌러 진단을 봅니다" };
  } else if (!hasOutput) {
    model = { key: "model", name: "모델", state: "active", detail: `${modules} 블록 · Output 없음`,
              hint: "마지막에 Output 블록을 잇습니다" };
  } else {
    model = { key: "model", name: "모델", state: "done",
              detail: `${modules} 블록 · ${formatCount(input.params)}`,
              hint: "블록을 눌러 값을 고칩니다" };
  }

  const args = (train?.args ?? {}) as Record<string, unknown>;
  const trainStage: Stage = train
    ? { key: "train", name: "학습", state: "done",
        detail: `${String(args.optimizer ?? "adamw")} · lr ${String(args.lr ?? 0.001)} · ${String(args.steps ?? 500)} step`,
        hint: "Train 블록을 눌러 값을 바꿉니다" }
    : { key: "train", name: "학습", state: "pending", detail: "Train 블록 없음",
        hint: "팔레트에서 Train을 놓고 Output 뒤에 잇습니다" };

  let runStage: Stage;
  if (!run) {
    runStage = { key: "run", name: "실행", state: "pending", detail: "버튼을 눌러야 시작",
                 hint: "Run 패널의 Run 또는 Train 블록의 학습 시작" };
  } else if (RUNNING.has(run.state)) {
    runStage = { key: "run", name: "실행", state: "active", detail: runLabel(input.runs),
                 hint: "Run 패널에서 Pause와 Stop" };
  } else if (run.state === "done") {
    runStage = { key: "run", name: "실행", state: "done", detail: runLabel(input.runs),
                 hint: "Run 패널에서 곡선을 봅니다" };
  } else {
    runStage = { key: "run", name: "실행", state: run.state === "paused" ? "active" : "error",
                 detail: runLabel(input.runs), hint: "Run 패널에서 봅니다" };
  }
  return [data, model, trainStage, runStage];
}
