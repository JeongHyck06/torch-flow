// "지금 어느 단계인가"를 한 곳에서 센다. 단계 표시줄(Stages)과 Train 노드 배지가 같이 쓴다.

import type { TrainRun } from "./api";
import { SYNTHETIC } from "./store";
import { formatCount } from "./theme";
import type { ModuleGraph, Node, NodeState } from "./types.gen";

export type StageState = "done" | "active" | "pending" | "error";

export interface Stage {
  key: "data" | "model" | "train" | "run" | "test";
  name: string;
  state: StageState;
  detail: string;
  hint: string;
  /** 단계별 안내문. 처음 보는 사람이 "그래서 뭘 누르지"를 한 문단 안에서 알아야 한다. */
  guide: string;
}

export const RUNNING = new Set(["running", "starting", "finalizing"]);
/** 체크포인트가 남는 끝 상태. 테스트는 이 뒤에만 된다. */
export const FINISHED = new Set(["done", "stopped"]);
const STRUCTURAL = new Set(["torchflow.Input", "torchflow.Output", "torchflow.Train"]);

export function kindOf(node: Node): string {
  return (node.type ?? "").split("@")[0];
}

/** run 상태의 한국어 이름. 영어 원시값이 화면에 나가지 않도록 여기 한 곳에서만 고른다. */
const RUN_STATE_LABELS: Record<string, string> = {
  starting: "학습 준비 중", running: "학습 중", finalizing: "최종 평가·저장 중",
  paused: "일시정지", done: "완료", stopped: "중지", failed: "실패",
};
export const runStateLabel = (state: string): string => RUN_STATE_LABELS[state] ?? state;

/** 마지막 run을 한 줄로. 없으면 빈 문자열. */
export function runLabel(runs: TrainRun[]): string {
  const run = runs[runs.length - 1];
  if (!run) return "";
  const step = `step ${run.step.toLocaleString()} / ${run.total.toLocaleString()}`;
  // 마지막 스텝을 지나도 평가와 체크포인트 저장이 남는다. 그 사이를 "완료"로 부르면
  // 사람은 끝난 줄 알고 테스트를 누르는데 체크포인트가 아직 없다.
  if (run.state === "finalizing" || (run.state === "running" && run.total > 0 && run.step >= run.total)) {
    return `최종 평가·저장 중 · ${step}`;
  }
  if (run.state === "running") return step;
  if (RUNNING.has(run.state) || run.state === "paused") return `${runStateLabel(run.state)} · ${step}`;
  return `${runStateLabel(run.state)} · step ${run.step.toLocaleString()}`;
}

export function computeStages(input: {
  graph: ModuleGraph | null;
  nodeStates: Record<string, NodeState>;
  dataset: string;
  /** 붙은 데이터의 파일이 있는가. 모르면(아직 안 읽음) undefined. */
  datasetAvailable?: boolean;
  runs: TrainRun[];
  params: number;
}): Stage[] {
  const nodes = input.graph?.graph.nodes ?? [];
  const hasInput = nodes.some((node) => kindOf(node) === "torchflow.Input");
  const hasOutput = nodes.some((node) => kindOf(node) === "torchflow.Output");
  const train = nodes.find((node) => kindOf(node) === "torchflow.Train");
  const brokenNodes = nodes.filter((node) => input.nodeStates[node.id]?.state === "error");
  const errors = brokenNodes.length;
  const modules = nodes.filter((node) => !STRUCTURAL.has(kindOf(node))).length;
  const run = input.runs[input.runs.length - 1];
  // 트레이스로 가져온 그래프는 코드를 만들지 않으므로 학습도 없다(hub.traced와 같은 판정).
  const traced = (input.graph?.meta as { source?: string } | undefined)?.source === "attach";

  const data: Stage = !SYNTHETIC.has(input.dataset)
    ? (input.datasetAvailable === false
      ? { key: "data", name: "데이터", state: "active", detail: `${input.dataset} · 내려받기 필요`,
          hint: "1 데이터를 누르고 내려받기를 누릅니다",
          guide: `${input.dataset} 데이터가 붙어 있지만 파일이 아직 없습니다. 1 데이터를 누르고 `
            + "'내려받기'를 누르세요. 파일을 받아야 학습이 됩니다." }
      : { key: "data", name: "데이터", state: "done", detail: input.dataset,
          hint: "Input을 더블클릭하면 바꿀 수 있습니다",
          guide: `${input.dataset} 데이터가 붙어 있습니다. 바꾸려면 1 데이터를 누르세요.` })
    : hasInput
      ? { key: "data", name: "데이터", state: "pending", detail: "Input에 불러오기",
          hint: "Input 블록을 더블클릭해 데이터를 넣습니다",
          guide: "Input 블록에 데이터를 넣으세요. 1 데이터를 누르면 MNIST·CIFAR-10 같은 내장 데이터를 "
            + "내려받거나 내 폴더(클래스별 이미지 · CSV · x.npy+y.npy)를 고를 수 있습니다. 데이터 없이 "
            + "실행하면 무작위 합성 문제로 돌아서 정확도가 10 % 근처에 머뭅니다." }
      : { key: "data", name: "데이터", state: "pending", detail: "Input 블록 없음",
          hint: "Tab으로 팔레트를 열어 Input을 놓습니다",
          guide: "Tab을 눌러 팔레트를 열고 Input 블록을 놓으세요. 데이터는 이 블록으로 들어옵니다." };

  let model: Stage;
  if (!modules) {
    model = { key: "model", name: "모델", state: "pending", detail: "블록 없음",
              hint: "Tab으로 팔레트를 열어 블록을 놓고 잇습니다",
              guide: "Tab으로 팔레트를 열어 Conv2d, ReLU, MaxPool2d, Flatten, Linear 같은 블록을 놓으세요. "
                + "블록 오른쪽 점을 다음 블록 왼쪽 점으로 끌면 이어집니다. 마지막에는 Output 블록을 잇습니다." };
  } else if (errors) {
    const names = brokenNodes.map((node) => node.label || node.id).join(", ");
    model = { key: "model", name: "모델", state: "error", detail: `오류 ${errors}`,
              hint: "오류 블록을 눌러 진단을 봅니다",
              guide: `${names} 블록에 문제가 있습니다. 블록을 누르면 오른쪽에 원인과 '맞추기' 버튼이 보입니다. `
                + "필수 인자가 비었거나 앞 블록의 출력과 크기가 안 맞는 경우가 대부분입니다." };
  } else if (!hasOutput) {
    model = { key: "model", name: "모델", state: "active", detail: `${modules} 블록 · Output 없음`,
              hint: "마지막에 Output 블록을 잇습니다",
              guide: "모델 끝에 Output 블록을 이어 주세요. Output까지 이어져야 학습할 수 있습니다." };
  } else {
    model = { key: "model", name: "모델", state: "done",
              detail: `${modules} 블록 · ${formatCount(input.params)}`,
              hint: "블록을 눌러 값을 고칩니다",
              guide: "모델이 완성됐습니다. 블록을 누르면 값을 고칠 수 있고, 위의 Σ params가 모델 크기입니다." };
  }

  const args = (train?.args ?? {}) as Record<string, unknown>;
  const recipe = `${String(args.optimizer ?? "adamw")} · lr ${String(args.lr ?? 0.001)} · ${String(args.steps ?? 500)} step`;
  const trainStage: Stage = train
    ? { key: "train", name: "학습 설정", state: "done", detail: recipe,
        hint: "Train 블록을 눌러 값을 바꿉니다",
        guide: `학습 설정은 ${recipe} · batch ${String(args.batch ?? 32)}입니다. 값을 바꾸려면 Train 블록을 누르세요. `
          + "loss가 안 내려가면 lr을 10분의 1로, 정확도가 낮으면 steps를 늘려 보세요." }
    : { key: "train", name: "학습 설정", state: "pending", detail: "Train 블록 없음 · 기본값",
        hint: "눌러서 학습 블록을 Output 뒤에 넣습니다. 없으면 기본값으로 돕니다",
        guide: "3 학습 설정을 누르면 Train 블록이 Output 뒤에 붙습니다. optimizer·lr·steps·batch를 거기서 "
          + "고칩니다. 기본값 500 step은 빠른 확인용이라 정확도가 레시피보다 낮게 나옵니다." };

  let runStage: Stage;
  if (traced) {
    runStage = { key: "run", name: "학습", state: "pending", detail: "가져온 그래프 · 학습 없음",
                 hint: "트레이스로 가져온 그래프는 코드를 만들지 않습니다",
                 guide: "트레이스로 가져온 그래프는 구조와 실측 shape만 봅니다. 코드를 만들지 않으므로 "
                   + "학습은 원본 .py에서 하세요." };
  } else if (!run) {
    runStage = { key: "run", name: "학습", state: "pending", detail: "눌러서 학습 시작",
                 hint: "학습 블록의 값으로 학습을 시작합니다",
                 guide: "4 학습을 누르면 이 설정으로 학습이 시작되고 아래 패널에 loss 곡선이 그려집니다. "
                   + "loss가 내려가고 acc가 올라가면 배우고 있는 것입니다." };
  } else if (RUNNING.has(run.state)) {
    runStage = { key: "run", name: "학습", state: "active", detail: runLabel(input.runs),
                 hint: "누르면 아래 패널 · 일시정지와 중지는 패널에서",
                 guide: "학습 중입니다. 아래 패널에서 곡선을 보고 일시정지·중지로 조절합니다. loss가 "
                   + "ln(클래스 수) 근처에 그대로면 데이터가 없거나 lr이 맞지 않는 것입니다." };
  } else if (run.state === "done") {
    runStage = { key: "run", name: "학습", state: "done", detail: runLabel(input.runs),
                 hint: "다시 누르면 새 run을 시작합니다",
                 guide: "학습이 끝났습니다. 5 테스트로 정확도를 재거나, 값을 바꿔 4 학습으로 새 run을 시작하세요." };
  } else if (run.state === "paused") {
    runStage = { key: "run", name: "학습", state: "active", detail: runLabel(input.runs),
                 hint: "누르면 아래 패널 · 이어 가기와 중지는 패널에서",
                 guide: "일시정지 상태입니다. 아래 패널에서 이어 가거나 멈출 수 있습니다." };
  } else {
    // 여기에 영어 오류 원문을 붙이면 안내문이 통째로 RuntimeError가 된다. 무엇을 하면
    // 되는지만 한국어로 말하고, 원문과 traceback은 아래 실패 카드가 펼쳐서 보여 준다.
    runStage = { key: "run", name: "학습", state: "error", detail: runLabel(input.runs),
                 hint: "다시 누르면 새 실행 · 이어 하려면 패널의 재개",
                 guide: run.state !== "failed"
                   ? "학습을 중간에 멈췄습니다. 아래 패널에서 이어 하거나 4 학습으로 새로 시작하세요."
                   : run.cpu_retry
                     ? "Apple GPU(MPS)가 이 모델의 연산 하나를 지원하지 않아 학습이 멈췄습니다. "
                       + "아래 패널의 'CPU로 다시 실행'을 누르면 같은 설정으로 CPU에서 이어서 해 봅니다."
                     : "학습에 실패했습니다. 아래 패널에 무엇이 잘못됐는지와 오류 원문이 있습니다." };
  }
  let testStage: Stage;
  if (!run || !FINISHED.has(run.state)) {
    testStage = { key: "test", name: "테스트", state: "pending", detail: "학습이 끝나면",
                  hint: "학습이 끝나면 보지 않은 데이터로 정확도를 잽니다",
                  guide: "학습이 끝나면 학습에 쓰지 않은 데이터로 정확도를 잽니다." };
  } else if (run.test_acc === undefined) {
    testStage = { key: "test", name: "테스트", state: "active", detail: "눌러서 정확도 재기",
                  hint: "체크포인트를 학습이 보지 않은 분할에 돌립니다",
                  guide: "5 테스트를 누르면 체크포인트를 보지 않은 데이터에 돌려 정확도, 클래스별 정확도, "
                    + "틀린 샘플을 보여 줍니다." };
  } else {
    testStage = { key: "test", name: "테스트", state: "done",
                  detail: `정확도 ${(run.test_acc * 100).toFixed(1)} %`,
                  hint: "Run 패널의 테스트 탭에서 틀린 샘플을 봅니다",
                  guide: `정확도 ${(run.test_acc * 100).toFixed(1)} %입니다. 아래 테스트 탭에서 틀린 샘플을 보고, `
                    + "Code 탭에서 PyTorch 코드를 꺼내거나 저장으로 그래프를 남기세요." };
  }
  return [data, model, trainStage, runStage, testStage];
}
