// Run 한 번의 시작. 단계 표시줄의 4 실행이 부르는 유일한 시작 경로다.
//
// L2는 절대 자동으로 돌지 않는다 - 블록을 잇거나 값을 바꿔도 학습은 사람이 이 함수를
// 부르는 버튼을 눌러야 시작한다(§5.4의 auto = {L0, L1}).

import { startTraining } from "./api";
import type { PortSpec, TrainOptions, TrainRun } from "./api";
import { useStore } from "./store";

export interface StartResult {
  error?: string;
  fix?: { node: string; ports_out: PortSpec[] } | null;
  run?: TrainRun;
}

// 시작 요청은 한 번에 하나만 나간다. 어떤 경로로든 요청이 겹치면 두 번째부터는 여기서 끝난다 -
// 실제로 클릭 한 번 뒤에 요청이 초당 열 번씩 6분 동안 나가 run 300개가 뜬 적이 있다.
let inFlight = false;
let lastStart = 0;

/** Train 블록이 있으면 hub가 그 값으로 job을 만든다. 없으면 `options`가 Run 폼의 값이다. */
export async function startRun(
  options: Omit<TrainOptions, "dataset" | "recipe" | "smoke"> = {}, smoke = false,
): Promise<StartResult> {
  // 막은 요청은 오류를 **돌려주기만** 한다. 전역 startError에 쓰면 다음 시작 때까지
  // 남아 낡은 문구가 된다 - 보여 줄 필요가 있는 쪽(CPU 재실행 버튼)이 받아서 띄운다.
  if (inFlight) return { error: "이미 시작 요청이 나가 있습니다" };
  // 연타·자동 반복은 한 번으로 친다. 1초 안의 두 번째 요청은 hub까지 가지 않는다.
  if (Date.now() - lastStart < 1000) return { error: "방금 시작 요청을 보냈습니다 - 잠시 뒤 다시 누르세요" };
  lastStart = Date.now();
  inFlight = true;
  try {
    const { dataset, recipe, graph, trainingDevice } = useStore.getState();
    useStore.getState().setStartResult(null, null);
    const result = await startTraining(
      smoke ? { ...options, device: options.device ?? trainingDevice, smoke: true, dataset, recipe: recipe ?? undefined }
        : { dataset, recipe: recipe ?? undefined, device: trainingDevice, ...options });
    if (useStore.getState().graph !== graph) return {};
    // 409면 hub가 막고 있는 run을 같이 준다. 그 run이 끝나면 이 오류는 거짓이 되므로
    // Trainer가 run 목록을 보고 스스로 지운다.
    useStore.getState().setStartResult(result.error ?? null, result.fix ?? null,
                                       result.error ? result.run_id ?? null : null);
    if (result.error) return { error: result.error, fix: result.fix ?? null };
    useStore.getState().setRuns([...useStore.getState().runs, result]);
    return { run: result };
  } catch (error) {
    const message = `학습 시작 요청에 실패했습니다: ${String(error)}`;
    useStore.getState().setStartResult(message, null);
    return { error: message };
  } finally {
    inFlight = false;
  }
}
