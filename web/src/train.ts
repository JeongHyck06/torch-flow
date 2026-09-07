// Run 한 번의 시작. Run 패널과 Train 블록의 Inspector가 같은 함수를 쓴다.
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

/** Train 블록이 있으면 hub가 그 값으로 job을 만든다. 없으면 `options`가 Run 폼의 값이다. */
export async function startRun(
  options: Omit<TrainOptions, "dataset" | "recipe" | "smoke"> = {}, smoke = false,
): Promise<StartResult> {
  if (inFlight) return { error: "이미 시작 요청이 나가 있습니다" };
  inFlight = true;
  try {
    const { dataset, recipe } = useStore.getState();
    const result = await startTraining(
      smoke ? { smoke: true, dataset, recipe: recipe ?? undefined }
        : { dataset, recipe: recipe ?? undefined, ...options });
    if (result.error) return { error: result.error, fix: result.fix ?? null };
    useStore.getState().setRuns([...useStore.getState().runs, result]);
    return { run: result };
  } finally {
    inFlight = false;
  }
}
