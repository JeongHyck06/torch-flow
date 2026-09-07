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

/** Train 블록이 있으면 hub가 그 값으로 job을 만든다. 없으면 `options`가 Run 폼의 값이다. */
export async function startRun(
  options: Omit<TrainOptions, "dataset" | "recipe" | "smoke"> = {}, smoke = false,
): Promise<StartResult> {
  const { dataset, recipe } = useStore.getState();
  const result = await startTraining(
    smoke ? { smoke: true, dataset, recipe: recipe ?? undefined }
      : { dataset, recipe: recipe ?? undefined, ...options });
  if (result.error) return { error: result.error, fix: result.fix ?? null };
  useStore.getState().setRuns([...useStore.getState().runs, result]);
  return { run: result };
}
