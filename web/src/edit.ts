// 편집 한 번의 왕복: 역 op 계산 -> 전송 -> 승인된 그래프 재조회 (기획서 §8.2.2).
//
// 여기가 store와 api 사이에 있는 이유는 순환 import를 피하기 위해서다. store는
// 상태만 들고, 전송은 api가 하고, 둘을 엮는 순서만 이 파일에 있다.

import { fetchGraph, sendOp } from "./api";
import { inverseOf } from "./graph/ops";
import type { Op } from "./graph/ops";
import { currentScope, useStore } from "./store";

type Direction = "do" | "undo" | "redo";

async function run(operation: Op, direction: Direction): Promise<string | null> {
  const store = useStore.getState();
  const scope = currentScope(store);
  // 역 op는 편집 **직전** 그래프에서만 정확하다 - 보내기 전에 계산한다.
  const inverse = scope ? inverseOf(scope, operation) : null;

  try {
    const result = await sendOp(operation);
    const { graph, seq } = await fetchGraph();
    useStore.getState().setGraph(graph, seq);
    for (const state of result.node_states) useStore.getState().applyNodeState(state);
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }

  const after = useStore.getState();
  if (inverse) {
    if (direction === "undo") after.pushRedo(inverse);
    else after.pushUndo(inverse, direction === "do");
  }
  if (direction !== "do") after.setDirty(true);
  return null;
}

/** 편집 하나. 실패하면 오류 문구를 돌려주고 그래프는 그대로다(서버가 권위). */
export function applyEdit(operation: Op): Promise<string | null> {
  return run(operation, "do");
}

export async function undo(): Promise<string | null> {
  const operation = useStore.getState().takeUndo();
  return operation ? run(operation, "undo") : null;
}

export async function redo(): Promise<string | null> {
  const operation = useStore.getState().takeRedo();
  return operation ? run(operation, "redo") : null;
}
