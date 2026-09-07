// 지금 어느 단계인지 - 데이터, 모델, 학습, 실행. 캔버스 왼쪽 위에 떠 있고 누르면 그 단계로 간다.
//
// 학습은 여기서 시작하지 않는다. 실행 칸은 Run 패널을 열 뿐이고, 시작은 사람이 Run이나
// 학습 시작 버튼을 눌러야 한다.

import { useReactFlow } from "@xyflow/react";

import { computeStages, kindOf } from "../stages";
import type { Stage } from "../stages";
import { useStore } from "../store";

const GLYPH: Record<Stage["state"], string> = { done: "✓", active: "●", pending: "○", error: "~" };

export function Stages() {
  const graph = useStore((state) => state.graph);
  const nodeStates = useStore((state) => state.nodeStates);
  const dataset = useStore((state) => state.dataset);
  const runs = useStore((state) => state.runs);
  const params = useStore((state) => state.totals.params);
  const openData = useStore((state) => state.openData);
  const openPalette = useStore((state) => state.openPalette);
  const runPanel = useStore((state) => state.runPanel);
  const toggleRunPanel = useStore((state) => state.toggleRunPanel);
  const select = useStore((state) => state.select);
  const focus = useStore((state) => state.focus);
  const { screenToFlowPosition } = useReactFlow();

  const stages = computeStages({ graph, nodeStates, dataset, runs, params });
  const nodes = graph?.graph.nodes ?? [];
  const palette = () => openPalette(screenToFlowPosition({ x: window.innerWidth / 2, y: 240 }));
  const pick = (id: string) => { select(id); focus(id); };

  const go = (stage: Stage) => {
    switch (stage.key) {
      case "data":
        if (nodes.some((node) => kindOf(node) === "torchflow.Input")) openData();
        else palette();
        break;
      case "model": {
        const broken = nodes.find((node) => nodeStates[node.id]?.state === "error");
        if (broken) pick(broken.id);
        else if (stage.state !== "done") palette();
        break;
      }
      case "train": {
        const train = nodes.find((node) => kindOf(node) === "torchflow.Train");
        if (train) pick(train.id);
        else palette();
        break;
      }
      case "run":
        if (!runPanel) toggleRunPanel();
        break;
    }
  };

  return (
    <ol className="stages nopan" aria-label="단계">
      {stages.map((stage, index) => (
        <li key={stage.key} className={`stages__item stages__item--${stage.state}`}>
          <button className="stages__button" onClick={() => go(stage)} title={stage.hint}>
            <span className="stages__name">
              <span className="stages__glyph" aria-hidden>{GLYPH[stage.state]}</span>
              {index + 1} {stage.name}
            </span>
            <span className="stages__detail mono">{stage.detail}</span>
          </button>
        </li>
      ))}
    </ol>
  );
}
