// 지금 어느 단계인지 - 데이터, 모델, 학습, 실행, 테스트. 캔버스 왼쪽 위에 떠 있고 **누르면 그
// 단계가 진행된다**. 조작은 여기 한 곳이다: 학습 시작과 테스트는 여기서만 시작하고, 아래 패널은
// 결과를 보는 곳(돌고 있는 run의 일시정지·중지만 예외), Inspector는 고른 블록의 설정이다.
//
// L2는 자동으로 돌지 않는다(§5.4) - 4 실행도 사람이 누른 클릭이다.

import { useEffect, useState } from "react";
import { useReactFlow } from "@xyflow/react";

import { fetchDatasets, fetchRegistry, saveLayout } from "../api";
import { applyEdit } from "../edit";
import { layeredLayout } from "../graph/layout";
import { addBlockOp, op } from "../graph/ops";
import { RUNNING, computeStages, kindOf } from "../stages";
import type { Stage } from "../stages";
import { useStore } from "../store";
import { startRun } from "../train";

const GLYPH: Record<Stage["state"], string> = { done: "✓", active: "●", pending: "○", error: "~" };

// 학습 블록 추가는 한 번에 하나만 나간다. 단계 버튼에 포커스가 남은 채 Space를 연타하면(키 자동
// 반복이면 초당 30번) 응답이 오기 전에 같은 클릭이 되풀이돼 Train이 그 수만큼 생겼다.
let addingTrain = false;

/** 안내문의 행동 버튼 이름. 없으면 버튼도 없다. */
function actionOf(stage: Stage): string | null {
  switch (stage.key) {
    case "data":
      if (stage.state === "done") return null;
      return stage.detail.includes("블록 없음") ? "팔레트 열기"
        : stage.detail.includes("내려받기") ? "데이터 창 열기" : "데이터 고르기";
    case "model": return stage.state === "error" ? "오류 블록 보기" : stage.state === "done" ? null : "팔레트 열기";
    case "train": return stage.state === "done" ? null : "학습 블록 넣기";
    case "run":
      if (stage.detail.includes("가져온")) return null;
      return stage.state === "pending" ? "학습 시작" : stage.state === "active" ? "곡선 보기" : "다시 시작";
    case "test": return stage.state === "active" ? "정확도 재기" : stage.state === "done" ? "결과 보기" : null;
  }
}

export function Stages() {
  const graph = useStore((state) => state.graph);
  const nodeStates = useStore((state) => state.nodeStates);
  const dataset = useStore((state) => state.dataset);
  const dataOpen = useStore((state) => state.dataOpen);
  const runs = useStore((state) => state.runs);
  const params = useStore((state) => state.totals.params);
  const openData = useStore((state) => state.openData);
  const openPalette = useStore((state) => state.openPalette);
  const openRunPanel = useStore((state) => state.openRunPanel);
  const select = useStore((state) => state.select);
  const focus = useStore((state) => state.focus);
  const notice = useStore((state) => state.notice);
  const setNotice = useStore((state) => state.setNotice);
  const startError = useStore((state) => state.startError);
  const guideHidden = useStore((state) => state.guideHidden);
  const setGuideHidden = useStore((state) => state.setGuideHidden);
  const { screenToFlowPosition } = useReactFlow();

  // 붙은 데이터의 파일이 있는지. 데이터 창이 닫힐 때 다시 읽는다 - 거기서 내려받으니까.
  const [available, setAvailable] = useState<Record<string, boolean>>({});
  useEffect(() => {
    if (dataOpen) return;
    fetchDatasets()
      .then((list) => setAvailable(Object.fromEntries(list.map((entry) => [entry.name, entry.available]))))
      .catch(() => undefined);
  }, [dataOpen]);

  // 거부된 편집 같은 한 줄 알림은 잠깐만 떠 있는다.
  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 8000);
    return () => window.clearTimeout(timer);
  }, [notice, setNotice]);

  const stages = computeStages({ graph, nodeStates, dataset, runs, params,
                                 datasetAvailable: available[dataset] });
  const nodes = graph?.graph.nodes ?? [];
  const palette = () => openPalette(screenToFlowPosition({ x: window.innerWidth / 2, y: 240 }));
  const pick = (id: string) => { select(id); focus(id); };

  // 학습 블록을 Output 뒤에 잇고 고른다. 팔레트에서 찾아 놓고 선을 잇는 세 동작이 한 번이다.
  // op 둘을 따로 보내는 이유는 batch에 역 op가 없어서다 - 따로면 둘 다 실행 취소가 된다.
  const addTrain = async () => {
    // 렌더 시점의 nodes가 아니라 지금 그래프를 본다 - 방금 넣은 Train이 응답 전이면 nodes엔 없다.
    const latest = useStore.getState().graph;
    if (!latest || addingTrain) return;
    const current = latest.graph.nodes ?? [];
    if (current.some((node) => kindOf(node) === "torchflow.Train")) return;
    addingTrain = true;
    try {
      const block = (await fetchRegistry()).find((one) => one.type === "torchflow.Train");
      if (!block) return;
      const { op: add, nodeId } = addBlockOp(latest.graph, block, null);
      const failed = await applyEdit(add);
      if (failed) { setNotice(failed); return; }
      const output = current.find((node) => kindOf(node) === "torchflow.Output");
      if (output) {
        await applyEdit(op("connect", { src: `${output.id}.output`, dst: `${nodeId}.input` }));
        const auto = layeredLayout(current, (latest.graph.edges ?? []) as [string, string][]);
        const base = useStore.getState().positions[output.id] ?? auto[output.id];
        if (base) {
          const spot = { x: base.x + 200, y: base.y };
          useStore.getState().setPosition(nodeId, spot);
          void saveLayout(nodeId, spot);
        }
      }
      pick(nodeId);
    } finally {
      addingTrain = false;
    }
  };

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
        else void addTrain();
        break;
      }
      case "run": {
        if (stage.detail.includes("가져온")) {
          setNotice(stage.guide);
          break;
        }
        const run = runs[runs.length - 1];
        if (run && (RUNNING.has(run.state) || run.state === "paused")) {
          openRunPanel("curves");
          break;
        }
        // 유일한 시작 버튼. 곡선이든 오류든 결과는 아래 패널에 보인다.
        void startRun().then(() => openRunPanel("curves"));
        break;
      }
      case "test":
        // 결과가 없는 끝난 run은 탭이 열리는 순간 돌아간다(TestPanel).
        openRunPanel("test");
        break;
    }
  };

  // 안내문이 가리키는 단계: 오류가 있으면 그것, 아니면 돌고 있는 것, 아니면 첫 미완료.
  const current = stages.find((stage) => stage.state === "error")
    ?? [...stages].reverse().find((stage) => stage.state === "active")
    ?? stages.find((stage) => stage.state === "pending")
    ?? stages[stages.length - 1];
  const action = actionOf(current);
  const line = notice ?? startError;

  return (
    <>
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
        <li className="stages__item">
          <button className="stages__button stages__help" onClick={() => setGuideHidden(!guideHidden)}
                  title={guideHidden ? "단계별 안내 보기" : "단계별 안내 숨기기"} aria-pressed={!guideHidden}>
            ?
          </button>
        </li>
      </ol>
      {!guideHidden && (
        <aside className="guide nopan" aria-label="단계별 안내">
          <p className="guide__kicker">
            {current.state === "error" ? "고칠 것" : "지금 할 일"} · {stages.indexOf(current) + 1} {current.name}
          </p>
          <p className="guide__text">{current.guide}</p>
          {line && <p className="guide__notice mono">{line}</p>}
          <div className="guide__actions">
            {action && <button className="solid" onClick={() => go(current)}>{action}</button>}
            <button className="ghost" onClick={() => setGuideHidden(true)}>안내 숨기기</button>
          </div>
        </aside>
      )}
    </>
  );
}
