import { useCallback, useEffect, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";

import { Canvas } from "./components/Canvas";
import { CodeView } from "./components/CodeView";
import { DataDialog } from "./components/DataDialog";
import { Inspector } from "./components/Inspector";
import { LayerStrip } from "./components/LayerStrip";
import { RunPanel } from "./components/RunPanel";
import { StartScreen } from "./components/StartScreen";
import { TopBar } from "./components/TopBar";
import { connect, fetchGraph, fetchHealth, fetchLayout, fetchTraining, runShapes } from "./api";
import { useStore } from "./store";

export function App() {
  const openGraph = useStore((state) => state.openGraph);
  const applyNodeState = useStore((state) => state.applyNodeState);
  const setKernel = useStore((state) => state.setKernel);
  const setTotals = useStore((state) => state.setTotals);
  const graph = useStore((state) => state.graph);
  const tab = useStore((state) => state.tab);
  const dataOpen = useStore((state) => state.dataOpen);
  const setRuns = useStore((state) => state.setRuns);
  const open = graph !== null;

  // run 상태는 그래프가 열려 있는 동안 2초마다 읽는다. 패널이 닫혀 있어도 단계 표시줄과
  // Train 노드가 "지금 학습 중인지"를 보여야 한다.
  useEffect(() => {
    if (!open) return;
    let stop = false;
    const tick = () => fetchTraining().then((runs) => { if (!stop) setRuns(runs); }).catch(() => undefined);
    void tick();
    const timer = window.setInterval(tick, 2000);
    return () => { stop = true; window.clearInterval(timer); };
  }, [open, setRuns]);
  const [checked, setChecked] = useState(false);

  const load = useCallback(async () => {
    const health = await fetchHealth();
    setKernel(health.kernel.alive);
    // 학습 장치 목록은 커널이 알려 준 것이다. 원격 GPU 서버면 그 서버의 카드가 온다.
    useStore.getState().setDevices(health.devices ?? []);
    useStore.getState().setAttached(Boolean(health.attached));
    setChecked(true);
    if (health.graph_open === false) return;

    // 좌표를 그래프보다 먼저 넣는다. 순서가 반대면 노드가 자동 배치 자리에 한 번
    // 그려졌다가 layout.json 자리로 튀고, 그 사이에 맞춘 뷰가 어긋난 채 남는다.
    const [{ graph: ir, seq, dirty, project }, positions] = await Promise.all([fetchGraph(), fetchLayout()]);
    useStore.getState().setPositions(positions);
    openGraph(ir, seq);
    // 저장 여부는 hub가 안다. 새로고침한 뒤에도 "저장 안 됨"이 남아야 편집을 잃지 않는다.
    useStore.getState().setDirty(Boolean(dirty));
    useStore.getState().setProject(project ?? null);
    const shapes = await runShapes();
    for (const state of shapes.node_states) applyNodeState(state);
    // 파라미터 총계는 L0 패스가 센 값이 정본이다(§2.2 상단 바).
    setTotals({ params: shapes.total_params });
    setKernel(true);
  }, [openGraph, applyNodeState, setKernel, setTotals]);

  useEffect(() => {
    const disconnect = connect();
    load().catch((error) => console.error(error));
    return disconnect;
  }, [load]);

  if (!graph) {
    return checked ? <StartScreen onOpened={() => void load()} /> : null;
  }

  return (
    <div className="app">
      <TopBar />
      <main className="body">
        {tab === "code" ? <CodeView /> : (
          <>
            <LayerStrip />
            <div className="canvas">
              <ReactFlowProvider>
                <Canvas />
              </ReactFlowProvider>
              <RunPanel />
              {dataOpen && <DataDialog />}
            </div>
          </>
        )}
        <Inspector />
      </main>
    </div>
  );
}
