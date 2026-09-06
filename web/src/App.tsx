import { useCallback, useEffect, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";

import { Canvas } from "./components/Canvas";
import { CodeView } from "./components/CodeView";
import { Inspector } from "./components/Inspector";
import { RunPanel } from "./components/RunPanel";
import { StartScreen } from "./components/StartScreen";
import { TopBar } from "./components/TopBar";
import { connect, fetchGraph, fetchHealth, fetchLayout, runShapes } from "./api";
import { useStore } from "./store";

export function App() {
  const setGraph = useStore((state) => state.setGraph);
  const applyNodeState = useStore((state) => state.applyNodeState);
  const setKernel = useStore((state) => state.setKernel);
  const setTotals = useStore((state) => state.setTotals);
  const graph = useStore((state) => state.graph);
  const tab = useStore((state) => state.tab);
  const [checked, setChecked] = useState(false);

  const load = useCallback(async () => {
    const health = await fetchHealth();
    setKernel(health.kernel.alive);
    useStore.getState().setAttached(Boolean(health.attached));
    setChecked(true);
    if (health.graph_open === false) return;

    // 좌표를 그래프보다 먼저 넣는다. 순서가 반대면 노드가 자동 배치 자리에 한 번
    // 그려졌다가 layout.json 자리로 튀고, 그 사이에 맞춘 뷰가 어긋난 채 남는다.
    const [{ graph: ir, seq }, positions] = await Promise.all([fetchGraph(), fetchLayout()]);
    useStore.getState().setPositions(positions);
    setGraph(ir, seq);
    const shapes = await runShapes();
    for (const state of shapes.node_states) applyNodeState(state);
    // 파라미터 총계는 L0 패스가 센 값이 정본이다(§2.2 상단 바).
    setTotals({ params: shapes.total_params });
    setKernel(true);
  }, [setGraph, applyNodeState, setKernel, setTotals]);

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
          <div className="canvas">
            <ReactFlowProvider>
              <Canvas />
            </ReactFlowProvider>
            <RunPanel />
          </div>
        )}
        <Inspector />
      </main>
    </div>
  );
}
