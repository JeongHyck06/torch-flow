import { useCallback, useEffect, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";

import { Canvas } from "./components/Canvas";
import { Inspector } from "./components/Inspector";
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
  const [checked, setChecked] = useState(false);

  const load = useCallback(async () => {
    const health = await fetchHealth();
    setKernel(health.kernel.alive);
    useStore.getState().setAttached(Boolean(health.attached));
    setChecked(true);
    if (health.graph_open === false) return;

    const { graph: ir, seq } = await fetchGraph();
    setGraph(ir, seq);
    useStore.getState().setPositions(await fetchLayout());
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
        <div className="canvas">
          <ReactFlowProvider>
            <Canvas />
          </ReactFlowProvider>
        </div>
        <Inspector />
      </main>
    </div>
  );
}
