// 캔버스. 키보드만으로 완주할 수 있어야 하고(§2.2), 마우스로 노드를 옮길 수 있어야 한다.
//
// 좌표는 IR이 아니라 `layout.json`에 산다(§10.1) - 그래서 노드를 옮겨도 그래프
// 의미는 바뀌지 않고, git에서는 `merge=ours`로 충돌하지 않는다.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background, BackgroundVariant, Controls, MiniMap, ReactFlow, applyNodeChanges,
  useNodesInitialized, useReactFlow,
} from "@xyflow/react";
import type {
  Connection, NodeChange, NodeMouseHandler, Node as FlowNode,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { NodeCard } from "./NodeCard";
import { Palette } from "./Palette";
import { currentScope, useStore } from "../store";
import { saveLayout } from "../api";
import { applyEdit, redo, undo } from "../edit";
import { op, removeNodeOp } from "../graph/ops";
import { enterableComposite, lodOf, toFlow } from "../graph/toFlow";
import { topologicalIds } from "../graph/layout";
import { categoryColor, categoryOf } from "../theme";

const nodeTypes = { tf: NodeCard };

export function Canvas() {
  const graph = useStore((state) => state.graph);
  const scopes = useStore((state) => state.scopes);
  const nodeStates = useStore((state) => state.nodeStates);
  const selected = useStore((state) => state.selected);
  const focused = useStore((state) => state.focused);
  const positions = useStore((state) => state.positions);
  const select = useStore((state) => state.select);
  const focus = useStore((state) => state.focus);
  const enterScope = useStore((state) => state.enterScope);
  const popToScope = useStore((state) => state.popToScope);
  const gradOverlay = useStore((state) => state.gradOverlay);
  const setPosition = useStore((state) => state.setPosition);
  const openPalette = useStore((state) => state.openPalette);
  const paletteAt = useStore((state) => state.paletteAt);

  const [zoom, setZoom] = useState(1);
  // 뷰가 포커스를 따라가는 것은 키보드 탐색일 때만이다. 클릭에도 따라가면
  // 노드를 집으려던 손 밑에서 캔버스가 움직여 포트를 이을 수 없다.
  const followFocus = useRef(false);
  const { fitView, screenToFlowPosition } = useReactFlow();
  const scope = currentScope({ graph, scopes });
  const callPath = scopes[scopes.length - 1].callPath;

  const computed = useMemo(() => {
    if (!graph || !scope) return { nodes: [], edges: [] };
    return toFlow(scope, nodeStates,
      { lod: lodOf(zoom), selected, focused, callPath, gradOverlay, positions });
  }, [graph, scope, callPath, nodeStates, zoom, selected, focused, gradOverlay, positions]);

  // 드래그 중에는 로컬 상태가 권위를 갖는다 - 매 프레임 스토어를 때리면 끊긴다.
  const [nodes, setNodes] = useState<FlowNode[]>(computed.nodes);
  useEffect(() => { setNodes(computed.nodes); }, [computed.nodes]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes((current) => applyNodeChanges(changes, current)),
    [],
  );

  const onNodeDragStop = useCallback((_event: unknown, node: FlowNode) => {
    const key = callPath ? `${callPath}/${node.id}` : node.id;
    setPosition(key, node.position);
    // 좌표는 그래프 의미가 아니므로 실행 상태를 건드리지 않는다(engine의 VISUAL_OPS).
    void saveLayout(key, node.position);
  }, [callPath, setPosition]);

  const composite = scopes[scopes.length - 1].name === "$graph"
    ? null : scopes[scopes.length - 1].name;

  const remove = useCallback((nodeId: string) => {
    void applyEdit(removeNodeOp(nodeId, composite));
    if (selected === nodeId) select(null);
    focus(null);
  }, [composite, selected, select, focus]);

  const order = useMemo(
    () => (scope ? topologicalIds(scope.nodes ?? [], (scope.edges ?? []) as [string, string][]) : []),
    [scope],
  );

  const enter = useCallback((nodeId: string) => {
    if (!scope) return;
    const composite = enterableComposite(scope, nodeId);
    if (!composite) return;
    enterScope({
      name: composite,
      label: composite,
      callPath: callPath ? `${callPath}/${nodeId}` : nodeId,
    });
  }, [scope, callPath, enterScope]);

  // 화살표는 위상 순으로 움직인다 - 그래프를 읽는 순서가 곧 탐색 순서다.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;
      if (!order.length && event.key !== "Tab") return;
      const index = focused ? order.indexOf(focused) : -1;

      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        void (event.shiftKey ? redo() : undo());
        event.preventDefault();
        return;
      }
      if (event.key === "Tab") {
        // 키보드로 열면 화면 위쪽 가운데에 놓는다 - 마우스는 커서 자리에(§4.2).
        openPalette(screenToFlowPosition({ x: window.innerWidth / 2, y: 240 }));
        event.preventDefault();
        return;
      }
      if ((event.key === "Delete" || event.key === "Backspace") && (focused || selected)) {
        remove((focused ?? selected) as string);
        event.preventDefault();
        return;
      }

      followFocus.current = true;
      switch (event.key) {
        case "ArrowRight": case "ArrowDown":
          focus(order[Math.min(index + 1, order.length - 1)] ?? order[0]); break;
        case "ArrowLeft": case "ArrowUp":
          focus(order[Math.max(index - 1, 0)] ?? order[0]); break;
        case "Home": focus(order[0]); break;
        case "End": focus(order[order.length - 1]); break;
        case "Enter": if (focused) select(focused); break;
        case "]": if (focused) enter(focused); break;
        case "Escape":
          if (scopes.length > 1) popToScope(scopes.length - 2);
          else select(null);
          break;
        case "f": fitView({ duration: 200 }); return;
        default: return;
      }
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [order, focused, selected, scopes, focus, select, enter, popToScope, fitView,
      openPalette, screenToFlowPosition, remove]);

  useEffect(() => {
    if (!focused || !followFocus.current) return;
    followFocus.current = false;
    fitView({ nodes: [{ id: focused }], maxZoom: 1.2, duration: 200 });
  }, [focused, fitView]);

  // 뷰를 자동으로 맞추는 때는 둘뿐이다: 스코프를 옮겼을 때와, 그래프가 처음
  // 들어왔을 때. 편집마다 맞추면(setGraph이 scopes를 새로 만든다) 사용자가 잡아 둔
  // 확대·위치가 블록을 놓을 때마다 날아간다.
  const scopeKey = scopes.map((entry) => entry.callPath).join("/");
  const fitted = useRef(false);
  useEffect(() => { fitted.current = false; fitView({ duration: 0 }); }, [scopeKey, fitView]);
  // 노드가 실측되기 전에 맞추면 0x0 기준으로 맞춰져 화면 밖으로 나간다.
  // 노드가 0개일 때도 "초기화됨"이라 노드가 실제로 들어온 뒤라야 의미가 있다.
  const measured = useNodesInitialized() && computed.nodes.length > 0;
  useEffect(() => {
    if (fitted.current || !measured) return;
    fitted.current = true;
    fitView({ duration: 0 });
  }, [measured, fitView]);

  const onNodeClick: NodeMouseHandler = (_event, node) => { select(node.id); focus(node.id); };
  const onNodeDoubleClick: NodeMouseHandler = (_event, node) => enter(node.id);

  const onPaneDoubleClick = (event: React.MouseEvent) => {
    openPalette(screenToFlowPosition({ x: event.clientX, y: event.clientY }));
  };

  // 포트를 이어 붙이면 그대로 connect op다. 배선이 곧 그래프 의미다(§8.2.2).
  //
  // 노드 카드에는 아직 포트가 좌우 하나씩뿐이라 핸들에 이름이 없다. 출력 포트
  // 이름은 IR이 알고 있으므로(`ports_out`) 거기서 가져온다 - Input 노드의 출력은
  // `output`이 아니라 `x`이고, 이름이 틀리면 L0가 "미연결"로 본다.
  const onConnect = useCallback((connection: Connection) => {
    if (!connection.source || !connection.target || !scope) return;
    const source = (scope.nodes ?? []).find((node) => node.id === connection.source);
    const outPort = connection.sourceHandle ?? source?.ports_out?.[0]?.name ?? "output";
    void applyEdit(op("connect", {
      ...(composite ? { composite } : {}),
      src: `${connection.source}.${outPort}`,
      dst: `${connection.target}.${connection.targetHandle ?? "input"}`,
    }));
  }, [composite, scope]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={computed.edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onNodeDragStop={onNodeDragStop}
      onNodeClick={onNodeClick}
      onNodeDoubleClick={onNodeDoubleClick}
      onDoubleClick={onPaneDoubleClick}
      onConnect={onConnect}
      onMove={(_event, viewport) => setZoom(viewport.zoom)}
      fitView
      minZoom={0.1}
      proOptions={{ hideAttribution: true }}
      nodesDraggable
      aria-label="모델 그래프 캔버스"
    >
      <Background variant={BackgroundVariant.Lines} gap={72} size={1} color="var(--border-subtle)" />
      <MiniMap
        pannable zoomable
        bgColor="var(--surface-panel)"
        maskColor="rgba(249, 250, 251, 0.7)"
        nodeColor={(node) => {
          const data = node.data as { category?: string };
          return categoryColor(data.category ?? categoryOf(undefined, false));
        }}
      />
      <Controls showInteractive={false} />
      {nodes.length === 0 && !paletteAt && (
        <div className="emptycanvas">
          <p className="emptycanvas__title">빈 그래프</p>
          <p className="emptycanvas__sub">아무 데나 더블클릭하거나 Tab 을 눌러 첫 블록을 놓으세요</p>
          <p className="emptycanvas__keys mono">
            <span><kbd>Tab</kbd> 팔레트</span>
            <span><kbd>Delete</kbd> 삭제</span>
            <span><kbd>Cmd+Z</kbd> 실행 취소</span>
          </p>
        </div>
      )}
      <Palette />
    </ReactFlow>
  );
}
