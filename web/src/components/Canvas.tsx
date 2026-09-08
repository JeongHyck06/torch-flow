// 캔버스. 키보드만으로 완주할 수 있어야 하고(§2.2), 마우스로 노드를 옮길 수 있어야 한다.
//
// 좌표는 IR이 아니라 `layout.json`에 산다(§10.1) - 그래서 노드를 옮겨도 그래프
// 의미는 바뀌지 않고, git에서는 `merge=ours`로 충돌하지 않는다.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background, BackgroundVariant, Controls, MiniMap, ReactFlow, applyEdgeChanges, applyNodeChanges,
  useNodesInitialized, useReactFlow, useStore as useFlowStore,
} from "@xyflow/react";
import type {
  Connection, Edge, EdgeChange, EdgeMouseHandler, FinalConnectionState, NodeChange,
  NodeMouseHandler, Node as FlowNode,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { NodeCard } from "./NodeCard";
import { Palette } from "./Palette";
import { Stages } from "./Stages";
import { SYNTHETIC, currentScope, useStore } from "../store";
import { saveLayout } from "../api";
import { applyEdit, redo, undo } from "../edit";
import { groupOp, op, removeNodeOp } from "../graph/ops";
import { enterableComposite, lodOf, toFlow } from "../graph/toFlow";
import { runLabel } from "../stages";
import { topologicalIds } from "../graph/layout";
import { categoryColor, categoryOf } from "../theme";

const nodeTypes = { tf: NodeCard };
const MIN_FIT_ZOOM = 0.75;

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
  const dataset = useStore((state) => state.dataset);
  const openData = useStore((state) => state.openData);
  const runs = useStore((state) => state.runs);
  const runPanel = useStore((state) => state.runPanel);
  const toggleRunPanel = useStore((state) => state.toggleRunPanel);
  const run = runLabel(runs);

  const [zoom, setZoom] = useState(1);
  // 뷰가 포커스를 따라가는 것은 키보드 탐색일 때만이다. 클릭에도 따라가면
  // 노드를 집으려던 손 밑에서 캔버스가 움직여 포트를 이을 수 없다.
  const followFocus = useRef(false);
  const { fitView, screenToFlowPosition, getNodesBounds, setViewport } = useReactFlow();
  const viewWidth = useFlowStore((flow) => flow.width);
  const viewHeight = useFlowStore((flow) => flow.height);
  const scope = currentScope({ graph, scopes });
  const callPath = scopes[scopes.length - 1].callPath;

  const computed = useMemo(() => {
    if (!graph || !scope) return { nodes: [], edges: [] };
    return toFlow(scope, nodeStates,
      { lod: lodOf(zoom), selected, focused, callPath, gradOverlay, positions,
        dataset: SYNTHETIC.has(dataset) ? "" : dataset, run });
  }, [graph, scope, callPath, nodeStates, zoom, selected, focused, gradOverlay, positions, dataset, run]);

  // 드래그 중에는 로컬 상태가 권위를 갖는다 - 매 프레임 스토어를 때리면 끊긴다.
  const [nodes, setNodes] = useState<FlowNode[]>(computed.nodes);
  // 다시 그릴 때 잰 크기(measured)를 넘겨준다. React Flow는 크기를 모르는 노드를 잴 때까지 숨기므로,
  // 폴링이나 probe로 노드 객체가 새로 만들어질 때마다 블록이 잠깐 사라졌다 돌아왔다.
  useEffect(() => {
    setNodes((current) => {
      const known = new Map(current.map((node) => [node.id, node]));
      return computed.nodes.map((node) => {
        const previous = known.get(node.id);
        // 여러 개를 고른 상태(Cmd+클릭, Shift+드래그)는 React Flow의 로컬 상태에만 있다.
        // 스토어의 단일 선택으로 덮으면 폴링이 한 번 돌 때마다 선택이 하나로 줄어든다.
        const selected = node.selected || previous?.selected === true;
        return previous?.measured?.width
      });
    });
  }, [computed.nodes]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodes((current) => applyNodeChanges(changes, current)),
    [],
  );
  // 엣지도 로컬 상태를 둔다 - 선택(클릭)이 여기 남아야 Delete가 어느 선인지 안다.
  const [edges, setEdges] = useState<Edge[]>(computed.edges);
  useEffect(() => { setEdges(computed.edges); }, [computed.edges]);
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => setEdges((current) => applyEdgeChanges(changes, current)),
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

  // 선을 끊는 것도 평범한 disconnect op다 - 실행 취소하면 connect로 돌아온다.
  const disconnect = useCallback((edge: Edge) => {
    const { src, dst } = edge.data as { src: string; dst: string };
    void applyEdit(op("disconnect", { ...(composite ? { composite } : {}), src, dst }));
  }, [composite]);

  // Cmd+G: 고른 블록들을 묶음 블록 하나로. 되돌리기는 batch 역 op 하나로 통째로 돌아온다.
  const group = useCallback(async () => {
    const live = currentScope(useStore.getState());
    if (!live) return;
    const ids = nodes.filter((node) => node.selected).map((node) => node.id);
    const result = groupOp(live, ids, composite,
                           Object.keys(useStore.getState().graph?.composites ?? {}));
    if ("error" in result) return useStore.getState().setNotice(result.error);
    const failed = await applyEdit(result.op);
    if (failed) return useStore.getState().setNotice(failed);
    // 새 블록은 묶인 것들의 한가운데에 놓는다 - 자동 배치가 화면 밖에 두면 찾을 수 없다.
    const picked = nodes.filter((node) => ids.includes(node.id));
    const spot = {
      x: picked.reduce((sum, node) => sum + node.position.x, 0) / picked.length,
      y: picked.reduce((sum, node) => sum + node.position.y, 0) / picked.length,
    };
    const key = callPath ? `${callPath}/${result.nodeId}` : result.nodeId;
    setPosition(key, spot);
    void saveLayout(key, spot);
    select(result.nodeId);
    focus(result.nodeId);
    useStore.getState().setNotice(`${picked.length}개를 묶음 블록 ${result.name}으로 만들었습니다`);
  }, [nodes, composite, callPath, setPosition, select, focus]);

  const order = useMemo(
    () => (scope ? topologicalIds(scope.nodes ?? [], (scope.edges ?? []) as [string, string][]) : []),
    [scope],
  );

  const enter = useCallback((nodeId: string) => {
    if (!scope) return;
    const composite = enterableComposite(scope, nodeId);
    if (!composite) return;
    const node = scope.nodes?.find((candidate) => candidate.id === nodeId);
    const instance = node?.call ? scope.instances?.[node.call] : undefined;
    // Repeat 본문의 상태 키에는 회차가 붙는다(`blocks#0/…`, l0의 _call_repeat). 첫 회차를 본다.
    const hop = instance?.type === "torchflow.Repeat" ? `${nodeId}#0` : nodeId;
    enterScope({
      name: composite,
      label: composite,
      callPath: callPath ? `${callPath}/${hop}` : hop,
    });
  }, [scope, callPath, enterScope]);

  // 전체 보기. 확대는 0.75 아래로 내려가지 않는다 - 그 아래면 글자가 화면에서 9px도 안 된다.
  // 다 안 들어가면 가운데가 아니라 왼쪽 끝(Input)부터 보이게 맞춘다. 읽는 순서가 왼쪽에서 시작한다.
  const fitReadable = useCallback((duration = 0) => {
    const box = getNodesBounds(nodes.map((node) => node.id));
    if (!box.width || !viewWidth || !viewHeight) return;
    const pad = 40;
    const zoom = Math.min(1, Math.max(MIN_FIT_ZOOM,
      Math.min((viewWidth - 2 * pad) / box.width, (viewHeight - 2 * pad) / box.height)));
    const fits = box.width * zoom <= viewWidth - 2 * pad;
    const x = fits ? (viewWidth - box.width * zoom) / 2 - box.x * zoom : pad - box.x * zoom;
    const y = (viewHeight - box.height * zoom) / 2 - box.y * zoom;
    void setViewport({ x, y, zoom }, { duration });
  }, [nodes, viewWidth, viewHeight, getNodesBounds, setViewport]);

  // 화살표는 위상 순으로 움직인다 - 그래프를 읽는 순서가 곧 탐색 순서다.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.closest("input, textarea, select, [contenteditable='true']")) return;
      // Space·Enter를 꾹 누르면 키 자동 반복이 초당 30번 버튼을 누른다 - 학습 시작 요청이 그 수만큼
      // 나가 터미널이 도배됐다. 반복 키는 아무 버튼도 누르지 못하게 한다.
      if (event.repeat && (event.key === " " || event.key === "Enter")) {
        event.preventDefault();
        return;
      }
      // Backspace는 노드나 캔버스에 포커스가 있을 때만 지운다. 버튼(단계 표시줄, 모듈 트리)에
      // 포커스가 남은 채 입력창인 줄 알고 누르면 블록이 확인 없이 사라졌다.
      if (event.key === "Backspace" && target !== document.body
          && !target.closest(".react-flow__node, .react-flow__pane")) return;
      // 데이터 창이 떠 있으면 키는 그 창의 것이다 - Tab이 뒤에서 팔레트를 열면 안 된다.
      if (useStore.getState().dataOpen) return;
      if (!order.length && event.key !== "Tab") return;
      const index = focused ? order.indexOf(focused) : -1;

      // Space는 포커스한 블록을 선택에 넣고 뺀다. 마우스 없이도 Cmd+G까지 갈 수 있어야 한다(§2.2).
      if (event.key === " " && focused
          && (target === document.body || target.closest(".react-flow__node, .react-flow__pane"))) {
        setNodes((current) => current.map((one) =>
          one.id === focused ? { ...one, selected: !one.selected } : one));
        event.preventDefault();
        return;
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "g") {
        event.preventDefault();
        void group();
        return;
      }
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
      if (event.key === "Delete" || event.key === "Backspace") {
        // 클릭해 둔 선이 있으면 선이 먼저다. 노드 선택은 선을 클릭할 때 비운다.
        const edge = edges.find((candidate) => candidate.selected);
        if (edge) disconnect(edge);
        else if (focused || selected) remove((focused ?? selected) as string);
        else return;
        event.preventDefault();
        return;
      }

      // 뷰가 따라가는 것은 키보드 탐색뿐이다. Esc 같은 키가 이 플래그를 켜 두면 다음 클릭에 뷰가 튄다.
      const go = (id: string | undefined) => { followFocus.current = true; focus(id ?? null); };
      switch (event.key) {
        case "ArrowRight": case "ArrowDown":
          go(order[Math.min(index + 1, order.length - 1)] ?? order[0]); break;
        case "ArrowLeft": case "ArrowUp":
          go(order[Math.max(index - 1, 0)] ?? order[0]); break;
        case "Home": go(order[0]); break;
        case "End": go(order[order.length - 1]); break;
        case "Enter": if (focused) select(focused); break;
        case "]": if (focused) enter(focused); break;
        case "Escape":
          if (scopes.length > 1) popToScope(scopes.length - 2);
          else select(null);
          break;
        case "f": fitReadable(240); return;
        default: return;
      }
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [order, focused, selected, scopes, focus, select, enter, popToScope, fitReadable,
      openPalette, screenToFlowPosition, remove, edges, disconnect, group]);

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
  // 확대 상한 1: 노드 하나짜리 그래프를 2배로 키우면 다음 블록이 놓이는 자리가 화면 밖이다.
  useEffect(() => { fitted.current = false; }, [scopeKey]);
  // 노드가 실측되기 전에 맞추면 0x0 기준으로 맞춰져 화면 밖으로 나간다.
  // 노드가 0개일 때도 "초기화됨"이라 노드가 실제로 들어온 뒤라야 의미가 있다.
  const measured = useNodesInitialized() && computed.nodes.length > 0;
  useEffect(() => {
    if (fitted.current || !measured) return;
    fitted.current = true;
    fitReadable(0);
  }, [measured, fitReadable]);

  const onNodeClick: NodeMouseHandler = (event, node) => {
    // Cmd(또는 Shift)+클릭은 선택에 더하고 뺀다. React Flow의 키 추적에 기대지 않고
    // 여기서 직접 켜고 끄는 이유는, 그래야 무엇이 골라졌는지가 한 곳(로컬 nodes)에만 있기 때문이다.
    if (event.metaKey || event.ctrlKey || event.shiftKey) {
      setNodes((current) => current.map((one) =>
        one.id === node.id ? { ...one, selected: !one.selected } : one));
      return;
    }
    select(node.id);
    focus(node.id);
  };
  // 선을 고르면 노드 선택은 비운다 - Delete가 노드를 지우면 안 된다.
  const onEdgeClick: EdgeMouseHandler = () => { select(null); focus(null); };
  const onNodeDoubleClick: NodeMouseHandler = (_event, node) => {
    // Input은 임포트 블록처럼 더블클릭으로 데이터 창을 연다 - 안으로 들어갈 것이 없다.
    const kind = scope?.nodes?.find((one) => one.id === node.id)?.type?.split("@")[0];
    if (kind === "torchflow.Input") openData();
    // Train은 Run 패널을 열 뿐이다. 학습은 거기서 버튼을 눌러야 시작한다.
    else if (kind === "torchflow.Train") { if (!runPanel) toggleRunPanel(); }
    else enter(node.id);
  };

  const onPaneDoubleClick = (event: React.MouseEvent) => {
    // ReactFlow의 onDoubleClick은 노드 위에서도 발화한다 - 거기서는 컴포지트 진입이 맞다.
    if ((event.target as HTMLElement).closest(".react-flow__node")) return;
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

  // 엣지를 빈 곳에 떨어뜨리면 팔레트가 열린다 - 이을 수 있는 블록만 보이고, 고르면 그 선이
  // 그대로 이어진다(§4.2 컨텍스트 필터). 포트를 두 번 찍는 수고가 한 번으로 준다.
  const onConnectEnd = useCallback((event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
    if (state.isValid || !state.fromNode || state.fromHandle?.type !== "source") return;
    const point = "changedTouches" in event ? event.changedTouches[0] : event;
    const source = (scope?.nodes ?? []).find((node) => node.id === state.fromNode!.id);
    const port = state.fromHandle?.id ?? source?.ports_out?.[0]?.name ?? "output";
    openPalette(screenToFlowPosition({ x: point.clientX, y: point.clientY }),
                `${state.fromNode.id}.${port}`);
  }, [scope, openPalette, screenToFlowPosition]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onEdgeClick={onEdgeClick}
      deleteKeyCode={null}
      onNodeDragStop={onNodeDragStop}
      onNodeClick={onNodeClick}
      onNodeDoubleClick={onNodeDoubleClick}
      onDoubleClick={onPaneDoubleClick}
      onConnect={onConnect}
      onConnectEnd={onConnectEnd}
      onMove={(_event, viewport) => setZoom(viewport.zoom)}
      fitView
      minZoom={0.1}
      panOnScroll
      proOptions={{ hideAttribution: true }}
      nodesDraggable
      aria-label="모델 그래프 캔버스"
      // 노드 글자가 줌에 반비례해 커질 수 있게 현재 줌을 CSS로 넘긴다(.node--far/.node--mid).
      style={{ "--zoom": zoom } as React.CSSProperties}
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
      {nodes.length > 0 && <Stages />}
    </ReactFlow>
  );
}
