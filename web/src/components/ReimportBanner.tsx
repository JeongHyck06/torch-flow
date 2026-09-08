// 코드가 밖에서 바뀌었을 때 (기획서 §7.6.2).
//
// 자동으로 반영하지 않는다. 무엇이 달라지는지 먼저 보여 주고 사람이 정한다 -
// 편집기에서 고친 줄과 그래프의 편집이 서로를 조용히 덮으면 신뢰가 깨진다.

import { useState } from "react";

import { fetchGraph, reimport, runShapes } from "../api";
import { useStore } from "../store";
import { useDialog } from "../useDialog";

export function ReimportBanner() {
  const change = useStore((state) => state.sourceChange);
  const [preview, setPreview] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const dismiss = () => { setPreview(null); setError(null); };
  const box = useDialog<HTMLDivElement>(preview !== null || Boolean(error), dismiss);

  if (!change) return null;
  const file = change.path.split("/").pop();

  const look = async () => {
    setBusy(true);
    const result = await reimport({ path: change.path });
    setBusy(false);
    if (result.error) setError(result.error);
    else setPreview(result.preview ?? []);
  };

  const apply = async () => {
    setBusy(true);
    const result = await reimport({ path: change.path, apply: true });
    if (result.error) {
      setBusy(false);
      setError(result.error);
      return;
    }
    const { graph, seq } = await fetchGraph();
    useStore.getState().openGraph(graph, seq);
    const shapes = await runShapes();
    for (const state of shapes.node_states) useStore.getState().applyNodeState(state);
    useStore.getState().setTotals({ params: shapes.total_params });
    useStore.getState().setSourceChange(null);
    setBusy(false);
    setPreview(null);
  };

  return (
    <>
      <button className="reimport__chip" onClick={look} disabled={busy}
              title={change.path}>
        <span className="dot dot--warn" />
        {file} 가 밖에서 바뀌었습니다 · 다시 읽기
      </button>

      {(preview !== null || error) && (
        <div className="reimport__scrim" onClick={dismiss}>
          <div ref={box} className="reimport" role="dialog" aria-modal="true"
               aria-label="코드에서 다시 읽기" onClick={(event) => event.stopPropagation()}>
            <h3>코드에서 다시 읽기</h3>
            <p className="mono muted">{change.path}</p>
            {error && <p className="mono warn">{error}</p>}
            {preview !== null && (preview.length ? (
              <ul className="reimport__list mono">
                {preview.map((line) => <li key={line}>{line}</li>)}
              </ul>
            ) : (
              <p className="mono muted">그래프에 달라지는 것은 없습니다. 주석이나 공백만 바뀌었습니다.</p>
            ))}
            <p className="mono muted">
              좌표와 라벨, 프로브는 그대로 남습니다. 블록 id가 속성 이름에서 나오기 때문입니다.
            </p>
            <div className="reimport__actions">
              <button className="ghost" onClick={dismiss}>그대로 두기</button>
              <button className="solid" onClick={apply} disabled={busy}>
                {busy ? "읽는 중" : "코드를 그래프에 반영"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
