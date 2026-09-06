// Debug Console - 선택 노드의 마지막 probe 값을 표현식으로 조회한다 (기획서 §5.6.1).
//
// Figma Screens에는 이 패널이 없다(그려진 하단 탭은 Run 곡선·manifest·로그·히스토그램).
// 그래서 새 스타일을 만들지 않고 같은 언더라인 탭 안에 mono 한 줄 입력만 얹었다.
//
// Phase A는 읽기 전용 표현식이다. 문장 실행과 jedi 완성은 v1.

import { useState } from "react";
import { evalExpression } from "../api";
import type { EvalResult } from "../api";
import { useStore } from "../store";

interface Entry { expr: string; result: EvalResult }

export function Console() {
  const selected = useStore((state) => state.selected);
  const scopes = useStore((state) => state.scopes);
  const [expr, setExpr] = useState("");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [busy, setBusy] = useState(false);

  const callPath = scopes[scopes.length - 1].callPath;
  const node = selected ? (callPath ? `${callPath}/${selected}` : selected) : "";

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    const source = expr.trim();
    if (!source || busy) return;
    setBusy(true);
    try {
      const result = await evalExpression(source, node);
      setEntries((previous) => [...previous, { expr: source, result }]);
      setExpr("");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="console">
      <div className="console__log mono">
        {entries.length === 0 && (
          <p className="muted">
            x 입력 · y 출력 · m 모듈 인스턴스 · batch probe 입력. 읽기 전용 표현식만 됩니다
          </p>
        )}
        {entries.map((entry, index) => (
          <div key={index} className="console__entry">
            <div className="console__expr">
              <span className="muted">›</span> {entry.expr}
            </div>
            <pre className={`console__out${entry.result.ok ? "" : " warn"}`}>
              {entry.result.ok ? entry.result.text : entry.result.error}
            </pre>
            {entry.result.spec && (
              <div className="muted console__spec">
                [{entry.result.spec.shape.join(", ")}] {entry.result.spec.dtype}
                {" · "}{entry.result.spec.device}
              </div>
            )}
          </div>
        ))}
      </div>

      <form className="console__prompt mono" onSubmit={submit}>
        <span className="console__node" title="선택 노드">
          {selected ? selected : "노드 미선택"}
        </span>
        <input
          className="console__input mono"
          value={expr}
          onChange={(event) => setExpr(event.target.value)}
          placeholder="y.std()"
          aria-label="디버그 콘솔 표현식"
          spellCheck={false}
          disabled={busy}
        />
      </form>
    </div>
  );
}
