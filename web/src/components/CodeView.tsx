// Code 탭 - 그래프에서 생성된 PyTorch 코드 (기획서 §7.3 model-only).
//
// 읽기 전용이다. 디스크에는 저장·Run·Export 시점에만 쓰고 그 사이에는 메모리
// 렌더링이라는 규약(§7.6.2)을 따르므로, 여기서는 hub가 그때그때 만들어 준다.
// user-slot 편집과 파일 감시는 M6 후반이다.

import { useEffect, useState } from "react";

import { fetchCode } from "../api";
import { useStore } from "../store";

export function CodeView() {
  const seq = useStore((state) => state.seq);
  // seq만 보면 안 된다. 새 그래프의 seq는 0이라 편집 없이 다른 그래프를 열면
  // 값이 그대로여서 이전 그래프의 코드가 남는다.
  const graph = useStore((state) => state.graph);
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [lines, setLines] = useState(0);

  useEffect(() => {
    let cancelled = false;
    fetchCode().then((body) => {
      if (cancelled) return;
      setError(body.error ?? null);
      setCode(body.code ?? "");
      setLines(body.lines ?? 0);
    }).catch((reason) => setError(String(reason)));
    return () => { cancelled = true; };
  }, [seq, graph]);

  return (
    <section className="codeview" aria-label="생성 코드">
      <div className="codeview__bar mono">
        <span>model.py · {lines}줄 · 읽기 전용</span>
        <span className="muted">torchflow codegen graph/model.tfg.json --out model.py</span>
      </div>
      {error
        ? <p className="mono codeview__error">{error}</p>
        : <pre className="mono codeview__source">{code}</pre>}
    </section>
  );
}
