// 블록 검색 팔레트 (기획서 §4.2, Figma `06 Empty canvas`의 palette 55:228).
//
// 커서 위치 팝업이 기본이다: `Tab` 또는 빈 캔버스 더블클릭. 퍼지 + 별칭 검색
// (`bn`, `ln`, `mha`), 포트 시그니처 미리보기, `Enter`로 삽입.
// 컨텍스트 필터와 Replace는 M7이라 여기서는 검색만 한다.

import { useEffect, useMemo, useRef, useState } from "react";

import { fetchRegistry, saveLayout } from "../api";
import { applyEdit } from "../edit";
import { addBlockOp } from "../graph/ops";
import type { Block } from "../graph/ops";
import { currentScope, useStore } from "../store";

// 관례적으로 쓰는 줄임말. 검색어가 이것들이면 원래 이름으로도 친 것으로 친다.
const ALIASES: Record<string, string> = {
  bn: "BatchNorm", ln: "LayerNorm", gn: "GroupNorm", mha: "MultiheadAttention",
  fc: "Linear", mlp: "Linear", ce: "CrossEntropyLoss", gelu: "GELU", relu: "ReLU",
  conv: "Conv", pool: "Pool", norm: "Norm", drop: "Dropout", emb: "Embedding",
};

const LIMIT = 8;

export function Palette() {
  const at = useStore((state) => state.paletteAt);
  const close = useStore((state) => state.closePalette);
  const graph = useStore((state) => state.graph);
  const scopes = useStore((state) => state.scopes);
  const select = useStore((state) => state.select);

  const [blocks, setBlocks] = useState<Block[]>([]);
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => { fetchRegistry().then(setBlocks).catch(() => undefined); }, []);
  useEffect(() => { if (at) { setQuery(""); setCursor(0); setError(null); } }, [at]);
  useEffect(() => { input.current?.focus(); }, [at]);

  const found = useMemo(() => search(blocks, query), [blocks, query]);
  const scope = currentScope({ graph, scopes });
  const current = scopes[scopes.length - 1];

  if (!at || !scope) return null;

  const insert = async (block: Block) => {
    const { op, nodeId } = addBlockOp(scope, block, current.name === "$graph" ? null : current.name);
    const failed = await applyEdit(op);
    if (failed) { setError(failed); return; }
    // 놓은 자리에 그대로 있어야 한다. 좌표는 그래프가 아니라 layout.json에(§10.1).
    // 키보드로 연달아 열면 같은 자리가 나오므로 이미 찬 자리는 비켜 놓는다.
    const key = current.callPath ? `${current.callPath}/${nodeId}` : nodeId;
    const spot = freeSpot(at, Object.values(useStore.getState().positions));
    useStore.getState().setPosition(key, spot);
    void saveLayout(key, spot);
    select(nodeId);
    close();
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Escape") { close(); return; }
    if (event.key === "ArrowDown") { setCursor((c) => Math.min(c + 1, found.length - 1)); }
    else if (event.key === "ArrowUp") { setCursor((c) => Math.max(c - 1, 0)); }
    else if (event.key === "Enter") { if (found[cursor]) void insert(found[cursor]); }
    else return;
    event.preventDefault();
  };

  return (
    <>
      <div className="palette__scrim" onClick={close} />
      <div className="palette" role="dialog" aria-label="블록 검색">
        <div className="palette__search">
          <input
            ref={input}
            className="palette__input mono"
            value={query}
            placeholder="블록 검색"
            spellCheck={false}
            onChange={(event) => { setQuery(event.target.value); setCursor(0); }}
            onKeyDown={onKeyDown}
            aria-label="블록 검색"
          />
          <span className="palette__count mono">{found.length} / {blocks.length}</span>
        </div>

        <ul className="palette__list">
          {found.map((block, index) => (
            <li key={block.type}>
              <button
                className={`palette__row${index === cursor ? " palette__row--on" : ""}`}
                onMouseEnter={() => setCursor(index)}
                onClick={() => void insert(block)}
              >
                <span className="palette__head">
                  <span className="palette__name">{block.label}</span>
                  <span className="palette__category">{block.category}</span>
                </span>
                <span className="palette__ports mono">{signature(block)}</span>
              </button>
            </li>
          ))}
          {found.length === 0 && <li className="palette__empty mono muted">없습니다</li>}
        </ul>

        <div className="palette__footer mono muted">
          별칭도 됩니다 · bn ln mha {error && <span className="warn">{error}</span>}
        </div>
      </div>
    </>
  );
}

/** 이미 노드가 있는 자리면 오른쪽으로 비켜 간다. 간격은 노드 폭(216)을 넘겨야 겹치지 않는다. */
function freeSpot(at: { x: number; y: number }, taken: { x: number; y: number }[]) {
  const spot = { ...at };
  while (taken.some((other) => Math.abs(other.x - spot.x) < 240 && Math.abs(other.y - spot.y) < 96)) {
    spot.x += 260;
  }
  return spot;
}

/** 포트 이름만 안다. 포트 타입 시그니처(§4.3)는 레지스트리에 아직 없다. */
function signature(block: Block): string {
  const left = block.ports.in.join(", ") || "—";
  const right = block.ports.out.join(", ") || "—";
  return `${left} → ${right}`;
}

/** 부분 문자열 + 별칭 + 흩어진 글자 순서(퍼지). 짧은 이름이 먼저 온다. */
function search(blocks: Block[], query: string): Block[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return blocks.slice(0, LIMIT);
  const alias = (ALIASES[needle] ?? "").toLowerCase();

  const scored = blocks
    .map((block) => {
      const name = block.label.toLowerCase();
      const full = `${block.type} ${block.category}`.toLowerCase();
      if (name.startsWith(needle)) return { block, rank: 0 };
      if (alias && name.includes(alias)) return { block, rank: 1 };
      if (name.includes(needle)) return { block, rank: 2 };
      if (full.includes(needle)) return { block, rank: 3 };
      if (fuzzy(name, needle)) return { block, rank: 4 };
      return null;
    })
    .filter((entry): entry is { block: Block; rank: number } => entry !== null);

  scored.sort((a, b) => a.rank - b.rank || a.block.label.length - b.block.label.length);
  return scored.slice(0, LIMIT).map((entry) => entry.block);
}

function fuzzy(text: string, needle: string): boolean {
  let index = 0;
  for (const letter of text) {
    if (letter === needle[index]) index += 1;
    if (index === needle.length) return true;
  }
  return false;
}
