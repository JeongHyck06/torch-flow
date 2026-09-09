// 왼쪽 블록 독 - Figma `06 Empty canvas`의 Left dock(53:247).
//
// 캔버스 왼쪽에 늘 붙어 있는 블록 목록이다. 팔레트(`Tab`)는 커서 자리에 잠깐 떴다
// 사라지므로 "무엇을 놓을 수 있는지" 자체를 보여 주지 못했다. 여기서는 카테고리와
// 개수가 계속 보이고, 한 줄을 누르면 그 카테고리로 걸러 팔레트가 열린다.
//
// 개수는 레지스트리에서 실제로 오는 값이다(하드코딩 아님). MY BLOCKS는 지금 그래프가
// 들고 있는 묶음 블록이고 - 템플릿에서 온 것은 CATALOG 쪽 팔레트에 이미 있다.

import { useEffect, useState } from "react";

import { fetchLibrary } from "../api";
import type { Block } from "../graph/ops";
import { useStore } from "../store";

// 카테고리 막대 색. 값은 Figma 변수(`--category-*`)이고 tokens.css가 정본이다.
// Figma에 아직 스와치가 없는 카테고리는 중립 슬레이트로 떨어뜨린다 - 검증하지 않은
// 색을 지어내는 것보다 낫다(색은 계열 안에서만 움직인다).
// Figma가 세운 순서. 자주 쓰는 것이 위다 - 레지스트리는 카테고리 이름 가나다순으로
// 주므로 그대로 두면 "구조/제어"가 맨 위에 온다. 여기 없는 카테고리는 뒤에 붙는다.
const ORDER = ["기본 레이어", "컨볼루션", "어텐션/트랜스포머", "정규화", "활성화", "손실",
               "옵티마이저/스케줄러", "텐서 연산", "풀링/리샘플링", "순환/SSM", "데이터",
               "Code Cell", "학습 제어", "구조/제어"];

const BAR: Record<string, string> = {
  "기본 레이어": "--category-layer",
  "컨볼루션": "--category-conv",
  "어텐션/트랜스포머": "--category-attention",
  "정규화": "--category-norm",
  "활성화": "--category-activation",
  "손실": "--category-loss",
  "옵티마이저/스케줄러": "--category-optimizer",
  "텐서 연산": "--category-tensor-op",
  "데이터": "--category-data",
  "학습 제어": "--category-train-control",
  "구조/제어": "--category-structure",
  "Code Cell": "--category-code-cell",
};

export function BlockDock() {
  const composites = useStore((state) => state.graph?.composites);
  const graphName = useStore((state) => state.graph?.name);
  const [blocks, setBlocks] = useState<Block[]>([]);

  useEffect(() => {
    let stopped = false;
    fetchLibrary()
      .then((library) => { if (!stopped) setBlocks(library.blocks); })
      .catch(() => undefined);
    return () => { stopped = true; };
  }, []);

  // 개수 순으로 세우지 않는다 - 블록이 하나 늘고 줄 때마다 목록이 재배열되면 자리를 외울 수 없다.
  const catalog: { name: string; count: number }[] = [];
  for (const block of blocks) {
    const found = catalog.find((entry) => entry.name === block.category);
    if (found) found.count += 1;
    else catalog.push({ name: block.category, count: 1 });
  }
  const rank = (name: string) => (ORDER.indexOf(name) + 1 || ORDER.length + 1);
  catalog.sort((a, b) => rank(a.name) - rank(b.name) || a.name.localeCompare(b.name));

  const mine = Object.keys(composites ?? {});

  // 여는 것은 Canvas가 한다 - 팔레트를 놓을 캔버스 좌표는 거기서만 계산된다.
  const openWith = useStore((state) => state.requestPalette);

  return (
    <aside className="dock" aria-label="블록">
      <button className="dock__search" onClick={() => openWith("")}>
        <span className="dock__searchtext">블록 검색</span>
        <span className="dock__key">Tab</span>
      </button>

      <h3 className="dock__title">Catalog</h3>
      <div className="dock__list">
        {catalog.map((entry) => (
          <button key={entry.name} className="dock__row" onClick={() => openWith(entry.name)}>
            <span className="dock__bar"
                  style={{ background: `var(${BAR[entry.name] ?? "--category-other"})` }} />
            <span className="dock__name">{entry.name}</span>
            <span className="dock__count mono">{entry.count}</span>
          </button>
        ))}
        {!catalog.length && <p className="dock__empty muted">블록 목록을 읽는 중입니다</p>}
      </div>

      {mine.length > 0 && (
        <>
          <h3 className="dock__title">My blocks</h3>
          <div className="dock__list">
            {mine.map((name) => (
              <button key={name} className="dock__row" onClick={() => openWith(name)}
                      title={`${graphName ?? "이 그래프"}에서 만든 묶음 블록`}>
                <span className="dock__mine mono">{name}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </aside>
  );
}
