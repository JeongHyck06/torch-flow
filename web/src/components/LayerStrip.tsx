// 모듈 트리 - Figma `02 Attach mode`(43:2) 왼쪽의 MODULE TREE.
//
// 캔버스가 그래프의 **모양**을 보여준다면 이건 계층의 **순서**를 보여준다. 지금
// 보고 있는 스코프의 노드를 위에서 아래로 세우고, 각 줄에 실측 shape와 프로브가
// 만든 그림(Feature Map 또는 활성값 분포)을 건다. 캔버스와 같은 상태를 읽으므로
// 별도 요청이 없다 - 프로브가 돌 때마다 여기도 같이 갱신된다.
//
// 학습 중에는 3초마다 스스로 프로브를 돌려 그림을 살려 둔다. L2가 GPU를 잡고
// 있으면 커널이 알아서 CPU forward-only로 내려간다(§5.1.5) - 학습을 방해하지 않는다.

import { useEffect, useState } from "react";

import { fetchTraining, runProbe } from "../api";
import { currentScope, useStore } from "../store";
import { formatRatio, formatShape } from "../theme";
import { FeatureMap, Thumbnail } from "./NodeCard";

// ponytail: 학습 중 3초 고정 주기. 큰 모델에서 CPU 폴백 forward가 3초를 넘으면
// 요청이 쌓인다 - 그때는 직전 프로브의 소요 시간에 맞춰 늘릴 것.
const LIVE_EVERY = 3000;

export function LayerStrip() {
  const graph = useStore((state) => state.graph);
  const scopes = useStore((state) => state.scopes);
  const nodeStates = useStore((state) => state.nodeStates);
  const selected = useStore((state) => state.selected);
  const select = useStore((state) => state.select);
  const attached = useStore((state) => state.attached);
  const [where, setWhere] = useState<string | null>(null);

  useEffect(() => {
    // Attach 모드에서는 사용자 프로세스가 L1을 몰고 있다 - 우리가 부르면 안 된다.
    if (attached) return;
    let stopped = false;
    const tick = async () => {
      const runs = await fetchTraining().catch(() => []);
      if (stopped || !runs.some((run) => run.alive)) return setWhere(null);
      const result = await runProbe().catch(() => undefined);
      if (stopped || !result?.device) return;
      // 학습이 GPU를 잡으면 배지가 사라진다. 왜 사라졌는지 말해 주지 않으면
      // 사람은 그것을 고장으로 읽는다.
      setWhere(result.forward_only ? `${result.device} · forward only` : result.device);
    };
    const timer = window.setInterval(() => void tick(), LIVE_EVERY);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [attached]);

  const scope = graph ? currentScope({ graph, scopes }) : null;
  if (!scope) return null;
  const callPath = scopes[scopes.length - 1]?.callPath ?? "";
  const nodes = scope.nodes ?? [];

  return (
    <aside className="layers" aria-label="모듈 트리">
      <h3 className="layers__title">모듈 트리</h3>
      {where && <p className="layers__where mono muted">{where}</p>}
      <div className="layers__list">
        {nodes.map((node) => {
          const key = callPath ? `${callPath}/${node.id}` : node.id;
          const state = nodeStates[key];
          const badges = state?.badges ?? {};
          const feature = badges.feature as
            { size: number; pixels: string; channels: number; shown: number } | undefined;
          const histogram = badges.histogram as number[] | undefined;
          const ratio = badges.grad_ratio as number | undefined;
          const warn = (badges.grad_warn ?? badges.warn) as string | undefined;
          return (
            <button
              key={node.id}
              className={`layers__row${selected === node.id ? " layers__row--on" : ""}`}
              onClick={() => select(node.id)}
            >
              <span className="layers__thumb">
                {feature ? <FeatureMap map={feature} />
                  : histogram?.length ? <Thumbnail bins={histogram} />
                  : <span className="layers__blank" />}
              </span>
              <span className="layers__text">
                <span className="layers__name">{node.label}</span>
                <span className="layers__shape mono">
                  {formatShape(state?.spec?.shape as (string | number)[] | undefined)}
                </span>
              </span>
              {ratio !== undefined && (
                <span className={`layers__grad mono${warn ? " warn" : ""}`}>
                  {formatRatio(ratio)}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </aside>
  );
}
