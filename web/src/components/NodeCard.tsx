// 캔버스 노드 - Figma `Node` 컴포넌트(8:265)를 그대로 옮긴 것.
//
// 구조: [card = header-strip + body] + [state-ring(분리된 절대배치 사각형)] + [ports]
// 상태 링이 카드 테두리가 아닌 이유: 테두리를 물들이면 카테고리 스트립과
// 경쟁하고, 6px 바깥의 분리된 링이라야 색을 못 봐도 형태로 읽힌다.

import { useEffect, useRef } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { LOD_WIDTH, STATUS, categoryColor, formatCount, formatRatio, formatShape } from "../theme";
import type { NodeStateName } from "../theme";

export type Lod = "far" | "mid" | "near" | "focus";

/** Conv 출력의 채널 격자(§6.3 Feature Map). 커널이 보낸 회색조 바이트를 그대로 찍는다. */
function FeatureMap({ map }: { map: { size: number; pixels: string; channels: number; shown: number } }) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    const bytes = Uint8Array.from(atob(map.pixels), (character) => character.charCodeAt(0));
    const image = context.createImageData(map.size, map.size);
    for (let index = 0; index < bytes.length; index += 1) {
      // 회색조 한 바이트를 RGBA로 편다. 색은 쓰지 않는다 - 크롬은 무채색이다.
      const value = bytes[index];
      image.data.set([value, value, value, 255], index * 4);
    }
    context.putImageData(image, 0, 0);
  }, [map]);

  return (
    <canvas ref={canvas} width={map.size} height={map.size}
            className="node__thumb node__thumb--map"
            title={`채널 ${map.shown} / ${map.channels}`} />
  );
}

/** 64x40 썸네일에 들어가는 활성값 히스토그램. 축도 눈금도 없다 - 모양만 본다. */
function Thumbnail({ bins, color }: { bins: number[]; color?: string }) {
  const peak = Math.max(...bins, 1);
  const step = 64 / bins.length;
  return (
    <svg className="node__thumb node__thumb--chart" viewBox="0 0 64 40"
         role="img" aria-label="활성값 분포">
      {bins.map((count, index) => (
        <rect key={index} x={index * step} y={40 - (count / peak) * 40}
              width={Math.max(step - 0.2, 0.3)} height={(count / peak) * 40}
              fill={color ?? "var(--dtype-f32)"} />
      ))}
    </svg>
  );
}

export interface NodeCardData extends Record<string, unknown> {
  label: string;
  category: string;
  typeLabel: string;
  params?: string;
  state: NodeStateName;
  shape?: (string | number)[];
  dtype?: string;
  elapsedMs?: number;
  cacheHit?: boolean;
  errorMessage?: string;
  paramCount?: number;
  gradNorm?: number;
  gradRatio?: number;
  gradWarn?: string;
  gradColor?: string;
  histogram?: number[];
  feature?: { size: number; pixels: string; channels: number; shown: number };
  folded?: boolean;
  enterable: boolean;
  lod: Lod;
  selected: boolean;
  focused: boolean;
}

export function NodeCard({ data }: NodeProps) {
  const node = data as NodeCardData;
  const status = STATUS[node.state] ?? STATUS.idle;
  const lod = node.lod === "focus" ? "near" : node.lod;
  const showBody = lod !== "far";
  const showNear = lod === "near";

  const ariaLabel = [
    node.label, node.typeLabel, status.label,
    node.shape ? `shape ${formatShape(node.shape)}` : null,
    node.gradRatio !== undefined ? `grad ratio ${formatRatio(node.gradRatio)}` : null,
    node.gradWarn ? `gradient ${node.gradWarn}` : null,
    node.errorMessage ?? null,
  ].filter(Boolean).join(", ");

  return (
    <div
      className={`node node--${lod}${node.focused ? " node--focused" : ""}`}
      role="group"
      aria-label={ariaLabel}
      aria-current={node.focused ? "true" : undefined}
      style={{ width: LOD_WIDTH[lod], opacity: status.dim ? 0.6 : 1 }}
    >
      <Handle type="target" position={Position.Left} className="port" />

      <div className={`node__card${node.selected ? " node__card--selected" : ""}`}>
        <div className="node__strip" style={{ background: categoryColor(node.category) }} />

        <div className="node__body">
          <div className="node__head">
            <span className="node__title">{node.label}</span>
            {status.glyph && (
              <span className="node__glyph" style={{ color: status.color }} title={status.label}>
                {status.glyph}
              </span>
            )}
            {node.enterable && <span className="node__enter" title="더블클릭으로 진입">›</span>}
          </div>

          {showBody && (
            <p className="node__params">
              {showNear && node.params ? node.params : node.typeLabel}
            </p>
          )}

          {showNear ? (
            <div className="node__mid">
              {/* 썸네일은 채울 것이 있을 때만 - 빈 회색 박스는 확대할 때마다 눈에 걸린다.
                  프로브 결과(Grad-Flow 색, Feature Map)가 들어오면 나타난다(§6.1). */}
              {node.feature ? (
                <FeatureMap map={node.feature} />
              ) : node.histogram?.length ? (
                // 활성값 분포 썸네일(§6.3 Histogram). 막대 색은 Grad-Flow 색을 따른다 -
                // 한 칸에 "얼마나 큰 기울기"와 "값이 어떻게 퍼져 있나"가 같이 담긴다.
                <Thumbnail bins={node.histogram} color={node.gradColor} />
              ) : node.gradColor ? (
                <div className="node__thumb"
                     style={{ background: node.gradColor, borderColor: "transparent" }} />
              ) : null}
              <span className="node__shape">{formatShape(node.shape)}</span>
            </div>
          ) : showBody ? (
            <p className="node__shape">{formatShape(node.shape)}</p>
          ) : null}

          {showBody && (
            <div className="node__badges">
              {node.paramCount ? (
                <span className="badge"><span className="badge__glyph">↯</span>
                  {formatCount(node.paramCount)}</span>
              ) : null}
              {node.cacheHit && <span className="badge">cached</span>}
              {!node.cacheHit && node.elapsedMs !== undefined && node.elapsedMs >= 1 && (
                <span className="badge badge--ok">✓ {node.elapsedMs.toFixed(0)} ms</span>
              )}
              {node.gradRatio !== undefined && (
                <span className={`badge ${node.gradWarn ? "badge--warn" : "badge--ok"}`}>
                  {node.gradWarn ? "~ " : ""}grad {formatRatio(node.gradRatio)}
                </span>
              )}
              {node.folded && node.gradNorm !== undefined && (
                <span className={`badge ${node.gradWarn ? "badge--warn" : ""}`}
                      title="접힌 내부의 최대 ‖g‖ (§6.1)">
                  max {node.gradNorm.toExponential(1)}
                </span>
              )}
            </div>
          )}

          {showNear && node.errorMessage && (
            <p className="node__error">{node.errorMessage}</p>
          )}
        </div>
      </div>

      {/* 상태 링 - 카드 바깥 6px, 별도 사각형. 색 + 선 스타일 + 글리프의 두 번째 축. */}
      {status.ring && (
        <div
          className="node__ring"
          aria-hidden
          style={{ borderColor: status.ring, borderStyle: status.dashed ? "dashed" : "solid" }}
        />
      )}

      <Handle type="source" position={Position.Right} className="port" />
    </div>
  );
}
