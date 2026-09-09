// 블록 검색 팔레트 (기획서 §4.2, Figma `06 Empty canvas`의 palette 55:228).
//
// 커서 위치 팝업이 기본이다: `Tab` 또는 빈 캔버스 더블클릭. 퍼지 + 별칭 검색
// (`bn`, `ln`, `mha`), 포트 시그니처 미리보기, `Enter`로 삽입.
//
// 컨텍스트 두 가지가 더 있다(§4.2):
// - 엣지 끝을 빈 곳에 떨어뜨려 열면 **입력이 있는 블록만** 보이고, 고르면 그 선이 바로 이어진다.
// - 블록을 고른 상태로 열면 `Cmd+Enter`가 **바꾸기**다. 앞뒤 배선을 그대로 두고 블록만 갈아 낀다.

import { useEffect, useMemo, useRef, useState } from 'react';

import { fetchLibrary, saveLayout } from '../api';
import type { CompositeEntry } from '../api';
import { applyEdit } from '../edit';
import { layeredLayout } from '../graph/layout';
import { addBlockOp, op, replaceOp } from '../graph/ops';
import type { Block } from '../graph/ops';
import { currentScope, useStore } from '../store';
import { useDialog } from '../useDialog';

// 관례적으로 쓰는 줄임말. 검색어가 이것들이면 원래 이름으로도 친 것으로 친다.
const ALIASES: Record<string, string> = {
    bn: 'BatchNorm',
    ln: 'LayerNorm',
    gn: 'GroupNorm',
    mha: 'MultiheadAttention',
    fc: 'Linear',
    mlp: 'Linear',
    ce: 'CrossEntropyLoss',
    gelu: 'GELU',
    relu: 'ReLU',
    conv: 'Conv',
    pool: 'Pool',
    norm: 'Norm',
    drop: 'Dropout',
    emb: 'Embedding',
};

// 검색 없이 열면 자주 쓰는 순서로 전부 보인다(목록은 스크롤). 8개만 보이면 Conv2d가
// 타이핑해야 나온다는 것을 알 길이 없었다. 같은 순위끼리는 이 목록이 앞선다.
const COMMON = ["Input", "Output", "Conv2d", "Linear", "ReLU", "BatchNorm2d", "MaxPool2d",
                "Flatten", "Dropout", "AdaptiveAvgPool2d", "LayerNorm", "GELU", "Embedding",
                "MultiheadAttention", "Train"];
const commonRank = (block: Block) => {
    const at = COMMON.indexOf(block.label);
    return at < 0 ? COMMON.length : at;
};

export function Palette() {
    const at = useStore((state) => state.paletteAt);
    const from = useStore((state) => state.paletteFrom);
    const preset = useStore((state) => state.paletteQuery);
    const close = useStore((state) => state.closePalette);
    const graph = useStore((state) => state.graph);
    const scopes = useStore((state) => state.scopes);
    const select = useStore((state) => state.select);
    const selected = useStore((state) => state.selected);

    const [blocks, setBlocks] = useState<Block[]>([]);
    const [query, setQuery] = useState('');
    const [cursor, setCursor] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const input = useRef<HTMLInputElement>(null);
    const list = useRef<HTMLUListElement>(null);
    // 마우스가 목록 위에 가만히 있어도 목록이 다시 그려지면 mouseenter가 난다 - 그러면
    // 키보드로 고른 줄이 마우스 밑 줄로 바뀌어 Enter가 엉뚱한 블록을 넣는다.
    const mouse = useRef({ x: -1, y: -1 });

    useEffect(() => {
        // 열 때마다 다시 읽는다 - 지금 그래프에 새로 생긴 컴포지트도 목록에 있어야 한다.
        if (!at) return;
        fetchLibrary()
            .then(({ blocks: leaves, composites }) => setBlocks([...leaves, ...composites.map(asBlock)]))
            .catch(() => undefined);
    }, [at]);
    useEffect(() => {
        if (at) {
            // 왼쪽 독에서 카테고리를 눌러 열면 그 이름이 이미 들어가 있다.
            setQuery(preset);
            setCursor(0);
            setError(null);
        }
    }, [at, preset]);
    useEffect(() => {
        input.current?.focus();
    }, [at]);
    useEffect(() => {
        list.current?.querySelector('.palette__row--on')?.scrollIntoView({ block: 'nearest' });
    }, [cursor]);

    // 엣지에서 열었으면 이을 수 있는 것만 보여 준다. 포트 타입은 아직 없으므로(§4.3)
    // "입력이 하나라도 있는가"가 지금 쓸 수 있는 유일한 조건이다.
    const candidates = useMemo(
        () => (from ? blocks.filter((block) => block.ports.in.length > 0) : blocks), [blocks, from]);
    const found = useMemo(() => search(candidates, query), [candidates, query]);
    // 팔레트도 창이다: Tab이 캔버스로 새지 않고 닫으면 부르던 자리로 돌아간다.
    const box = useDialog<HTMLDivElement>(Boolean(at), close);
    const scope = currentScope({ graph, scopes });
    const current = scopes[scopes.length - 1];

    if (!at || !scope) return null;

    const replaceable = Boolean(selected && !from);

    const insert = async (block: Block, replace = false) => {
        // 묶음 블록은 정의가 그래프에 있어야 부를 수 있다. 없을 때만 정의를 먼저 넣는다.
        const name = block.type.replace(/^composite:/, '');
        if (block.source === 'composite' && !graph?.composites?.[name]) {
            const defined = await applyEdit(op('define_composite', { name, body: block.composite }));
            if (defined) {
                setError(defined);
                return;
            }
        }
        const scopeNow = currentScope(useStore.getState()) ?? scope;
        const composite = current.name === '$graph' ? null : current.name;
        let nodeId: string;
        if (replace && selected) {
            const swap = replaceOp(scopeNow, selected, block, composite);
            if ('error' in swap) {
                setError(swap.error);
                return;
            }
            const refused = await applyEdit(swap.op);
            if (refused) {
                setError(refused);
                return;
            }
            // 자리도 물려받는다 - 바꾼 블록이 딴 데로 뛰면 무엇이 바뀌었는지 알 수 없다.
            const keyOf = (id: string) => (current.callPath ? `${current.callPath}/${id}` : id);
            const spot = useStore.getState().positions[keyOf(selected)];
            if (spot) {
                useStore.getState().setPosition(keyOf(swap.nodeId), spot);
                void saveLayout(keyOf(swap.nodeId), spot);
            }
            select(swap.nodeId);
            close();
            return;
        }
        const { op: add, nodeId: fresh } = addBlockOp(scopeNow, block, composite);
        nodeId = fresh;
        // 엣지에서 열었으면 놓는 것과 잇는 것이 한 번의 편집이다 - 실행 취소도 한 번이다.
        const first = block.ports.in[0] ?? 'input';
        const failed = await applyEdit(
            from ? op('batch', {}, [add, op('connect', {
                ...(composite ? { composite } : {}), src: from, dst: `${nodeId}.${first}`,
            })]) : add);
        if (failed) {
            setError(failed);
            return;
        }
        // 놓은 자리에 그대로 있어야 한다. 좌표는 그래프가 아니라 layout.json에(§10.1).
        // 키보드로 연달아 열면 같은 자리가 나오므로 이미 찬 자리는 비켜 놓는다.
        const keyOf = (id: string) => (current.callPath ? `${current.callPath}/${id}` : id);
        const key = keyOf(nodeId);
        // 점유 판정은 이 스코프에 실제로 그려진 노드만 본다. layout.json 전체를 보면 다른
        // 그래프와 지워진 노드의 좌표까지 자리를 차지해 새 블록이 화면 밖까지 밀린다.
        const auto = layeredLayout(scope.nodes ?? [], (scope.edges ?? []) as [string, string][]);
        const stored = useStore.getState().positions;
        const taken = (scope.nodes ?? [])
            .map((node) => stored[keyOf(node.id)] ?? auto[node.id])
            .filter((position): position is { x: number; y: number } => Boolean(position));
        const spot = freeSpot(at, taken);
        useStore.getState().setPosition(key, spot);
        void saveLayout(key, spot);
        select(nodeId);
        close();
    };

    const onKeyDown = (event: React.KeyboardEvent) => {
        if (event.key === 'Escape') {
            close();
            return;
        }
        if (event.key === 'ArrowDown') {
            setCursor((c) => Math.min(c + 1, found.length - 1));
        } else if (event.key === 'ArrowUp') {
            setCursor((c) => Math.max(c - 1, 0));
        } else if (event.key === 'Enter') {
            if (found[cursor]) void insert(found[cursor], replaceable && (event.metaKey || event.ctrlKey));
        } else return;
        event.preventDefault();
    };

    return (
        <>
            <div className="palette__scrim" onClick={close} />
            <div ref={box} className="palette" role="dialog" aria-modal="true" aria-label="블록 검색">
                <div className="palette__search">
                    <input
                        ref={input}
                        className="palette__input mono"
                        value={query}
                        placeholder="블록 검색"
                        spellCheck={false}
                        onChange={(event) => {
                            setQuery(event.target.value);
                            setCursor(0);
                        }}
                        onKeyDown={onKeyDown}
                        aria-label="블록 검색"
                    />
                    <span className="palette__count mono">
                        {found.length} / {blocks.length}
                    </span>
                </div>

                <ul className="palette__list" ref={list}>
                    {found.map((block, index) => (
                        <li key={block.type}>
                            <button
                                className={`palette__row${index === cursor ? ' palette__row--on' : ''}`}
                                onMouseMove={(event) => {
                                    if (event.clientX === mouse.current.x && event.clientY === mouse.current.y) return;
                                    mouse.current = { x: event.clientX, y: event.clientY };
                                    setCursor(index);
                                }}
                                onClick={(event) =>
                                    void insert(block, replaceable && (event.metaKey || event.ctrlKey))}>
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
                    {from
                        ? '이을 수 있는 블록만 · 고르면 바로 이어집니다'
                        : replaceable
                          ? '별칭도 됩니다 · Cmd+Enter 는 고른 블록을 바꿉니다'
                          : '별칭도 됩니다 · bn ln mha'}
                    {error && <span className="warn"> {error}</span>}
                </div>
            </div>
        </>
    );
}

/** 템플릿의 컴포지트를 팔레트 항목으로. 포트 이름은 정의에서, 인자는 params에서 온다. */
function asBlock(entry: CompositeEntry): Block {
    return {
        type: `composite:${entry.name}`,
        label: entry.name,
        category: `묶음 블록 · ${entry.source}`,
        params: Object.fromEntries(Object.entries(entry.params).map(([key, spec]) =>
            [key, { type: spec.type, default: spec.default }])),
        ports: { in: (entry.ports.in ?? []).map((port) => port.name),
                 out: (entry.ports.out ?? []).map((port) => port.name) },
        doc: entry.doc ?? undefined,
        source: 'composite',
        composite: entry.composite,
    };
}

/** 이미 노드가 있는 자리면 오른쪽으로 비켜 간다. 간격은 노드 폭(216)을 넘겨야 겹치지 않는다. */
function freeSpot(at: { x: number; y: number }, taken: { x: number; y: number }[]) {
    const spot = { ...at };
    while (
        taken.some((other) => Math.abs(other.x - spot.x) < 240 && Math.abs(other.y - spot.y) < 96)
    ) {
        spot.x += 260;
    }
    return spot;
}

/** 포트 이름만 안다. 포트 타입 시그니처(§4.3)는 레지스트리에 아직 없다. */
function signature(block: Block): string {
    const left = block.ports.in.join(', ') || '—';
    const right = block.ports.out.join(', ') || '—';
    return `${left} → ${right}`;
}

/** 부분 문자열 + 별칭 + 흩어진 글자 순서(퍼지). 자주 쓰는 블록, 그다음 짧은 이름이 먼저다. */
function search(blocks: Block[], query: string): Block[] {
    const needle = query.trim().toLowerCase();
    if (!needle) return [...blocks].sort((a, b) => commonRank(a) - commonRank(b));
    const alias = (ALIASES[needle] ?? '').toLowerCase();

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

    scored.sort((a, b) => a.rank - b.rank || commonRank(a.block) - commonRank(b.block)
        || a.block.label.length - b.block.label.length);
    return scored.map((entry) => entry.block);
}

function fuzzy(text: string, needle: string): boolean {
    let index = 0;
    for (const letter of text) {
        if (letter === needle[index]) index += 1;
        if (index === needle.length) return true;
    }
    return false;
}
