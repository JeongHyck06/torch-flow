// 캔버스 시각 언어 - Figma 디자인 파일이 정본이다.
//
// 규칙 셋:
// 1. 색은 214° 단일 계열 하나. 카테고리 17개는 계열의 6단계에 alias된다.
//    계열 밖 색은 state/warn 앰버와 state/error 레드 둘뿐이다.
// 2. 상태는 색 + 링 선 스타일 + 글리프 3중 인코딩. 링은 노드 테두리가 아니라
//    6px 바깥의 분리된 사각형이다.
// 3. 크롬은 무채색. 색은 그래프(노드 스트립·엣지·상태 링·포트)에서만 나온다.

export type NodeStateName =
  | "idle" | "stale" | "queued" | "running" | "ok" | "ok.warn" | "error" | "blocked";

export interface StatusStyle {
  /** 링 색 CSS 변수. idle/ok는 링이 없다. */
  ring: string | null;
  dashed: boolean;
  /** 폰트에 내장된 흑백 글리프만 쓴다 - 이모지는 컬러 폴백되어 무채색 원칙을 깬다. */
  glyph: string;
  color: string;
  label: string;
  dim?: boolean;
}

export const STATUS: Record<NodeStateName, StatusStyle> = {
  idle:      { ring: null, dashed: false, glyph: "",    color: "var(--text-muted)",     label: "idle" },
  stale:     { ring: "var(--state-stale)",   dashed: true,  glyph: "◌", color: "var(--state-stale)",   label: "stale", dim: true },
  queued:    { ring: "var(--state-queued)",  dashed: true,  glyph: "···", color: "var(--state-queued)",  label: "queued" },
  running:   { ring: "var(--state-running)", dashed: false, glyph: "◐", color: "var(--state-running)", label: "running" },
  ok:        { ring: null, dashed: false, glyph: "",    color: "var(--state-ok)",       label: "ok" },
  "ok.warn": { ring: "var(--state-warn)",    dashed: false, glyph: "~", color: "var(--state-warn)",    label: "warning" },
  error:     { ring: "var(--state-error)",   dashed: false, glyph: "!", color: "var(--state-error)",   label: "error" },
  blocked:   { ring: "var(--state-blocked)", dashed: false, glyph: "⊘", color: "var(--state-blocked)", label: "blocked", dim: true },
};

// 카테고리 17개 -> 계열 단계. 단계는 Figma의 category/* 변수 그대로다.
const FAMILY = {
  compute: "var(--category-layer)",       // #1e3e66
  transform: "var(--category-norm)",      // #2a558d
  sequence: "var(--category-attention)",  // #366db5
  data: "var(--category-data)",           // #515f70
  tooling: "var(--category-code-cell)",   // #7c8898
} as const;

export const CATEGORY_FAMILY: Record<string, string> = {
  "기본 레이어": FAMILY.compute,
  "컨볼루션": FAMILY.compute,
  "텐서 연산": FAMILY.compute,
  "정규화": FAMILY.transform,
  "활성화": FAMILY.transform,
  "풀링/리샘플링": FAMILY.transform,
  "어텐션/트랜스포머": FAMILY.sequence,
  "순환/SSM": FAMILY.sequence,
  "데이터": FAMILY.data,
  "손실": FAMILY.data,
  "옵티마이저/스케줄러": FAMILY.data,
  "학습 제어": FAMILY.data,
  "평가": FAMILY.data,
  "구조/제어": FAMILY.tooling,
  "프로브": FAMILY.tooling,
  "Code Cell": FAMILY.tooling,
  "유틸": FAMILY.tooling,
};

export function categoryColor(category: string): string {
  return CATEGORY_FAMILY[category] ?? FAMILY.tooling;
}

// dtype -> 엣지 색. 계열 안에서 명도로만 가른다.
export const DTYPE_COLOR: Record<string, string> = {
  float32: "var(--dtype-f32)",
  float64: "#1a4f96",
  bfloat16: "#4a86c9",
  float16: "#5e93d1",
  int64: "#515f70",
  int32: "#66748a",
  bool: "#7c8898",
};

export function categoryOf(type: string | undefined, call: boolean): string {
  if (!type) return call ? "구조/제어" : "텐서 연산";
  if (type.startsWith("torchflow.") || type.startsWith("composite:")) return "구조/제어";
  if (type.startsWith("cell:")) return "Code Cell";
  const name = type.split("@")[0];
  if (name.startsWith("torch.nn.")) {
    const leaf = name.slice("torch.nn.".length);
    if (/Norm/.test(leaf)) return "정규화";
    if (/^Conv/.test(leaf)) return "컨볼루션";
    if (/Pool|Upsample/.test(leaf)) return "풀링/리샘플링";
    if (/Attention|Transformer/.test(leaf)) return "어텐션/트랜스포머";
    if (/LSTM|GRU|RNN/.test(leaf)) return "순환/SSM";
    if (/Loss/.test(leaf)) return "손실";
    if (/ReLU|GELU|SiLU|Sigmoid|Tanh|Softmax/.test(leaf)) return "활성화";
    return "기본 레이어";
  }
  return "텐서 연산";
}

/** LOD 단계별 노드 폭 (Figma Node 컴포넌트). */
export const LOD_WIDTH = { far: 136, mid: 200, near: 244, focus: 244 } as const;

/** 엣지 굵기 = log(원소 수). 텐서 크기가 굵기로 보인다(§6.1). */
export function edgeWidth(shape: (string | number)[] | undefined): number {
  if (!shape?.length) return 1;
  const elements = shape.reduce<number>(
    (acc, dim) => acc * (typeof dim === "number" ? dim : 32), 1);
  return Math.max(1, Math.min(4, Math.log10(Math.max(elements, 10)) - 1));
}

export function formatShape(shape: (string | number)[] | undefined): string {
  return shape?.length ? `[${shape.join(", ")}]` : "?";
}

export function formatCount(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return String(value);
}

export function formatRatio(value: number | undefined): string {
  if (value === undefined || value === null) return "—";
  const [mantissa, exponent] = value.toExponential(1).split("e");
  const power = Number(exponent);
  return power === 0 ? mantissa : `${mantissa}e${power}`;
}

/**
 * Grad-Flow 오버레이. 계열 안에서 명도로만 가른다 - 계열 밖 색을 쓰지 않는다.
 * 작은 grad는 옅게, 큰 grad는 짙게.
 */
export function gradColor(norm: number | undefined, range: [number, number]): string {
  if (!norm || norm <= 0) return "var(--state-idle)";
  const [low, high] = range;
  const t = high > low ? Math.min(1, Math.max(0, (Math.log10(norm) - low) / (high - low))) : 0.5;
  const lightness = 72 - t * 44;   // 72%(옅음) -> 28%(짙음), 색상은 214° 고정
  return `hsl(214 55% ${lightness}%)`;
}

export function gradRange(norms: number[]): [number, number] {
  const logs = norms.filter((value) => value > 0).map((value) => Math.log10(value));
  return logs.length ? [Math.min(...logs), Math.max(...logs)] : [0, 1];
}
