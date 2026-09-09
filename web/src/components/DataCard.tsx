// 데이터 정제 카드. 첫 화면(데이터로 시작)과 하단 패널의 `데이터` 탭이 같이 쓴다.
//
// 자동 인식(base) 위에 사람이 만진 레시피를 얹고, hub가 적용 결과(spec)와 미리보기를
// 돌려준다. 여기서는 적재하지 않는다 - 크기·클래스·분할 수는 세어서 알 수 있는 것들이다.

import { useEffect, useState } from "react";

import { previewDataset } from "../api";
import type { DatasetPreview, Recipe } from "../api";

const SIZES = [28, 32, 48, 64, 96, 128];

export function DataCard({ name, recipe, onRecipe }: {
  name: string; recipe: Recipe | null; onRecipe: (next: Recipe) => void;
}) {
  const [shown, setShown] = useState<DatasetPreview | null>(null);

  useEffect(() => {
    let cancelled = false;
    // 글자 하나마다 미리보기를 다시 세지 않는다.
    const timer = window.setTimeout(() => {
      previewDataset(name, recipe).then((body) => { if (!cancelled) setShown(body); })
        .catch((reason) => { if (!cancelled) setShown({ error: String(reason) } as DatasetPreview); });
    }, 250);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [name, recipe]);

  if (!shown) return <p className="mono muted">데이터를 읽는 중</p>;
  if (shown.error) return <p className="warn mono">{shown.error}</p>;
  const { base, spec, preview } = shown;
  const full = spec.recipe;
  const set = (key: string, value: unknown) => onRecipe({ ...(recipe ?? {}), [key]: value });
  const toggleIn = (key: string, all: string[], item: string, checked: boolean) => {
    const current = (full[key] as string[] | null | undefined) ?? all;
    set(key, checked ? all.filter((one) => one === item || current.includes(one))
      : current.filter((one) => one !== item));
  };
  const task = String(full.task ?? "classification");
  const target = spec.target;
  const summary = `Input [B, ${spec.shape.join(", ")}] · `
    + (full.series_column
        ? `과거 ${full.window}칸으로 다음 ${full.horizon}칸 · 시간 순 분할 · `
        : "")
    + (task === "regression"
        ? (target ? `회귀 · ${target.min} ~ ${target.max}` : "회귀")
        : `${spec.classes} 클래스`)
    + (spec.split ? ` · train ${spec.split.train.toLocaleString()} / val ${spec.split.val.toLocaleString()}` : "");

  // 내장 데이터는 레시피가 없다 - 왼쪽은 한 줄, 오른쪽 미리보기는 같다.
  const builtin = base.kind === "builtin";
  const classNames = base.class_names ?? [];
  const columns = base.columns ?? [];
  const label = String(full.label_column ?? "");
  const featureNames = columns.map((column) => column.name).filter((column) => column !== label);
  // 낱말이 여럿 든 글자 열만 문장 후보다. "yes"/"no"는 범주라 여기 오지 않는다.
  const textColumns = columns
    .filter((column) => column.name !== label && column.kind === "text" && (column.words ?? 0) >= 3)
    .map((column) => column.name);
  const numericColumns = columns
    .filter((column) => column.kind === "numeric")
    .map((column) => column.name);
  const counts = spec.class_counts ?? {};
  const peak = Math.max(1, ...Object.values(counts));

  return (
    <div className="datacard">
      <div>
        <h3>정제</h3>
        {builtin ? (
          <p className="mono muted">내장 데이터는 정제 설정이 없습니다 · 검증은 test 분할</p>
        ) : (
        <dl className="rows">
          {base.kind === "csv" && !full.series_column && (
            <div><dt>정답 열</dt><dd>
              <select className="trainer__select mono" value={label}
                      onChange={(event) => set("label_column", event.target.value)}>
                {columns.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}
              </select>
            </dd></div>
          )}
          {base.kind === "csv" && !full.series_column && (
            <div><dt>과제</dt><dd>
              {/* 정답 열에서 짐작한 값이 들어와 있다. 바꾸면 loss와 지표가 함께 바뀐다. */}
              <select className="trainer__select mono" value={task}
                      onChange={(event) => set("task", event.target.value)}>
                <option value="classification">분류</option>
                <option value="regression">회귀</option>
              </select>
            </dd></div>
          )}
          {base.kind === "csv" && numericColumns.length > 0 && (
            <div><dt>시계열</dt><dd>
              {/* 한 열의 시간 순 값을 창으로 자른다. 고르면 과제는 회귀로, 분할은 시간 순으로 못박힌다. */}
              <select className="trainer__select mono" value={String(full.series_column ?? "")}
                      onChange={(event) => set("series_column", event.target.value || null)}>
                <option value="">아님 (행마다 독립)</option>
                {numericColumns.map((column) => (
                  <option key={column} value={column}>{column}로 다음 값 예측</option>
                ))}
              </select>
            </dd></div>
          )}
          {base.kind === "csv" && full.series_column ? (
            <>
              <div><dt>과거 구간</dt><dd>
                <input className="mono field" type="number" min={2} value={Number(full.window ?? 24)}
                       onChange={(event) => set("window", Number(event.target.value))} />
              </dd></div>
              <div><dt>예측 구간</dt><dd>
                <input className="mono field" type="number" min={1} value={Number(full.horizon ?? 1)}
                       onChange={(event) => set("horizon", Number(event.target.value))} />
              </dd></div>
            </>
          ) : null}
          {base.kind === "csv" && !full.series_column && textColumns.length > 0 && (
            <div><dt>입력</dt><dd>
              <select className="trainer__select mono" value={String(full.text_column ?? "")}
                      onChange={(event) => set("text_column", event.target.value || null)}>
                <option value="">표 특징 열</option>
                {textColumns.map((column) => (
                  <option key={column} value={column}>{column} (문장)</option>
                ))}
              </select>
            </dd></div>
          )}
          {base.kind === "csv" && (
            <>
              <div><dt>결측</dt><dd>
                <select className="trainer__select mono" value={String(full.missing)}
                        onChange={(event) => set("missing", event.target.value)}>
                  <option value="drop">행 버리기</option>
                  <option value="mean">평균으로 채우기</option>
                </select>
              </dd></div>
              <div><dt>범주형 원-핫</dt><dd>
                <input type="checkbox" checked={Boolean(full.onehot)}
                       onChange={(event) => set("onehot", event.target.checked)} />
              </dd></div>
              <div><dt>정규화</dt><dd>
                <select className="trainer__select mono" value={String(full.normalize)}
                        onChange={(event) => set("normalize", event.target.value)}>
                  <option value="standard">표준화</option>
                  <option value="minmax">최소최대</option>
                  <option value="none">없음</option>
                </select>
              </dd></div>
            </>
          )}
          {base.kind === "image_folder" && (
            <>
              <div><dt>크기</dt><dd>
                <select className="trainer__select mono" value={String(full.size)}
                        onChange={(event) => set("size", Number(event.target.value))}>
                  {SIZES.map((size) => <option key={size} value={size}>{size} × {size}</option>)}
                </select>
              </dd></div>
              <div><dt>채널</dt><dd>
                <select className="trainer__select mono" value={String(full.channels)}
                        onChange={(event) => set("channels", Number(event.target.value))}>
                  <option value="3">RGB</option>
                  <option value="1">흑백</option>
                </select>
              </dd></div>
              <div><dt>정규화</dt><dd>
                <select className="trainer__select mono" value={String(full.normalize)}
                        onChange={(event) => set("normalize", event.target.value)}>
                  <option value="standard">train 평균·표준편차</option>
                  <option value="fixed">0.5 고정</option>
                </select>
              </dd></div>
            </>
          )}
          {base.kind === "arrays" && (
            <>
              <div><dt>스케일</dt><dd>
                <select className="trainer__select mono" value={String(full.scale)}
                        onChange={(event) => set("scale", event.target.value)}>
                  <option value="auto">uint8이면 /255</option>
                  <option value="none">그대로</option>
                </select>
              </dd></div>
              <div><dt>정규화</dt><dd>
                <select className="trainer__select mono" value={String(full.normalize)}
                        onChange={(event) => set("normalize", event.target.value)}>
                  <option value="none">없음</option>
                  <option value="standard">표준화</option>
                </select>
              </dd></div>
            </>
          )}
          <div><dt>val 비율</dt><dd>
            <input className="mono field" type="number" min={0.05} max={0.5} step={0.05}
                   value={Number(full.val_fraction)}
                   onChange={(event) => set("val_fraction", Number(event.target.value))} />
          </dd></div>
          <div><dt>시드</dt><dd>
            <input className="mono field" type="number" value={Number(full.seed)}
                   onChange={(event) => set("seed", Number(event.target.value))} />
          </dd></div>
          <div><dt>샘플 수 제한</dt><dd>
            <input className="mono field" type="number" min={2} placeholder="전체"
                   value={full.limit ? Number(full.limit) : ""}
                   onChange={(event) => set("limit", event.target.value ? Number(event.target.value) : null)} />
          </dd></div>
        </dl>
        )}

        {base.kind === "csv" && !full.text_column && !full.series_column && (
          <>
            <h3>특징 열</h3>
            <div className="datacard__checks mono">
              {featureNames.map((column) => (
                <label key={column}>
                  <input type="checkbox"
                         checked={((full.features as string[] | null | undefined) ?? featureNames).includes(column)}
                         onChange={(event) => toggleIn("features", featureNames, column, event.target.checked)} />
                  {column}
                </label>
              ))}
            </div>
          </>
        )}

        {task === "regression" && target && (
          <>
            <h3>정답 범위</h3>
            <dl className="rows mono">
              <div><dt>최소</dt><dd>{target.min}</dd></div>
              <div><dt>최대</dt><dd>{target.max}</dd></div>
              <div><dt>평균</dt><dd>{Number(target.mean).toFixed(4)}</dd></div>
              <div><dt>서로 다른 값</dt><dd>{target.distinct}</dd></div>
            </dl>
            <p className="muted">
              학습할 때는 train 분할 통계로 표준화하고, 화면의 오차는 원래 단위로 되돌려 보여 줍니다.
            </p>
          </>
        )}

        {task !== "regression" && classNames.length > 0 && (
          <>
            <h3>클래스</h3>
            <div className="datacard__checks mono">
              {classNames.map((one) => (
                <label key={one}>
                  <input type="checkbox"
                         checked={((full.classes as string[] | null | undefined) ?? classNames).includes(one)}
                         onChange={(event) => toggleIn("classes", classNames, one, event.target.checked)} />
                  {one}
                </label>
              ))}
            </div>
          </>
        )}
      </div>

      <div>
        <h3>미리보기</h3>
        {Object.keys(counts).length > 0 && (
          <div className="datacard__bars mono">
            {Object.entries(counts).map(([one, count]) => (
              <div key={one} className="datacard__bar">
                <span>{one}</span>
                <span className="datacard__fill" style={{ width: `${(count / peak) * 100}%` }} />
                <span className="muted">{count.toLocaleString()}</span>
              </div>
            ))}
          </div>
        )}
        {preview.thumbnails && (
          <div className="datacard__thumbs">
            <div className="mono muted datacard__thumblabels">
              {preview.thumbnails.labels.map((one) => (
                <span key={one} style={{ height: preview.thumbnails!.tile }}>{one}</span>
              ))}
            </div>
            <img src={`data:image/png;base64,${preview.thumbnails.png}`} alt="클래스별 샘플" />
          </div>
        )}
        {preview.header && (
          <div className="datacard__table">
            <table>
              <thead><tr>{preview.header.map((one) => <th key={one}>{one}</th>)}</tr></thead>
              <tbody>
                {(preview.rows ?? []).map((row, index) => (
                  <tr key={index}>{row.map((cell, at) => <td key={at}>{cell || "·"}</td>)}</tr>
                ))}
              </tbody>
            </table>
            <p className="mono muted">
              {columns.map((column) =>
                `${column.name} ${column.kind === "numeric" ? "숫자" : "문자"}`
                + (column.missing ? ` 결측 ${column.missing}` : "")).join(" · ")}
            </p>
          </div>
        )}
        <p className="mono datacard__summary">{summary}</p>
        {spec.problem && <p className="warn mono">{spec.problem}</p>}
      </div>
    </div>
  );
}
