// 첫 화면. Figma `01 First run`(42:3)을 그대로 옮겼다.
//
// 진입점 다섯: 모델 .py 드롭 / 체크포인트 드롭 / 검증된 템플릿 / Attach 스니펫 /
// 빈 캔버스. 마지막은 시각적으로 가장 조용하게 두고 그 아래 한 줄이 템플릿을
// 먼저 권한다.
//
// 아직 동작하지 않는 진입점은 언제 되는지 밝히고 비활성으로 둔다. 눌러도 아무
// 일이 없는 버튼은 없는 버튼보다 나쁘다.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  addDatasetFolder, fetchDatasets, fetchStart, importSource, inspectSource, newGraph, openGraph,
  uploadDatasetFile,
} from "../api";
import type { Candidate, DatasetInfo, Recipe, RecentGraph, StartInfo, Template } from "../api";
import { DataCard } from "./DataCard";

const KIND_LABEL: Record<string, string> = {
  builtin: "내장", image_folder: "이미지 폴더", csv: "CSV 표", arrays: "npy 배열",
};

interface Dropped {
  filename: string;
  source: string;
  candidates: Candidate[];
}

/** 드롭된 항목을 파일 목록으로 편다. 폴더는 재귀로 들어간다(webkitGetAsEntry). */
async function walk(items: DataTransferItem[]): Promise<{ path: string; file: File }[]> {
  const out: { path: string; file: File }[] = [];
  const visit = async (entry: FileSystemEntry, prefix: string): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) =>
        (entry as FileSystemFileEntry).file(resolve, reject));
      if (!file.name.startsWith(".")) out.push({ path: prefix + file.name, file });
      return;
    }
    if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      const read = (): Promise<FileSystemEntry[]> =>
        new Promise((resolve, reject) => reader.readEntries(resolve, reject));
      for (let batch = await read(); batch.length; batch = await read()) {
        for (const child of batch) await visit(child, `${prefix}${entry.name}/`);
      }
    }
  };
  for (const item of items) {
    const entry = item.webkitGetAsEntry?.();
    if (entry) await visit(entry, "");
  }
  return out;
}

export function StartScreen({ onOpened }: { onOpened: () => void }) {
  const [info, setInfo] = useState<StartInfo | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dropped, setDropped] = useState<Dropped | null>(null);
  const [factory, setFactory] = useState("");
  const [example, setExample] = useState("x=B,3,32,32:f32");
  const [hover, setHover] = useState(false);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [folder, setFolder] = useState("");
  const [card, setCard] = useState<DatasetInfo | null>(null);
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  const picker = useRef<HTMLInputElement>(null);
  const folderPicker = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState<string | null>(null);

  useEffect(() => {
    fetchStart().then(setInfo).catch(() => undefined);
    fetchDatasets().then(setDatasets).catch(() => undefined);
  }, []);

  // 데이터부터 시작: Input/Output이 그 데이터 규격으로 깔린 새 그래프.
  const startFromDataset = async (entry: DatasetInfo) => {
    setBusy(entry.name);
    setError(null);
    const result = await newGraph(entry.label, entry.name, recipe);
    setBusy(null);
    if (result.error) setError(result.error);
    else onOpened();
  };

  // 데이터 불러오기: 폴더(클래스별 하위 폴더)나 파일들을 data/<이름>/ 로 올리고 카드를 연다.
  const importData = async (entries: { path: string; file: File }[]) => {
    if (!entries.length) return;
    const first = entries[0].path.split("/");
    const name = (first.length > 1 ? first[0] : entries[0].file.name.replace(/\.[^.]+$/, ""))
      .replace(/[^A-Za-z0-9._-]/g, "_") || "dropped";
    setBusy("upload");
    setError(null);
    for (const [index, entry] of entries.entries()) {
      setProgress(`${name} 올리는 중 ${index + 1} / ${entries.length}`);
      // 폴더째 놓으면 첫 조각이 폴더 이름이다 - data/<이름>/ 아래 상대 경로만 남긴다.
      const relative = first.length > 1 ? entry.path.split("/").slice(1).join("/") : entry.path;
      const result = await uploadDatasetFile(name, relative, entry.file);
      if (result.error) { setError(result.error); setBusy(null); setProgress(null); return; }
    }
    setProgress(null);
    setBusy(null);
    const found = await fetchDatasets();
    setDatasets(found);
    const entry = found.find((one) => one.name === name);
    if (entry) { setCard(entry); setRecipe(null); }
    else setError(`${name}: 이미지 폴더(클래스별 하위 폴더), CSV, x.npy+y.npy 중 무엇도 아닙니다`);
  };

  const addFolder = async () => {
    const path = folder.trim();
    if (!path) return;
    setBusy("folder");
    setError(null);
    const result = await addDatasetFolder(path);
    setBusy(null);
    if (result.error) { setError(result.error); return; }
    setFolder("");
    setDatasets(await fetchDatasets());
  };

  const startEmpty = async () => {
    setBusy("empty");
    setError(null);
    const result = await newGraph();
    setBusy(null);
    if (result.error) setError(result.error);
    else onOpened();
  };

  const open = async (target: Template | RecentGraph, id?: string) => {
    setBusy(id ?? target.path);
    setError(null);
    const result = await openGraph(target.path);
    setBusy(null);
    if (result.error) setError(result.error);
    else onOpened();
  };

  const accept = useCallback(async (file: File) => {
    setError(null);
    if (!file.name.endsWith(".py")) {
      setError(`.py 파일이 필요합니다 (${file.name})`);
      return;
    }
    const source = await file.text();
    const found = await inspectSource(source);
    if (found.error) {
      setError(found.error);
      return;
    }
    if (!found.candidates?.length) {
      setError("nn.Module을 상속한 클래스를 찾지 못했습니다");
      return;
    }
    // 가장 마지막에 정의된 클래스가 보통 최상위 모델이다.
    const last = found.candidates[found.candidates.length - 1];
    setDropped({ filename: file.name, source, candidates: found.candidates });
    setFactory(last.suggestion);
  }, []);

  const runImport = async () => {
    if (!dropped) return;
    setBusy("import");
    setError(null);
    const result = await importSource({
      filename: dropped.filename, source: dropped.source, factory, example,
    });
    setBusy(null);
    if (result.error) setError(`${result.stage ?? "import"}: ${result.error}`);
    else onOpened();
  };

  return (
    <div className="start">
      <header className="topbar topbar--start">
        <span className="wordmark">
          <span className="wordmark__dot" aria-hidden />
          torchflow
          <span className="wordmark__version">0.0.1</span>
        </span>
        <div className="devices">
          {(info?.devices ?? []).map((device) => (
            <span key={device.name} className="devices__chip">
              <span className="dot dot--on" />
              {device.label}
            </span>
          ))}
        </div>
      </header>

      <div className="start__content">
        <div className="start__heading">
          <h1>무엇으로 시작할까요</h1>
          <p>기존 코드를 가져오거나, 검증된 템플릿에서 출발하거나, 돌아가는 학습
            스크립트에 붙일 수 있습니다.</p>
        </div>

        <div
          className={`dropband${hover ? " dropband--hover" : ""}`}
          onDragOver={(event) => { event.preventDefault(); setHover(true); }}
          onDragLeave={() => setHover(false)}
          onDrop={(event) => {
            event.preventDefault();
            setHover(false);
            const items = Array.from(event.dataTransfer.items ?? []);
            const entry = items[0]?.webkitGetAsEntry?.();
            const file = event.dataTransfer.files[0];
            // 폴더거나 데이터 파일이면 데이터 불러오기, .py면 모델 import.
            if (entry?.isDirectory || (file && /\.(csv|npy|npz)$/i.test(file.name))) {
              void walk(items).then(importData);
            } else if (file) {
              void accept(file);
            }
          }}
        >
          <button className="dropband__half" onClick={() => picker.current?.click()}>
            <span className="dropband__title">모델 .py 드롭</span>
            <span className="dropband__sub">인스턴스로 만들어 실측 트레이스합니다</span>
            <span className="dropband__hint mono">
              {dropped ? dropped.filename : "끌어다 놓거나 눌러서 고르세요"}
            </span>
          </button>
          <div className="dropband__divider" />
          <button className="dropband__half" onClick={() => folderPicker.current?.click()}>
            <span className="dropband__title">데이터 폴더 드롭</span>
            <span className="dropband__sub">클래스별 이미지 폴더 · CSV · x.npy+y.npy</span>
            <span className="dropband__hint mono">
              {progress ?? "끌어다 놓거나 눌러서 폴더를 고르세요"}
            </span>
          </button>
          <div className="dropband__divider" />
          <div className="dropband__half dropband__half--off">
            <span className="dropband__title">체크포인트 드롭</span>
            <span className="dropband__sub">.pt / .ckpt / safetensors</span>
            <span className="dropband__hint mono">state_dict에서 구조를 역추정합니다</span>
            <span className="soon">v1</span>
          </div>
          <input
            ref={folderPicker} type="file" hidden multiple
            // @ts-expect-error webkitdirectory는 표준 속성이 아니지만 모든 주요 브라우저가 지원한다.
            webkitdirectory=""
            onChange={(event) => {
              const files = Array.from(event.target.files ?? []);
              void importData(files.map((file) => ({
                path: (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name, file,
              })));
              event.target.value = "";
            }}
          />
          <input
            ref={picker} type="file" accept=".py" hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void accept(file);
              event.target.value = "";
            }}
          />
        </div>

        {dropped && (
          <div className="importform">
            <label>
              <span>인스턴스</span>
              <input
                className="mono" value={factory}
                onChange={(event) => setFactory(event.target.value)}
                spellCheck={false} autoFocus
              />
            </label>
            <label>
              <span>예시 입력</span>
              <input
                className="mono" value={example}
                onChange={(event) => setExample(event.target.value)}
                spellCheck={false}
              />
            </label>
            <div className="importform__actions">
              <span className="mono muted">
                후보 {dropped.candidates.map((item) => item.name).join(", ")}
              </span>
              <button className="ghost" onClick={() => setDropped(null)}>취소</button>
              <button className="solid" onClick={runImport} disabled={busy === "import"}>
                {busy === "import" ? "여는 중" : "그래프로 열기"}
              </button>
            </div>
          </div>
        )}

        {error && <p className="mono start__error">{error}</p>}

        <div className="start__cols">
          <section>
            <h3>검증된 템플릿</h3>
            <ul className="templates">
              {(info?.templates ?? []).map((template) => (
                <li key={template.id}>
                  <button
                    className="templates__row"
                    disabled={!template.available || busy !== null}
                    onClick={() => open(template, template.id)}
                  >
                    <span className="templates__left">
                      <span className="templates__name">{template.name}</span>
                      <span className="templates__recipe mono">{template.recipe}</span>
                    </span>
                    <span className="templates__metric mono">
                      {busy === template.id ? "여는 중"
                        : template.available ? template.metric : "아직 없음"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>

          <section>
            <h3>내 그래프</h3>
            {info?.recent?.length ? (
              <ul className="templates">
                {info.recent.map((entry) => (
                  <li key={entry.path}>
                    <button className="templates__row" disabled={busy !== null}
                            onClick={() => open(entry)} title={entry.path}>
                      <span className="templates__left">
                        <span className="templates__name">{entry.name}</span>
                        <span className="templates__recipe mono">{entry.file}</span>
                      </span>
                      <span className="templates__metric mono">
                        {busy === entry.path ? "여는 중"
                          : new Date(entry.modified * 1000).toLocaleDateString("ko-KR",
                              { month: "numeric", day: "numeric" })}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mono muted">아직 저장한 그래프가 없습니다. 저장하면 graph/ 에 쌓입니다</p>
            )}
          </section>

          <section>
            <h3>내 데이터로 시작</h3>
            <ul className="templates">
              {datasets.map((entry) => (
                <li key={entry.name}>
                  <button className={`templates__row${card?.name === entry.name ? " templates__row--on" : ""}`}
                          disabled={busy !== null}
                          onClick={() => { setCard(entry); setRecipe(null); }}
                          title="정제 설정을 보고 이 데이터로 새 그래프를 엽니다">
                    <span className="templates__left">
                      <span className="templates__name">{entry.label}</span>
                      <span className="templates__recipe mono">
                        {KIND_LABEL[entry.kind] ?? entry.kind}
                        {entry.count ? ` · ${entry.count.toLocaleString()}개` : ""}
                        {` · [${entry.shape.join(", ")}]`}
                        {entry.source === "builtin" && !entry.available ? " · 내려받기는 Run 패널에서" : ""}
                      </span>
                    </span>
                    <span className="templates__metric mono">
                      {busy === entry.name ? "여는 중" : `${entry.classes} 클래스`}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            <div className="datafolder">
              <input
                className="mono" value={folder} spellCheck={false}
                placeholder="폴더 경로 (클래스별 하위 폴더 · CSV · x.npy+y.npy)"
                aria-label="데이터 폴더 경로"
                onChange={(event) => setFolder(event.target.value)}
                onKeyDown={(event) => { if (event.key === "Enter") void addFolder(); }}
              />
              <button className="ghost" onClick={() => void addFolder()}
                      disabled={busy !== null || !folder.trim()}>
                {busy === "folder" ? "확인 중" : "추가"}
              </button>
            </div>
            <p className="mono muted scratch__hint">data/ 아래 폴더는 자동으로 뜹니다. 학습은 8:2로 나눠 검증합니다</p>
          </section>

          <section>
            <h3>돌아가는 스크립트에 붙이기</h3>
            <div className="snippet">
              <span className="snippet__dim">import torchflow as tf</span>
              <span />
              <span>tf.watch(model, example_input=x)</span>
              <span>tf.log(step, loss=loss.item())</span>
            </div>
            <div className="snippet__notes">
              <p>코드가 그대로 정본입니다. 편집 UI 없이 shape 오버레이, grad-flow,
                run 기록만 얻습니다.</p>
              <p className="mono">원격 GPU: ssh -L 8765:127.0.0.1:8765 gpu-box</p>
            </div>
          </section>
        </div>

        {card && (
          <section className="datacard-wrap">
            <div className="scratch__row">
              <div>
                <span className="scratch__title">{card.label}</span>
                <p>자동으로 알아본 것 위에 정제 설정을 얹습니다. 설정은 그래프에 남아 run과 함께 재현됩니다.</p>
              </div>
              <button className="ghost" onClick={() => setCard(null)}>취소</button>
              <button className="solid" onClick={() => void startFromDataset(card)} disabled={busy !== null}>
                {busy === card.name ? "여는 중" : "이 설정으로 그래프 시작"}
              </button>
            </div>
            <DataCard name={card.name} recipe={recipe} onRecipe={setRecipe} />
          </section>
        )}

        <section className="scratch">
          <div className="scratch__row">
            <div>
              <span className="scratch__title">빈 그래프에서 설계하기</span>
              <p>아무것도 없이 시작합니다. 블록을 놓으면 포트 타입이 연결 가능한
                것만 제안하고, 필수 포트가 비면 그 자리에서 알려줍니다.</p>
            </div>
            <button className="ghost" onClick={startEmpty} disabled={busy !== null}
                    title="Tab 으로 팔레트를 열어 첫 블록을 놓습니다">
              {busy === "empty" ? "여는 중" : "빈 캔버스 열기"}
            </button>
          </div>
          <p className="mono muted scratch__hint">
            처음이라면 위의 검증된 템플릿을 열어 뜯어보는 쪽이 빠릅니다
          </p>
        </section>
      </div>

      <footer className="start__footer mono">
        <span>{location.host}</span>
        <span>state-dir {info?.state_dir ?? "…"}</span>
        <span>torch {info?.torch_version ?? "…"}</span>
      </footer>
    </div>
  );
}
