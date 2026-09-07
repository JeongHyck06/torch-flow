// Input 블록의 데이터 창. 임포트 블록처럼 Input을 더블클릭하면 뜬다.
//
// 폴더를 끌어다 놓거나 골라 data/<이름>/ 에 올리고, 정제 카드를 거쳐 Input 규격과
// experiment.data를 한 번에 맞춘다(POST /api/data). Figma에 이 창은 없다 - `05 Dataset`은
// Experiment 탭 설계라 아직 멀고, 팔레트(`06 Empty canvas`)의 떠 있는 패널 규칙만 따랐다.

import { useEffect, useRef, useState } from "react";

import {
  addDatasetFolder, downloadDataset, fetchDatasets, setGraphData, uploadDatasetFile,
} from "../api";
import type { DatasetInfo, Recipe } from "../api";
import { SYNTHETIC, useStore } from "../store";
import { DataCard } from "./DataCard";

const KIND_LABEL: Record<string, string> = {
  builtin: "내장", image_folder: "이미지 폴더", csv: "CSV 표", arrays: "npy 배열",
};

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

export function DataDialog() {
  const dataset = useStore((state) => state.dataset);
  const recipe = useStore((state) => state.recipe);
  const setData = useStore((state) => state.setData);
  const close = useStore((state) => state.closeData);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  // 이미 붙은 데이터가 있으면 그것부터 보여 준다. 레시피는 적용 전까지 초안이다.
  const [chosen, setChosen] = useState<string | null>(SYNTHETIC.has(dataset) ? null : dataset);
  const [draft, setDraft] = useState<Recipe | null>(SYNTHETIC.has(dataset) ? null : recipe);
  const [folder, setFolder] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState(false);
  const picker = useRef<HTMLInputElement>(null);

  const reload = async () => {
    const found = await fetchDatasets();
    setDatasets(found);
    return found;
  };
  useEffect(() => { void reload().catch(() => undefined); }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") close(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [close]);

  const pick = (name: string) => { setChosen(name); setDraft(null); setError(null); };

  // 폴더(클래스별 하위 폴더)나 파일들을 data/<이름>/ 로 올리고 그 데이터를 고른다.
  const importData = async (entries: { path: string; file: File }[]) => {
    if (!entries.length) return;
    const first = entries[0].path.split("/");
    const name = (first.length > 1 ? first[0] : entries[0].file.name.replace(/\.[^.]+$/, ""))
      .replace(/[^A-Za-z0-9._-]/g, "_") || "dropped";
    setError(null);
    for (const [index, entry] of entries.entries()) {
      setBusy(`${name} 올리는 중 ${index + 1} / ${entries.length}`);
      // 폴더째 놓으면 첫 조각이 폴더 이름이다 - data/<이름>/ 아래 상대 경로만 남긴다.
      const relative = first.length > 1 ? entry.path.split("/").slice(1).join("/") : entry.path;
      const result = await uploadDatasetFile(name, relative, entry.file);
      if (result.error) { setError(result.error); setBusy(null); return; }
    }
    setBusy(null);
    const found = await reload();
    if (found.some((one) => one.name === name)) pick(name);
    else setError(`${name}: 이미지 폴더(클래스별 하위 폴더), CSV, x.npy+y.npy 중 무엇도 아닙니다`);
  };

  const addFolder = async () => {
    const path = folder.trim();
    if (!path) return;
    setBusy("폴더 확인 중");
    setError(null);
    const result = await addDatasetFolder(path);
    setBusy(null);
    if (result.error) { setError(result.error); return; }
    setFolder("");
    await reload();
    pick(result.name);
  };

  // 다운로드는 사람이 누른 버튼에서만 시작한다(§3.1).
  const download = async () => {
    if (!chosen) return;
    setBusy("내려받는 중");
    const result = await downloadDataset(chosen);
    setBusy(null);
    if (result.error) setError(result.error);
    else await reload();
  };

  // hub가 experiment.data를 적고 Input을 set_ports op로 맞춰 브로드캐스트한다.
  const apply = async () => {
    if (!chosen) return;
    setBusy("적용 중");
    const result = await setGraphData(chosen, draft);
    setBusy(null);
    if (result.error) { setError(result.error); return; }
    setData(chosen, result.spec?.recipe ?? draft);
    close();
  };

  const entry = datasets.find((one) => one.name === chosen);
  const needsDownload = entry?.source === "builtin" && !entry.available;

  return (
    <>
      <div className="datadialog__scrim" onClick={close} />
      <div
        className="datadialog" role="dialog" aria-modal="true" aria-label="Input 데이터"
        onDragOver={(event) => { event.preventDefault(); setHover(true); }}
        onDragLeave={() => setHover(false)}
        onDrop={(event) => {
          event.preventDefault();
          setHover(false);
          void walk(Array.from(event.dataTransfer.items ?? [])).then(importData);
        }}
      >
        <div className="datadialog__head">
          <span className="datadialog__title">Input 데이터</span>
          <span className="mono muted">{busy ?? "고른 데이터의 규격으로 Input이 맞춰집니다"}</span>
          <button className="runpanel__close" onClick={close}>닫기</button>
        </div>

        <div className="datadialog__body">
          <div className="datadialog__side">
            <button
              className={`datadialog__drop${hover ? " datadialog__drop--hover" : ""}`}
              onClick={() => picker.current?.click()}
            >
              <span className="dropband__title">데이터 폴더 드롭</span>
              <span className="dropband__sub">클래스별 이미지 폴더 · CSV · x.npy+y.npy</span>
              <span className="dropband__hint mono">끌어다 놓거나 눌러서 폴더를 고르세요</span>
            </button>
            <input
              ref={picker} type="file" hidden multiple
              // @ts-expect-error webkitdirectory는 표준 속성이 아니지만 모든 주요 브라우저가 지원한다.
              webkitdirectory=""
              onChange={(event) => {
                const files = Array.from(event.target.files ?? []);
                void importData(files.map((file) => ({
                  path: (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name,
                  file,
                })));
                event.target.value = "";
              }}
            />
            <div className="datafolder">
              <input
                className="mono" value={folder} spellCheck={false}
                placeholder="또는 폴더 경로"
                aria-label="데이터 폴더 경로"
                onChange={(event) => setFolder(event.target.value)}
                onKeyDown={(event) => { if (event.key === "Enter") void addFolder(); }}
              />
              <button className="ghost" onClick={() => void addFolder()}
                      disabled={busy !== null || !folder.trim()}>
                추가
              </button>
            </div>

            <ul className="templates">
              {datasets.map((one) => (
                <li key={one.name}>
                  <button
                    className={`templates__row${chosen === one.name ? " templates__row--on" : ""}`}
                    onClick={() => pick(one.name)}
                  >
                    <span className="templates__left">
                      <span className="templates__name">{one.label}</span>
                      <span className="templates__recipe mono">
                        {KIND_LABEL[one.kind] ?? one.kind}
                        {one.count ? ` · ${one.count.toLocaleString()}개` : ""}
                        {` · [${one.shape.join(", ")}]`}
                      </span>
                    </span>
                    <span className="templates__metric mono">{one.classes} 클래스</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          <div className="datadialog__main">
            {chosen ? (
              <>
                {needsDownload && (
                  <div className="trainer">
                    <button className="trainer__run" onClick={() => void download()}
                            disabled={busy !== null}>
                      내려받기{entry?.size_mb ? ` ${entry.size_mb} MB` : ""}
                    </button>
                    <span className="mono muted">네트워크는 이 버튼을 눌러야만 탑니다</span>
                  </div>
                )}
                <DataCard name={chosen} recipe={draft} onRecipe={setDraft} />
              </>
            ) : (
              <p className="mono muted">왼쪽에서 데이터를 고르거나 폴더를 끌어다 놓으세요</p>
            )}
          </div>
        </div>

        {error && <p className="mono warn datadialog__error">{error}</p>}

        <div className="datadialog__foot">
          <button className="ghost" onClick={close}>취소</button>
          <button className="solid" onClick={() => void apply()} disabled={!chosen || busy !== null}>
            Input에 적용
          </button>
        </div>
      </div>
    </>
  );
}
