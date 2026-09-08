// 첫 화면. Figma `01 First run`(42:3)을 그대로 옮겼다.
//
// 진입점 다섯: 모델 .py 드롭 / 체크포인트 드롭 / 검증된 템플릿 / Attach 스니펫 /
// 빈 캔버스. 마지막은 시각적으로 가장 조용하게 두고 그 아래 한 줄이 템플릿을
// 먼저 권한다.
//
// 아직 동작하지 않는 진입점은 언제 되는지 밝히고 비활성으로 둔다. 눌러도 아무
// 일이 없는 버튼은 없는 버튼보다 나쁘다.

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchStart, importSource, inspectSource, newGraph, openGraph, openProject } from "../api";
import type { Candidate, RecentGraph, StartInfo, Template } from "../api";

interface Dropped {
  filename: string;
  source: string;
  candidates: Candidate[];
}

export function StartScreen({ onOpened }: { onOpened: () => void }) {
  const [info, setInfo] = useState<StartInfo | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dropped, setDropped] = useState<Dropped | null>(null);
  const [factory, setFactory] = useState("");
  const [example, setExample] = useState("x=B,3,32,32:f32");
  const [hover, setHover] = useState(false);
  const [projectPath, setProjectPath] = useState("");

  // 프로젝트 폴더는 그래프·좌표·생성 코드·run·데이터를 한 곳에 둔다(Figma 00 New project).
  const openFolder = async (dir: string) => {
    if (!dir.trim()) return;
    setBusy(dir);
    setError(null);
    const result = await openProject(dir.trim());
    setBusy(null);
    if (result.error) setError(result.error);
    else onOpened();
  };
  const picker = useRef<HTMLInputElement>(null);

  useEffect(() => { fetchStart().then(setInfo).catch(() => undefined); }, []);

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
            const file = event.dataTransfer.files[0];
            if (file) void accept(file);
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
          <div className="dropband__half dropband__half--off">
            <span className="dropband__title">체크포인트 드롭</span>
            <span className="dropband__sub">.pt / .ckpt / safetensors</span>
            <span className="dropband__hint mono">state_dict에서 구조를 역추정합니다</span>
            <span className="soon">준비 중</span>
          </div>
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
            <h3>프로젝트</h3>
            {info === null ? (
              <p className="mono muted">불러오는 중</p>
            ) : info.projects?.length ? (
              <ul className="templates">
                {info.projects.map((entry) => (
                  <li key={entry.dir}>
                    <button className="templates__row" disabled={busy !== null}
                            onClick={() => void openFolder(entry.dir)} title={entry.dir}>
                      <span className="templates__left">
                        <span className="templates__name">{entry.dir.split("/").filter(Boolean).pop()}</span>
                        <span className="templates__recipe mono">그래프 {entry.name} · {entry.dir}</span>
                      </span>
                      <span className="templates__metric mono">
                        {busy === entry.dir ? "여는 중"
                          : entry.opened ? new Date(entry.opened * 1000).toLocaleDateString("ko-KR",
                              { month: "numeric", day: "numeric" }) : ""}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mono muted">아직 없습니다. 편집 화면의 저장을 누르면 projects/ 아래에 폴더가 생기고 그래프·좌표·코드·run·데이터가 그 안에 모입니다</p>
            )}
            <div className="datafolder">
              <input className="mono" value={projectPath} placeholder="또는 프로젝트 폴더 경로"
                     spellCheck={false} aria-label="프로젝트 폴더 경로"
                     onChange={(event) => setProjectPath(event.target.value)}
                     onKeyDown={(event) => { if (event.key === "Enter") void openFolder(projectPath); }} />
              <button className="ghost" disabled={!projectPath.trim() || busy !== null}
                      onClick={() => void openFolder(projectPath)}>열기</button>
            </div>
          </section>

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
