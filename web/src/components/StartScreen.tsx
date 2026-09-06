// 첫 화면 - Figma `01 First run`(42:3).
//
// §2.2: 진입점 다섯 개(템플릿 / .py 드롭 / 체크포인트 드롭 / Attach 스니펫 /
// 빈 캔버스). 마지막 항목은 시각적으로 가장 조용하게 두고 그 아래에 템플릿을
// 먼저 권하는 한 줄을 둔다.
//
// 아직 동작하지 않는 진입점은 비활성으로 그린다. 눌러도 아무 일이 없는 버튼은
// 없는 버튼보다 나쁘다.

import { useEffect, useState } from "react";
import { fetchStart, openGraph } from "../api";
import type { StartInfo, Template } from "../api";

const ATTACH_SNIPPET = `import torchflow as tf

tf.watch(model, example_input=x)
tf.log(step, loss=loss.item())`;

export function StartScreen({ onOpened }: { onOpened: () => void }) {
  const [info, setInfo] = useState<StartInfo | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { fetchStart().then(setInfo).catch(() => undefined); }, []);

  const open = async (template: Template) => {
    setBusy(template.id);
    setError(null);
    try {
      const result = await openGraph(template.path);
      if (result.error) setError(result.error);
      else onOpened();
    } finally {
      setBusy(null);
    }
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
              {device.name}
              {device.note ? <span className="muted"> torch {device.note}</span> : null}
            </span>
          ))}
        </div>
      </header>

      <div className="start__inner">
        <h1>무엇으로 시작할까요</h1>
        <p className="start__lede">
          기존 코드를 가져오거나, 검증된 템플릿에서 출발하거나, 돌아가는 학습
          스크립트에 붙일 수 있습니다.
        </p>

        <div className="dropband">
          <section className="dropband__half" aria-disabled>
            <h2>모델 .py 드롭</h2>
            <p>AST로 읽어 그래프로 펼칩니다</p>
            <p className="mono muted">--factory "DiT(depth=28, ...)" 폼이 이어서 뜹니다</p>
            <span className="soon">M8</span>
          </section>
          <section className="dropband__half" aria-disabled>
            <h2>체크포인트 드롭</h2>
            <p>.pt / .ckpt / safetensors</p>
            <p className="mono muted">state_dict에서 구조를 역추정합니다</p>
            <span className="soon">v1</span>
          </section>
        </div>

        <div className="start__cols">
          <section>
            <h3>검증된 템플릿</h3>
            <ul className="templates">
              {(info?.templates ?? []).map((template) => (
                <li key={template.id}>
                  <button
                    className="templates__row"
                    disabled={!template.available || busy !== null}
                    onClick={() => open(template)}
                  >
                    <span className="templates__name">{template.name}</span>
                    <span className="templates__metric">
                      {busy === template.id ? "여는 중…"
                        : template.available ? template.metric : "아직 없음"}
                    </span>
                    <span className="templates__recipe mono">{template.recipe}</span>
                  </button>
                </li>
              ))}
            </ul>
            {error && <p className="mono start__error">{error}</p>}
          </section>

          <section>
            <h3>돌아가는 스크립트에 붙이기</h3>
            <pre className="snippet">{ATTACH_SNIPPET}</pre>
            <p className="start__note">
              코드가 그대로 정본입니다. 편집 UI 없이 shape 오버레이 · grad-flow ·
              run 기록만 얹습니다.
            </p>
            <p className="mono muted">원격 GPU: ssh -L 8765:127.0.0.1:8765 gpu-box</p>
          </section>
        </div>

        {/* 가장 조용한 진입점. 아래 한 줄이 템플릿을 먼저 권한다(§2.2). */}
        <section className="start__blank">
          <div>
            <h3>빈 그래프에서 설계하기</h3>
            <p className="start__note">
              아무것도 없이 시작합니다. 블록을 놓으면 포트 타입이 연결 가능한 것만
              제안하고, 필수 포트가 비면 그 자리에서 알려줍니다.
            </p>
            <p className="muted start__hint">
              처음이라면 위의 검증된 템플릿을 열어 뜯어보는 쪽이 빠릅니다
            </p>
          </div>
          <button className="ghost" disabled title="IR 편집은 M5">
            빈 캔버스 열기
          </button>
        </section>

        <footer className="start__footer mono muted">
          {location.host} · state-dir {info?.state_dir ?? "…"}
          {info?.torch_version ? ` · torch ${info.torch_version}` : ""}
          {info?.devices?.length ? ` · ${info.devices[0].name}` : ""}
        </footer>
      </div>
    </div>
  );
}
