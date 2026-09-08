// 모델 테스트 탭. 학습이 끝난 run의 체크포인트를 학습이 보지 않은 분할에 돌려 본다.
//
// 정확도 숫자만으로는 무엇을 틀리는지 모른다 - 클래스별 정확도, 혼동 행렬, 그리고 틀린
// 샘플부터 보여 준다. 실행은 hub가 워커 --test 프로세스로 돌리고 끝날 때까지 기다린다.
// Figma에 이 탭은 없다 - Run 패널의 다른 탭(데이터 카드 두 열, 블록 상세 격자)을 따랐다.

import { useEffect, useState } from "react";

import { fetchTest, runTest } from "../api";
import type { TestResult } from "../api";
import { FINISHED, RUNNING } from "../stages";
import { useStore } from "../store";

export function TestPanel() {
  const runs = useStore((state) => state.runs);
  const [picked, setPicked] = useState<string | null>(null);
  const [result, setResult] = useState<TestResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // 기본은 끝난 run 중 마지막. 돌고 있는 run은 체크포인트가 아직 없다.
  const fallback = [...runs].reverse().find((run) => !RUNNING.has(run.state)) ?? runs[runs.length - 1];
  const runId = picked ?? fallback?.run_id ?? "";

  const start = async () => {
    setBusy(true);
    setError(null);
    const body = await runTest(runId)
      .catch((reason) => ({ ok: false, error: String(reason) } as TestResult));
    setBusy(false);
    if (body.ok) setResult(body);
    else setError(body.error ?? "테스트에 실패했습니다");
  };

  useEffect(() => {
    if (!runId) { setResult(null); return; }
    let cancelled = false;
    fetchTest(runId).then((body) => {
      if (cancelled) return;
      setResult(body);
      setError(null);
      // 끝난 run에 결과가 없으면 바로 돌린다 - 탭을 연 것이 곧 "재 보자"다. 몇 초면 끝난다.
      const run = useStore.getState().runs.find((one) => one.run_id === runId);
      if (!body && run && FINISHED.has(run.state)) void start();
    }).catch(() => undefined);
    return () => { cancelled = true; };
    // start는 runId만 닫아 둔다 - runId가 바뀔 때만 다시 돌면 된다.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  if (!runs.length) {
    return <p className="mono muted">학습을 한 번 돌리면 그 체크포인트를 여기서 테스트할 수 있습니다</p>;
  }

  return (
    <>
      <div className="trainer">
        <button className="trainer__run" onClick={() => void start()} disabled={busy || !runId}>
          {busy ? "테스트 중" : "테스트 실행"}
        </button>
        <label className="trainer__field">run
          <select className="trainer__select mono" value={runId}
                  onChange={(event) => setPicked(event.target.value)}>
            {runs.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id} · {run.state} · step {run.step.toLocaleString()}
              </option>
            ))}
          </select>
        </label>
        {result?.ok && (
          <span className="mono trainer__progress">
            정확도 {((result.acc ?? 0) * 100).toFixed(1)} % · loss {(result.loss ?? 0).toFixed(4)}
            {" · "}{result.split} {(result.count ?? 0).toLocaleString()}개
            {" · "}step {(result.step ?? 0).toLocaleString()}{result.device ? ` · ${result.device}` : ""}
          </span>
        )}
        <span className="muted trainer__note">
          학습이 보지 않은 데이터로 잽니다 · 끝난 run은 탭을 열면 바로 돌아갑니다 · 다시 누르면 다시 잽니다
        </span>
        {error && <span className="warn mono">{error}</span>}
      </div>
      {result?.ok ? <Report result={result} /> : (
        !busy && !error && <p className="mono muted">테스트 실행을 누르면 결과가 여기에 보입니다</p>
      )}
    </>
  );
}

function Report({ result }: { result: TestResult }) {
  const classes = result.classes ?? [];
  const perClass = result.per_class ?? [];
  const confusion = result.confusion ?? [];
  const samples = result.samples ?? [];
  return (
    <div className="testpanel">
      <div>
        <h3>클래스별 정확도</h3>
        <div className="datacard__bars mono">
          {perClass.map((one) => (
            <div key={one.name} className="datacard__bar">
              <span>{one.name}</span>
              <span className="datacard__fill"
                    style={{ width: `${one.count ? (one.correct / one.count) * 100 : 0}%` }} />
              <span className="muted">{one.correct}/{one.count}</span>
            </div>
          ))}
        </div>
        {classes.length <= 20 && (
          <>
            <h3>혼동 행렬<span className="muted">행 정답 · 열 예측</span></h3>
            <div className="datacard__table">
              <table className="testpanel__matrix">
                <thead>
                  <tr><th />{classes.map((one) => <th key={one}>{one}</th>)}</tr>
                </thead>
                <tbody>
                  {confusion.map((row, truth) => (
                    <tr key={truth}>
                      <th>{classes[truth]}</th>
                      {row.map((count, guess) => (
                        <td key={guess}
                            className={truth === guess ? "testpanel__hit" : count ? "testpanel__miss" : "muted"}>
                          {count || "·"}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
      <div>
        <h3>샘플<span className="muted">틀린 것은 빨간 테두리, 아래는 예측</span></h3>
        {samples.length ? (
          <div className="testpanel__tiles">
            {samples.map((one, index) => (
              <figure key={index}
                      className={`testpanel__tile${one.pred === one.label ? "" : " testpanel__tile--wrong"}`}
                      title={`예측 ${classes[one.pred]} (${(one.prob * 100).toFixed(0)} %) · 정답 ${classes[one.label]}`}>
                <img src={`data:image/png;base64,${one.png}`}
                     alt={`정답 ${classes[one.label]}, 예측 ${classes[one.pred]}`} />
                <figcaption className="mono">
                  {classes[one.pred]}
                  {one.pred !== one.label && <span className="testpanel__truth">정답 {classes[one.label]}</span>}
                </figcaption>
              </figure>
            ))}
          </div>
        ) : <p className="mono muted">이미지 입력이 아니라 샘플 그림은 없습니다</p>}
      </div>
    </div>
  );
}
