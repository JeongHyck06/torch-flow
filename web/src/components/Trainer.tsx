// 지금 run의 상태와 조작 (기획서 §5.7, §13.1 M7). 하단 패널 곡선 탭의 머리에 붙는다.
//
// **시작은 여기서 하지 않는다** - 단계 표시줄의 4 학습이 유일한 시작 버튼이고(train.ts startRun),
// 여기는 돌고 있는 run의 Pause/Stop/lr(HOT, §5.7.1)과 멈춘 run의 재개만 둔다. 재시작이 필요한
// 변경은 **알리기만** 한다(ADR-05). run 종류 표기는 Figma `04 Runs`(48:147)를 따른다:
// exploratory는 †로 표시하고 집계에서 기본 제외, reported는 hparam 동결이라 편집하면 run이 갈라진다.

import { useEffect, useState } from "react";

import { controlTraining, fetchDatasets } from "../api";
import type { DatasetInfo } from "../api";
import { applyEdit } from "../edit";
import { op } from "../graph/ops";
import { RUNNING, runLabel } from "../stages";
import { useStore } from "../store";
import { startRun } from "../train";

export function Trainer({ onChange }: { onChange: () => void }) {
  // run 목록은 App이 2초마다 채운다 - 패널이 닫혀 있어도 단계 표시줄이 같은 값을 본다.
  const runs = useStore((state) => state.runs);
  const dataset = useStore((state) => state.dataset);
  // 시작 요청의 오류는 단계 표시줄이 만들고 여기서 보여 준다 - 고칠 버튼이 여기 있으므로.
  const startError = useStore((state) => state.startError);
  const startFix = useStore((state) => state.startFix);
  const setStartResult = useStore((state) => state.setStartResult);
  const openRunPanel = useStore((state) => state.openRunPanel);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [lr, setLr] = useState(0.001);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const active = runs.find((run) => RUNNING.has(run.state) || run.state === "paused");
  const last = runs[runs.length - 1];
  const device = useStore((state) => state.trainingDevice);
  const devices = useStore((state) => state.devices);
  const [retrying, setRetrying] = useState(false);
  const finalizing = active?.state === "finalizing" || Boolean(active && active.total > 0 && active.step >= active.total);
  useEffect(() => { setError(null); setNote(null); }, [last?.run_id]);
  // "이미 학습이 돌고 있습니다"는 그 run이 끝나면 더 이상 참이 아니다. 사람이 다시 누를
  // 때까지 남아 있으면 새 run이 도는 중에도 옛 거절 문구가 붙어 있다.
  const startErrorRun = useStore((state) => state.startErrorRun);
  useEffect(() => {
    if (!startErrorRun) return;
    const blocking = runs.find((run) => run.run_id === startErrorRun);
    if (!blocking || !(RUNNING.has(blocking.state) || blocking.state === "paused")) {
      setStartResult(null, null);
    }
  }, [runs, startErrorRun, setStartResult]);
  const retryCPU = async () => {
    if (!last || retrying) return;
    setRetrying(true);
    try {
      const result = await startRun({ retry_run: last.run_id, device: "cpu" });
      // 연타 가드에 걸려 되돌아온 경우까지 말해 준다 - 조용하면 버튼이 먹통으로 보인다.
      if (result.error) setError(result.error);
      else useStore.setState({ trainingDevice: "cpu" });
    } finally { setRetrying(false); }
  };
  const chosen = datasets.find((entry) => entry.name === dataset);

  useEffect(() => { fetchDatasets().then(setDatasets).catch(() => undefined); }, []);

  // 돌고 있는 동안은 곡선을 run 목록과 같은 박자로 다시 읽는다.
  useEffect(() => {
    if (runs.some((run) => RUNNING.has(run.state))) onChange();
  }, [runs, onChange]);

  const applyFix = async () => {
    if (!startFix) return;
    const failed = await applyEdit(op("set_ports", { node: startFix.node, ports_out: startFix.ports_out }));
    if (failed) { setError(failed); return; }
    setStartResult(null, null);
    onChange();
  };

  const send = async (cmd: string, extra: Record<string, unknown> = {}) => {
    const target = active ?? last;
    if (!target) return;
    setError(null);
    setNote(null);
    const result = await controlTraining(target.run_id, cmd, extra);
    // 재시작이 필요한 변경은 적용되지 않는다 - 무엇을 해야 하는지만 말한다.
    if (result.restart_required) setError(result.message ?? "재시작이 필요한 변경입니다");
    // reported run을 건드리면 원 run은 멈추고 갈라진 run이 이어 간다(§5.7.2).
    if (result.forked_from) setNote(`run이 갈라졌습니다 · ${result.run_id}`);
    onChange();
  };

  const reported = active?.kind === "reported";
  // 스케줄러가 돌면 사람이 고치는 lr은 base다. 실제 lr은 스케줄이 정하므로
  // 둘을 같이 보여 주지 않으면 "입력한 값과 다른 값이 돈다"로 읽힌다.
  const scheduled = Boolean(active?.scheduler);
  const factor = active?.base_lr && active?.lr ? active.lr / active.base_lr : null;

  return (
    <div className="trainer">
      <label className="trainer__field">학습 장치
        <select aria-label="학습 장치" value={device} disabled={Boolean(active) || retrying}
                onChange={(event) => useStore.setState({ trainingDevice: event.target.value })}>
          <option value="auto">자동 선택</option>
          <option value="cpu">CPU</option>
          {/* 커널이 실제로 본 가속기만 고를 수 있게 한다 - 없는 장치를 골라 워커가 죽는 일이 없다. */}
          {devices.map((one) => (
            <option key={one.name} value={one.name}>{one.label}</option>
          ))}
        </select>
      </label>
      {!active ? (
        <>
          <span className="mono trainer__progress">
            {last ? runLabel([last]) : "아직 학습 기록이 없습니다"}
          </span>
          {last && last.state === "stopped" && (
            <button className="trainer__stop" onClick={() => void send("resume")}
                    title="체크포인트에서 이어 돕니다">
              재개
            </button>
          )}
          <button className="trainer__stop"
                  onClick={() => void startRun({}, true).then(() => openRunPanel("curves"))}
                  title="20 step · batch 8 · 결정적 - 학습이 되긴 하는지 30초 안에 봅니다">
            빠른 점검
          </button>
          <span className="muted trainer__note">
            {chosen
              ? `${chosen.label}${chosen.count ? ` ${chosen.count.toLocaleString()}개` : ""}`
                + ` · Input 규격 [B, ${chosen.shape.join(", ")}] · 출력 `
                + (chosen.classes ? `${chosen.classes} 클래스` : "값 하나 (회귀)")
                + (chosen.source === "user" ? " · 8:2로 나눠 검증" : " · 검증은 test 분할")
              : "합성 과제 · 무작위 입력에 고정 teacher 라벨"}
            {" · 시작은 단계 표시줄의 4 학습"}
          </span>
        </>
      ) : (
        <>
          <button className="trainer__run" disabled={finalizing || active.state === "starting"} onClick={() => void send(
            active.state === "paused" ? "resume" : "pause")}>
            {active.state === "paused" ? "이어 하기" : "일시정지"}
          </button>
          <button className="trainer__stop" disabled={finalizing} onClick={() => void send("stop")}>중지</button>
          <span className="mono trainer__progress">
            {runLabel([active])}
            {active.device ? ` · ${active.device}` : ""}
            {active.smoke ? " · smoke" : ""}
            {active.state === "paused" ? " · 일시정지" : ""}
            {active.nan_step !== undefined ? ` · NaN @ ${active.nan_step}` : ""}
          </span>
          {reported ? (
            <span className="mono trainer__progress" title="학습 설정을 고정한 결과입니다">보고용 결과</span>
          ) : (
            <button className="trainer__stop" onClick={() => void send("promote")}
                    title={"†는 탐색용 실행이라는 표시로 집계에서 빠집니다. 누르면 보고용으로 바꿔"
                           + " 설정을 고정하고, 이후 편집은 별도 실행으로 기록합니다"}>
              탐색용 실행 †
            </button>
          )}
          <label className="trainer__field">{scheduled ? "base lr" : "lr"}
            {/* 제어 입력이어야 한다. defaultValue는 마운트 때만 읽히는데 React가
                입력 DOM을 재사용해서 다른 값이 남아 있었다. */}
            <input type="number" step={0.0001} min={0} value={Number(lr.toPrecision(6))}
                   onChange={(event) => setLr(Number(event.target.value))}
                   onKeyDown={(event) => {
                     if (event.key !== "Enter") return;
                     void send("set_hparam", { path: "optim.lr", value: lr });
                   }} />
          </label>
          {scheduled && factor !== null && (
            <span className="mono trainer__progress">
              현재 lr {active.lr!.toExponential(2)} = base {active.base_lr!.toExponential(2)}
              {" × "}{factor.toFixed(3)}
            </span>
          )}
          <span className="muted trainer__note">
            {reported ? "lr을 바꾸면 run이 갈라집니다"
              : scheduled ? "Enter로 base를 옮깁니다. 스케줄 모양은 그대로입니다"
              : "lr은 Enter로 학습 중에 바뀝니다"}
          </span>
        </>
      )}
      {note && <span className="mono trainer__note">{note}</span>}
      {(error ?? startError) && <span className="warn mono">{error ?? startError}</span>}
      {startFix && (
        <button className="trainer__stop" onClick={() => void applyFix()}>
          Input을 [{startFix.ports_out[0].shape.join(", ")}]로 맞추기
        </button>
      )}
      {!active && last?.state === "failed" && (
        <section role="alert" className="training-failure">
          <strong>학습에 실패했습니다</strong>
          <p>{last.cpu_retry ? "Apple GPU가 이 연산을 지원하지 않습니다. CPU로 다시 실행할 수 있습니다."
            : "학습을 완료하지 못했습니다. 아래 오류와 학습 출력을 확인해 설정을 수정해 주세요."}</p>
          <span className="muted">실행: {last.run_id}</span>
          {last.cpu_retry && (
            <>
              <p>실패 당시 모델·데이터·학습 설정으로 처음부터 실행하며 장치만 CPU로 바꿉니다.</p>
              <button className="trainer__run" disabled={retrying} onClick={() => void retryCPU()}>
                {retrying ? "시작 요청 중…" : "CPU로 다시 실행"}
              </button>
            </>
          )}
          <details>
            <summary>오류 원문 보기</summary>
            <pre>{last.error?.detail ?? last.error?.message ?? "상세 오류가 없습니다"}</pre>
          </details>
          <button className="trainer__stop" onClick={() => openRunPanel("stdout")}>학습 출력 보기</button>
        </section>
      )}
      {active?.error && <span className="warn mono">{active.error.message}</span>}
    </div>
  );
}
