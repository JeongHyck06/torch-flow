// 학습 조작 (기획서 §5.7, §13.1 M7). 하단 Run 패널의 곡선 탭 머리에 붙는다.
//
// L2는 **절대 자동으로 돌지 않는다**(§5.4의 auto = {L0, L1}). 사람이 Run이나 Smoke를
// 눌러야 시작하고, lr은 학습 중에도 바꿀 수 있다(HOT, §5.7.1) - 재시작 없이
// param_groups에 반영된다. 재시작이 필요한 변경은 **알리기만** 한다(ADR-05).
//
// run 종류 표기는 Figma `04 Runs`(48:147)를 따른다: exploratory는 †로 표시하고
// 집계에서 기본 제외, reported는 hparam 동결이라 편집하면 run이 갈라진다.

import { useEffect, useState } from "react";

import { controlTraining, fetchTraining, startTraining } from "../api";
import type { TrainRun } from "../api";

const RUNNING = new Set(["running", "starting"]);

export function Trainer({ onChange }: { onChange: () => void }) {
  const [runs, setRuns] = useState<TrainRun[]>([]);
  const [steps, setSteps] = useState(500);
  const [batch, setBatch] = useState(32);
  const [lr, setLr] = useState(0.001);
  const [schedule, setSchedule] = useState("none");
  const [warmup, setWarmup] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const active = runs.find((run) => RUNNING.has(run.state) || run.state === "paused");
  const last = runs[runs.length - 1];

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      const found = await fetchTraining().catch(() => []);
      if (stop) return;
      setRuns(found);
      if (found.some((run) => RUNNING.has(run.state))) onChange();
    };
    void tick();
    const timer = window.setInterval(() => void tick(), 1000);
    return () => { stop = true; window.clearInterval(timer); };
  }, [onChange]);

  const start = async (smoke = false) => {
    setError(null);
    setNote(null);
    const result = await startTraining(
      smoke ? { smoke: true }
        : { steps, batch, lr, optimizer: "adamw", scheduler: schedule, warmup_steps: warmup });
    if (result.error) setError(result.error);
    else setRuns((previous) => [...previous, result]);
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
      {!active ? (
        <>
          <button className="trainer__run" onClick={() => void start()}>Run</button>
          <button className="trainer__stop" onClick={() => void start(true)}
                  title="20 step · batch 8 · 결정적 - 두 번 돌리면 loss가 같습니다">
            Smoke
          </button>
          {last && (
            <button className="trainer__stop" onClick={() => void send("resume", { steps })}>
              재개
            </button>
          )}
          <label className="trainer__field">steps
            <input type="number" min={1} value={steps}
                   onChange={(event) => setSteps(Number(event.target.value))} />
          </label>
          <label className="trainer__field">batch
            <input type="number" min={1} value={batch}
                   onChange={(event) => setBatch(Number(event.target.value))} />
          </label>
          <label className="trainer__field">lr
            <input type="number" step={0.0001} min={0} value={lr}
                   onChange={(event) => setLr(Number(event.target.value))} />
          </label>
          <label className="trainer__field">스케줄
            <select className="trainer__select mono" value={schedule}
                    onChange={(event) => setSchedule(event.target.value)}>
              <option value="none">없음</option>
              <option value="cosine">cosine</option>
            </select>
          </label>
          {schedule === "cosine" && (
            <label className="trainer__field">warmup
              <input type="number" min={0} value={warmup}
                     onChange={(event) => setWarmup(Number(event.target.value))} />
            </label>
          )}
          {last && (
            <span className="mono trainer__progress">
              {last.run_id} · {last.state} · step {last.step.toLocaleString()}
            </span>
          )}
          <span className="muted trainer__note">합성 과제 · 데이터셋 노드는 Experiment 탭에서</span>
        </>
      ) : (
        <>
          <button className="trainer__run" onClick={() => void send(
            active.state === "paused" ? "resume" : "pause")}>
            {active.state === "paused" ? "Resume" : "Pause"}
          </button>
          <button className="trainer__stop" onClick={() => void send("stop")}>Stop</button>
          <span className="mono trainer__progress">
            step {active.step.toLocaleString()} / {active.total.toLocaleString()}
            {active.device ? ` · ${active.device}` : ""}
            {active.smoke ? " · smoke" : ""}
            {active.state === "paused" ? " · 일시정지" : ""}
            {active.nan_step !== undefined ? ` · NaN @ ${active.nan_step}` : ""}
          </span>
          {reported ? (
            <span className="mono trainer__progress">reported</span>
          ) : (
            <button className="trainer__stop" onClick={() => void send("promote")}
                    title="reported로 올리면 hparam이 동결되고 편집하면 run이 갈라집니다">
              exploratory †
            </button>
          )}
          <label className="trainer__field">{scheduled ? "base lr" : "lr"}
            {/* 제어 입력이어야 한다. defaultValue는 마운트 때만 읽히는데 React가
                Run 폼의 입력 DOM을 재사용해서 batch 값이 남아 있었다. */}
            <input type="number" step={0.0001} min={0} value={lr}
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
      {error && <span className="warn mono">{error}</span>}
      {active?.error && <span className="warn mono">{active.error.message}</span>}
    </div>
  );
}
