// 학습 조작 (기획서 §5.7, §13.1 M7). 하단 Run 패널의 곡선 탭 머리에 붙는다.
//
// L2는 **절대 자동으로 돌지 않는다**(§5.4의 auto = {L0, L1}). 사람이 Run을 눌러야
// 시작하고, lr은 학습 중에도 바꿀 수 있다(HOT, §5.7.1) - 재시작 없이 param_groups에
// 반영되고 곡선에 마커로 남는다.

import { useEffect, useState } from "react";

import { controlTraining, fetchTraining, startTraining } from "../api";
import type { TrainRun } from "../api";

const RUNNING = new Set(["running", "starting"]);

export function Trainer({ onChange }: { onChange: () => void }) {
  const [runs, setRuns] = useState<TrainRun[]>([]);
  const [steps, setSteps] = useState(500);
  const [batch, setBatch] = useState(32);
  const [lr, setLr] = useState(0.001);
  const [error, setError] = useState<string | null>(null);

  const active = runs.find((run) => RUNNING.has(run.state) || run.state === "paused");

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

  const start = async () => {
    setError(null);
    const result = await startTraining({ steps, batch, lr, optimizer: "adamw" });
    if (result.error) setError(result.error);
    else setRuns((previous) => [...previous, result]);
  };

  const send = async (cmd: string, value?: number) => {
    if (!active) return;
    await controlTraining(active.run_id, cmd, value);
  };

  return (
    <div className="trainer">
      {!active ? (
        <>
          <button className="trainer__run" onClick={start}>Run</button>
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
            {active.state === "paused" ? " · 일시정지" : ""}
            {active.nan_step !== undefined ? ` · NaN @ ${active.nan_step}` : ""}
          </span>
          <label className="trainer__field">lr
            <input type="number" step={0.0001} min={0} defaultValue={lr}
                   onKeyDown={(event) => {
                     if (event.key !== "Enter") return;
                     const next = Number((event.target as HTMLInputElement).value);
                     setLr(next);
                     void send("set_lr", next);
                   }} />
          </label>
          <span className="muted trainer__note">lr은 Enter로 학습 중에 바뀝니다</span>
        </>
      )}
      {error && <span className="warn mono">{error}</span>}
      {active?.error && <span className="warn mono">{active.error.message}</span>}
    </div>
  );
}
