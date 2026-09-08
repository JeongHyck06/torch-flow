// 지금 run의 상태와 조작 (기획서 §5.7, §13.1 M7). 하단 패널 곡선 탭의 머리에 붙는다.
//
// **시작은 여기서 하지 않는다** - 단계 표시줄의 4 실행이 유일한 시작 버튼이고(train.ts startRun),
// 여기는 돌고 있는 run의 Pause/Stop/lr(HOT, §5.7.1)과 멈춘 run의 재개만 둔다. 재시작이 필요한
// 변경은 **알리기만** 한다(ADR-05). run 종류 표기는 Figma `04 Runs`(48:147)를 따른다:
// exploratory는 †로 표시하고 집계에서 기본 제외, reported는 hparam 동결이라 편집하면 run이 갈라진다.

import { useEffect, useState } from "react";

import { controlTraining, fetchDatasets } from "../api";
import type { DatasetInfo } from "../api";
import { applyEdit } from "../edit";
import { op } from "../graph/ops";
import { RUNNING } from "../stages";
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
      {!active ? (
        <>
          <span className="mono trainer__progress">
            {last ? `${last.run_id} · ${last.state} · step ${last.step.toLocaleString()}` : "아직 run이 없습니다"}
          </span>
          {last && last.state !== "done" && (
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
                + ` · Input 규격 [B, ${chosen.shape.join(", ")}] · 출력 ${chosen.classes} 클래스`
                + (chosen.source === "user" ? " · 8:2로 나눠 검증" : " · 검증은 test 분할")
              : "합성 과제 · 무작위 입력에 고정 teacher 라벨"}
            {" · 시작은 단계 표시줄의 4 실행"}
          </span>
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
      {active?.error && <span className="warn mono">{active.error.message}</span>}
    </div>
  );
}
