// 학습 출력 - 워커 프로세스가 찍은 표준출력 그대로.
//
// 곡선은 숫자를 보여주고 이건 **파이썬이 한 말**을 보여준다. 학습 스크립트를 직접
// 돌릴 때 보던 그 화면이고, traceback도 여기로 온다(§5.6.1 stdout 라우팅).
// 커서를 들고 늘어난 만큼만 받아 이어 붙인다 - 매번 전부 다시 받으면 스크롤이 튄다.

import { useEffect, useRef, useState } from "react";

import { fetchTrainStdout, fetchTraining } from "../api";

export function TrainLog() {
  const [text, setText] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const cursor = useRef(0);
  const box = useRef<HTMLPreElement>(null);
  // 지금 맨 아래를 보고 있는가. 붙이기 **전에** 재야 한다 - 붙인 뒤에 재면
  // 첫 덩어리(꼬리 40 KB)가 통째로 들어온 순간이 "위로 올려 읽는 중"으로 보인다.
  const follow = useRef(true);

  useEffect(() => {
    let stopped = false;
    const tick = async () => {
      const runs = await fetchTraining().catch(() => []);
      if (stopped || !runs.length) return;
      const active = runs.find((run) => run.alive) ?? runs[runs.length - 1];
      if (active.run_id !== runId) {
        // run이 바뀌면 커서도 새로 잡는다. 남은 커서로 읽으면 엉뚱한 자리가 나온다.
        cursor.current = 0;
        setText("");
        setRunId(active.run_id);
      }
      const chunk = await fetchTrainStdout(active.run_id, cursor.current);
      if (stopped || !chunk.text) return;
      const element = box.current;
      follow.current = !element
        || element.scrollHeight - element.scrollTop - element.clientHeight < 40;
      cursor.current = chunk.offset;
      setText((previous) => previous + chunk.text);
    };
    void tick();
    const timer = window.setInterval(() => void tick(), 1000);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [runId]);

  useEffect(() => {
    // 사람이 위로 올려 읽는 중이면 따라가지 않는다.
    if (follow.current && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [text]);

  if (!text) {
    return <p className="mono muted">학습을 시작하면 워커가 찍는 출력이 여기에 흐릅니다</p>;
  }

  return (
    <>
      <div className="trainlog__head mono muted">{runId} · stdout</div>
      <pre className="mono runpanel__json trainlog" ref={box}>{text}</pre>
    </>
  );
}
