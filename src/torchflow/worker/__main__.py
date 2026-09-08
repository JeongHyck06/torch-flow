"""L2 학습 워커 (기획서 §5.5.3, §5.7, §13.1 M7).

hub와 **분리해서** 돈다. `start_new_session=True`로 기동되므로 hub가 죽어도,
브라우저를 닫아도 학습은 계속된다. 통신은 파일 두 개뿐이다.

* ``runs/<id>/events.jsonl`` - append-only. 스칼라·상태·오류가 여기 쌓이고 hub는
  읽기만 한다. hub가 없어도 기록은 남는다.
* ``runs/<id>/control.json`` - hub가 쓰고 워커가 매 스텝 읽는다(pause·stop·lr 변경).
  zmq 연결을 유지하지 않으므로 hub 재시작에도 명령 경로가 끊기지 않는다.
  ``ponytail: 파일 폴링. 스텝당 stat 한 번이고, 명령 지연은 스텝 하나다.``

학습 대상은 **생성 코드 그대로**다(§7.2). 그래프에서 뽑은 ``model.py``를 실행해
모델을 만들기 때문에, 화면에서 보던 모델과 학습되는 모델이 같은 파일이다.

    python -m torchflow.worker --job runs/<id>/job.json
    python -m torchflow.worker --job runs/<id>/job.json --test    # ckpt를 test 분할에 돌려 test.json

"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

from .. import datasets
from ..datasets import load_any

HEARTBEAT_EVERY = 2.0      # 초. hub가 이 파일의 mtime으로 생존을 본다(§5.5.3).


class Events:
    """append-only 이벤트 로그. 한 줄이 하나의 사실이다."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def write(self, kind: str, **fields: Any) -> None:
        self.handle.write(json.dumps({"kind": kind, "wall": time.time(), **fields},
                                     ensure_ascii=False) + "\n")
        self.handle.flush()   # 크래시해도 마지막 스텝까지는 남아야 한다.


class Control:
    """hub가 남긴 명령. 없으면 그냥 계속 돈다."""

    def __init__(self, path: Path):
        self.path = path
        self.stamp = 0.0
        self.state: dict[str, Any] = {}

    def poll(self) -> dict[str, Any]:
        try:
            stamp = self.path.stat().st_mtime
        except OSError:
            return self.state
        if stamp != self.stamp:
            self.stamp = stamp
            try:
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass          # 반쯤 쓰인 파일이면 다음 스텝에 다시 읽는다.
        return self.state


def build_model(job: dict[str, Any]):
    """생성 코드를 실행해 모델을 만든다. 학습 대상 = 사용자가 export할 그 코드."""
    code = Path(job["code"]).read_text(encoding="utf-8")
    namespace: dict[str, Any] = {}
    exec(compile(code, job["code"], "exec"), namespace)   # noqa: S102 - 우리가 방금 생성한 코드다
    factory = namespace[job["class_name"]]
    return factory(**job.get("model_args", {}))


def build_data(job: dict[str, Any], device, generator, rng_state=None):
    """``(배치 이터레이터, 검증 함수 또는 None)``.

    기본은 **배울 것이 있는 합성 과제**다: 고정된 무작위 teacher 사영의 argmax를
    라벨로 쓴다. 순수 난수 라벨이면 손실이 ln(C)에서 평평해서 학습 루프가 도는지조차
    구분되지 않는다. 네트워크도 디스크도 건드리지 않는다(§3.1: 다운로드는 명시
    버튼으로만).

    ``dataset``이 ``teacher``/``noise``가 아니면 데이터셋 이름이다 - 내장(MNIST)이거나
    ``data/<이름>/``의 사용자 데이터. train 분할을 epoch마다 섞어 돌고 test 분할로 검증한다.
    """
    import torch

    name = job.get("dataset", "teacher")
    batch = int(job.get("batch", 32))
    if name not in ("teacher", "noise"):
        splits = load_any(name, Path(job.get("data_dir", "data")), job.get("recipe"))
        if rng_state is not None:
            generator.set_state(rng_state)
        # ponytail: 재개하면 epoch의 첫 배치부터 다시 본다. epoch 안 위치는 ckpt에 없다.
        return (_epochs(torch, splits["train"], batch, device, generator),
                _evaluator(torch, splits["test"], device))

    shape = job["input_shape"]
    classes = int(job.get("num_classes") or 10)
    return _synthetic(torch, shape, classes, batch, device, generator,
                      learnable=name != "noise", rng_state=rng_state), None


def _synthetic(torch, shape, classes, batch, device, generator, *, learnable, rng_state):
    features = 1
    for dim in shape[1:]:
        features *= int(dim)
    # teacher는 시드에서만 나온다. 재개할 때도 같은 과제여야 하므로 상태 복원은
    # teacher를 뽑은 **다음**이다 - 순서를 바꾸면 재개한 run이 다른 문제를 푼다.
    teacher = torch.randn(features, classes, generator=generator) if learnable else None
    if rng_state is not None:
        generator.set_state(rng_state)

    while True:
        inputs = torch.randn([batch, *[int(dim) for dim in shape[1:]]], generator=generator)
        if teacher is None:
            targets = torch.randint(0, classes, (batch,), generator=generator)
        else:
            targets = (inputs.flatten(1) @ teacher).argmax(dim=1)
        yield inputs.to(device), targets.to(device)


def _epochs(torch, split, batch, device, generator):
    """epoch마다 새로 섞는다. 순서는 데이터 generator에서 나오므로 시드가 같으면 같다."""
    inputs, targets = split
    while True:
        order = torch.randperm(len(inputs), generator=generator)
        for start in range(0, len(order) - batch + 1, batch):
            index = order[start:start + batch]
            yield inputs[index].to(device), targets[index].to(device)


def _evaluator(torch, split, device, batch: int = 1000):
    """test 분할 전체의 loss와 정확도. eval 모드로 돌고 원래 모드로 돌려놓는다."""
    inputs, targets = split

    def evaluate(model, loss_fn) -> dict[str, float]:
        was_training = model.training
        model.eval()
        total_loss = correct = 0.0
        with torch.no_grad():
            for start in range(0, len(inputs), batch):
                x = inputs[start:start + batch].to(device)
                y = targets[start:start + batch].to(device)
                output = model(x)
                total_loss += float(loss_fn(output, y)) * len(y)
                correct += float((output.argmax(dim=1) == y).sum())
        model.train(was_training)
        return {"val_loss": total_loss / len(inputs), "val_acc": correct / len(inputs)}

    return evaluate


def make_optimizer(torch, model, job: dict[str, Any]):
    name = str(job.get("optimizer", "adamw")).lower()
    lr = float(job.get("lr", 1e-3))
    weight_decay = float(job.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(torch, optimizer, job: dict[str, Any]):
    """warmup + cosine 하나. 없으면 ``None``이고 lr은 param_groups에 직접 쓴다.

    타입 하나만 두는 이유는 스케줄러 **변경**이 HOT이 아니라 SCHEDULER 분류(재생성 +
    ``last_epoch`` 복원)이고 그것은 v1이기 때문이다(§5.7.1). 여기서 필요한 것은
    "스케줄러가 도는 중에도 lr을 바꿀 수 있는가"뿐이다.

    ``LambdaLR`` 하나로 쓴다. ``SequentialLR``로 조합하면 ``base_lrs``가 중첩되어
    ScaleBaseLR이 안쪽 스케줄러까지 따라 들어가야 한다.
    """
    if str(job.get("scheduler", "none")).lower() != "cosine":
        return None
    total = max(int(job.get("steps", 1)), 1)
    warmup = max(int(job.get("warmup_steps", 0)), 0)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / (warmup + 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def scale_base_lr(scheduler, optimizer, factor: float) -> None:
    """ScaleBaseLR (§5.7.1). 스케줄러가 있을 때 lr을 바꾸는 유일한 올바른 방법.

    ``param_groups["lr"]``에 직접 쓰면 다음 ``scheduler.step()``이 ``base_lrs``에서
    다시 계산해 덮어쓴다 - 한 스텝만 반영되고 사라진다. 절대값이 아니라 비율인
    이유는 그룹마다 base가 다를 수 있기 때문이다(LLRD). 비율은 그 관계를 지킨다.

    현재 lr도 같이 옮긴다. 안 그러면 다음 스케줄러 스텝까지 한 스텝이 옛 lr로 돈다.
    """
    scheduler.base_lrs = [base * factor for base in scheduler.base_lrs]
    for group in optimizer.param_groups:
        group["initial_lr"] = group.get("initial_lr", group["lr"]) * factor
        group["lr"] = group["lr"] * factor


def base_lr_of(scheduler, optimizer) -> float:
    """UI가 "현재 lr = base x schedule"을 쓸 수 있게 base를 알려 준다."""
    if scheduler is not None:
        return float(scheduler.base_lrs[0])
    return float(optimizer.param_groups[0]["lr"])


def pick_device(torch, requested: str | None):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        # 인덱스를 박아 둔다 - hub가 "이 GPU는 L2가 쓴다"고 L1에 알릴 때 쓰는 이름이다.
        return torch.device("cuda", torch.cuda.current_device())
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def deterministic(torch) -> None:
    """Smoke의 전제(§10.4 CI 3단계). 여기서 재는 loss는 bitwise로 같아야 한다.

    ``warn_only``인 이유는 결정적 구현이 없는 op가 학습을 죽이면 안 되기 때문이다 -
    §10.4의 "비지원 op 경고·배지"가 이 자리다.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        # SDPA는 flash/mem-efficient가 비결정적이다. math만 남긴다.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)


def train(job: dict[str, Any], run_dir: Path) -> None:
    """학습 한 번. 학습 루프에서 죽어도 ``status failed``를 남긴다.

    setup 실패는 ``_train`` 안에서 잡지만 루프 안의 예외(shape 불일치 같은 것)는 밖으로
    나온다. 그러면 events.jsonl이 ``running``에서 끝나 hub 화면이 영원히 "step 0 / 500"
    이었다. traceback은 stdout.log에도 그대로 남도록 다시 던진다.
    """
    try:
        _train(job, run_dir)
    except Exception as exc:
        events = Events(run_dir / "events.jsonl")
        events.write("error", stage="train", message=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc())
        events.write("status", state="failed")
        raise


def _train(job: dict[str, Any], run_dir: Path) -> None:
    import torch
    from torch import nn

    events = Events(run_dir / "events.jsonl")
    control = Control(run_dir / "control.json")
    heartbeat = run_dir / "heartbeat"

    seed = int(job.get("seed", 0))
    resumed = 0
    try:
        if job.get("smoke") or job.get("deterministic"):
            deterministic(torch)
        torch.manual_seed(seed)
        model = build_model(job)
        device = pick_device(torch, job.get("device"))
        model.to(device).train()
        optimizer = make_optimizer(torch, model, job)
        scheduler = make_scheduler(torch, optimizer, job)
        loss_fn = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(seed)
        # 체크포인트에서 재개(§5.7.2). 가중치·옵티마이저·RNG 세 가지가 다 돌아와야
        # 이어 붙인 곡선이 끊긴 자리에서 계속된다.
        # CPU로 읽는다. RNG 상태는 CPU ByteTensor여야 하고, 가중치와 옵티마이저
        # 상태는 load_state_dict가 알아서 파라미터가 있는 디바이스로 옮긴다.
        state = torch.load(job["resume"], map_location="cpu") if job.get("resume") else None
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler"):
                scheduler.load_state_dict(state["scheduler"])
                # 재개하며 총 스텝이 늘면 스케줄의 모양이 달라진다. ckpt에 박힌 lr은
                # 옛 스케줄의 값이므로, 새 스케줄이 지금 자리에서 말하는 값으로 맞춘다.
                # 안 하면 재개 첫 스텝이 옛 스케줄의 마지막 lr(대개 0)로 돈다.
                for group, base, shape in zip(optimizer.param_groups, scheduler.base_lrs,
                                              scheduler.lr_lambdas):
                    group["lr"] = base * shape(scheduler.last_epoch)
            # ckpt는 저장 시점의 lr·wd를 들고 온다. 갈라진 run은 바로 그 값을 바꾸려고
            # 갈라진 것이므로 job의 값이 이겨야 한다. 스케줄러가 있으면 job의 lr은
            # 실제 lr이 아니라 base다.
            wanted = job.get("lr")
            if scheduler is not None:
                if wanted is not None and scheduler.base_lrs[0] > 0:
                    scale_base_lr(scheduler, optimizer, float(wanted) / scheduler.base_lrs[0])
            elif wanted is not None:
                for group in optimizer.param_groups:
                    group["lr"] = float(wanted)
            for group in optimizer.param_groups:
                group["weight_decay"] = float(job.get("weight_decay", group["weight_decay"]))
            torch.set_rng_state(state["rng"])
            resumed = int(state["step"])
        data, evaluate = build_data(job, device, generator,
                                    rng_state=state["data_rng"] if state else None)
    except Exception as exc:
        events.write("error", stage="setup", message=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc())
        events.write("status", state="failed")
        return

    total = int(job.get("steps", 200))
    eval_every = max(1, int(job.get("eval_every", 100)))
    nan_policy = job.get("nan_policy", "pause")
    # events.jsonl은 기계가 읽고 stdout은 사람이 읽는다. 학습 스크립트를 직접
    # 돌릴 때 보던 그 출력이 UI의 "학습 출력" 탭에 그대로 나온다.
    print(f"{'재개' if resumed else '시작'} · {device} · "
          f"{sum(p.numel() for p in model.parameters()):,} params · "
          f"step {resumed} / {total}" + (" · smoke" if job.get("smoke") else ""))
    events.write("status", state="running", device=str(device), steps=total,
                 step=resumed, resumed=resumed or None, smoke=bool(job.get("smoke")) or None,
                 scheduler=job.get("scheduler") if scheduler is not None else None,
                 params=sum(p.numel() for p in model.parameters()))

    last_beat = 0.0
    step = resumed
    while step < total:
        command = control.poll()
        if command.get("stop"):
            print(f"stop · step {step} · 체크포인트 저장")
            events.write("status", state="stopped", step=step)
            _checkpoint(torch, model, optimizer, scheduler, generator, run_dir, step)
            return
        if command.get("pause"):
            events.write("status", state="paused", step=step)
            while control.poll().get("pause") and not control.poll().get("stop"):
                _beat(heartbeat)
                time.sleep(0.2)
            if control.poll().get("stop"):
                events.write("status", state="stopped", step=step)
                _checkpoint(torch, model, optimizer, scheduler, generator, run_dir, step)
                return
            events.write("status", state="running", step=step)
        # HOT hparam은 재시작 없이 바로 반영한다(§5.7.1). 매 스텝 control에서 읽으므로
        # "지금 도는 값"이 곧 control.json의 값이다 - 워커가 따로 기억하지 않는다.
        if (wanted := command.get("lr")) is not None:
            wanted = float(wanted)
            current = base_lr_of(scheduler, optimizer)
            if current != wanted:
                if scheduler is not None and current > 0:
                    scale_base_lr(scheduler, optimizer, wanted / current)
                else:
                    for group in optimizer.param_groups:
                        group["lr"] = wanted
                events.write("hparam", step=step, path="optim.lr", value=wanted)
        if (wanted := command.get("weight_decay")) is not None:
            wanted = float(wanted)
            if optimizer.param_groups[0].get("weight_decay") != wanted:
                for group in optimizer.param_groups:
                    group["weight_decay"] = wanted
                events.write("hparam", step=step, path="optim.weight_decay", value=wanted)
        log_every = max(1, int(command.get("log_every", job.get("log_every", 1))))
        clip = float(command.get("grad_clip", job.get("grad_clip", 1e9)))

        inputs, targets = next(data)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = loss_fn(output, targets)

        if not torch.isfinite(loss):
            print(f"loss가 유한하지 않다 · step {step} · nan_policy={nan_policy}")
            events.write("numeric", step=step, nan=True)
            if nan_policy != "continue":
                _checkpoint(torch, model, optimizer, scheduler, generator, run_dir, step)
                events.write("status", state="paused" if nan_policy == "pause" else "stopped",
                             step=step, reason="nan")
                if nan_policy == "stop":
                    return
                while control.poll().get("pause", True) and not control.poll().get("stop"):
                    _beat(heartbeat)
                    time.sleep(0.2)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step += 1

        if step % log_every == 0 or step == total:
            value = float(loss.detach())
            lr = float(optimizer.param_groups[0]["lr"])
            base = base_lr_of(scheduler, optimizer)
            # 분류면 배치 정확도도 적는다 - loss 숫자만으로는 입문자가 "되고 있나"를 못 읽는다.
            acc = (float((output.argmax(dim=1) == targets).float().mean())
                   if output.dim() == 2 and targets.dim() == 1 else None)
            events.write("scalar", step=step, loss=value, lr=lr, grad_norm=float(grad_norm),
                         **({"acc": acc} if acc is not None else {}),
                         **({"base_lr": base} if scheduler is not None else {}))
            print(f"step {step:>6} / {total}   loss {value:.4f}"
                  + (f"   acc {acc:.3f}" if acc is not None else "")
                  + f"   lr {lr:.3g}"
                  + (f" = {base:.3g} x {lr / base if base else 0:.3f}"
                     if scheduler is not None else "")
                  + f"   |g| {float(grad_norm):.3f}")
        if step == total:
            events.write("status", state="finalizing", step=step)
        if evaluate is not None and (step % eval_every == 0 or step == total):
            metrics = evaluate(model, loss_fn)
            events.write("scalar", step=step, **metrics)
            print(f"eval {step:>6}          val_loss {metrics['val_loss']:.4f}"
                  f"   val_acc {metrics['val_acc']:.4f}")
        now = time.monotonic()
        if now - last_beat > HEARTBEAT_EVERY:
            last_beat = now
            _beat(heartbeat)

    _checkpoint(torch, model, optimizer, scheduler, generator, run_dir, step)
    print(f"done · step {step}")
    events.write("status", state="done", step=step)


def _beat(path: Path) -> None:
    path.write_text(str(time.time()), encoding="utf-8")


def _checkpoint(torch, model, optimizer, scheduler, generator, run_dir: Path,
                step: int) -> None:
    """모델·옵티마이저·RNG 둘. 재개가 여기서 읽는다(§5.7.2).

    데이터 generator를 같이 저장하는 이유는 합성 과제가 이 스트림에서 나오기
    때문이다 - 빠뜨리면 재개한 run이 이미 본 배치를 다시 본다.
    """
    # 임시 파일에 쓰고 원자적으로 갈아 끼운다. 제자리에 쓰면 fork나 재개가
    # 쓰다 만 파일을 읽는다 - torch.load가 "zip archive" 오류로 죽는다.
    staging = run_dir / "ckpt.pt.writing"
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "rng": torch.get_rng_state(),
                "data_rng": generator.get_state()}, staging)
    staging.replace(run_dir / "ckpt.pt")


# 모델 테스트: 학습이 보지 않은 분할에 체크포인트를 돌린다.

SAMPLE_TILES = 24          # 화면에 보여 줄 샘플 수. 틀린 것을 먼저 채운다.


def test(job: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """``ckpt.pt``를 test 분할에 돌려 정확도·혼동 행렬·샘플 예측을 ``test.json``에 적는다."""
    try:
        result = _test(job, run_dir)
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()}
    (run_dir / "test.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def _test(job: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import torch
    from torch import nn

    state = torch.load(run_dir / "ckpt.pt", map_location="cpu")
    model = build_model(job)
    model.load_state_dict(state["model"])
    device = pick_device(torch, job.get("device"))
    model.to(device).eval()

    name = job.get("dataset", "teacher")
    classes = int(job.get("num_classes") or 10)
    if name in ("teacher", "noise"):
        x, y = _synthetic_holdout(torch, job, name)
        names, split = [str(index) for index in range(classes)], "합성 홀드아웃"
    else:
        x, y = load_any(name, Path(job.get("data_dir", "data")), job.get("recipe"))["test"]
        spec = datasets.resolve(datasets.describe(name, job.get("data_dir", "data")), job.get("recipe"))
        names = spec.get("class_names") or [str(index) for index in range(classes)]
        split = "test" if name in datasets.CATALOGUE else "val"

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, preds, confs = 0.0, [], []
    with torch.no_grad():
        for start in range(0, len(x), 500):
            output = model(x[start:start + 500].to(device))
            total_loss += float(loss_fn(output, y[start:start + 500].to(device)))
            conf, pred = output.softmax(dim=1).max(dim=1)
            preds.append(pred.cpu())
            confs.append(conf.cpu())
    pred, conf = torch.cat(preds), torch.cat(confs)
    correct = pred == y
    confusion = torch.zeros(len(names), len(names), dtype=torch.int64)
    confusion.index_put_((y, pred), torch.ones_like(y), accumulate=True)
    print(f"test · {split} {len(y)}개 · acc {float(correct.float().mean()):.4f} · loss {total_loss / len(y):.4f}")
    return {"ok": True, "run_id": run_dir.name, "dataset": name, "split": split, "count": len(y),
            "loss": total_loss / len(y), "acc": float(correct.float().mean()),
            "step": int(state["step"]), "device": str(device), "classes": names,
            "per_class": [{"name": label, "count": int((y == index).sum()), "correct": int(confusion[index, index])}
                          for index, label in enumerate(names)],
            "confusion": confusion.tolist(),
            "samples": _sample_tiles(torch, x, y, pred, conf, correct) if x.dim() == 4 else [],
            "wall": time.time()}


def _synthetic_holdout(torch, job: dict[str, Any], name: str, count: int = 1000):
    """같은 teacher, 학습이 보지 않은 입력. teacher는 시드에서 뽑고 입력 스트림만 다른 시드다."""
    seed = int(job.get("seed", 0))
    holdout = torch.Generator().manual_seed(seed + 1)
    stream = _synthetic(torch, job["input_shape"], int(job.get("num_classes") or 10), count,
                        torch.device("cpu"), torch.Generator().manual_seed(seed),
                        learnable=name != "noise", rng_state=holdout.get_state())
    return next(stream)


def _sample_tiles(torch, x, y, pred, conf, correct) -> list[dict[str, Any]]:
    """[N, C, H, W] 입력 몇 개를 작은 PNG로. 틀린 것부터 채운다 - 그게 보고 싶은 것이다."""
    try:
        from PIL import Image
    except ImportError:
        return []
    import base64
    import io

    wrong = (~correct).nonzero().flatten()[:SAMPLE_TILES // 2]
    right = correct.nonzero().flatten()[:SAMPLE_TILES - len(wrong)]
    tiles = []
    for index in torch.cat([wrong, right]).sort().values.tolist():
        image = x[index].float()
        low, high = float(image.min()), float(image.max())
        pixels = ((image - low) / (high - low) * 255 if high > low else image * 0).to(torch.uint8)
        mode = "L" if pixels.shape[0] == 1 else "RGB"
        sheet = Image.frombytes(mode, (pixels.shape[2], pixels.shape[1]),
                                pixels[:3].permute(1, 2, 0).contiguous().numpy().tobytes())
        buffer = io.BytesIO()
        sheet.save(buffer, format="PNG")
        tiles.append({"png": base64.b64encode(buffer.getvalue()).decode(), "pred": int(pred[index]),
                      "label": int(y[index]), "prob": float(conf[index])})
    return tiles


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m torchflow.worker")
    parser.add_argument("--job", required=True)
    parser.add_argument("--test", action="store_true", help="학습 대신 ckpt를 test 분할에 돌린다")
    args = parser.parse_args()

    job_path = Path(args.job)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if args.test:
        test(job, job_path.parent)
    else:
        train(job, job_path.parent)


if __name__ == "__main__":
    main()
