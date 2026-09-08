"""``torchflow`` CLI."""

from __future__ import annotations

import json
import os
import signal
import sys
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

import click

from . import __version__
from .package import BACKENDS as PACKAGE_BACKENDS
from .paper import PRESETS


def _kv(pairs: tuple[str, ...]) -> dict:
    """``--hp dim=384`` 같은 인자를 파싱한다. 값은 JSON으로 읽되 실패 시 문자열."""
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


@click.group()
@click.version_option(__version__, prog_name="torchflow")
def main() -> None:
    """반응형 PyTorch 연구 워크벤치."""


@main.command()
@click.argument("graph", type=click.Path(exists=True, dir_okay=False))
@click.option("--hp", multiple=True, help="하이퍼파라미터 덮어쓰기 (name=value)")
@click.option("--rt", multiple=True, help="런타임 상수 (num_classes=10)")
def shapes(graph: str, hp: tuple[str, ...], rt: tuple[str, ...]) -> None:
    """L0 정적 패스를 돌려 노드별 shape를 표로 찍는다 (GPU 불필요)."""
    from .ir import load, validate
    from .kernel.l0 import run_pass

    ir = load(graph)
    problems = validate(ir)
    if problems:
        for problem in problems:
            click.secho(f"  {problem}", fg="red")
        sys.exit(1)

    result = run_pass(ir, hp=_kv(hp), rt=_kv(rt))
    for report in result.nodes:
        spec = report.spec
        shape = "·".join(str(d) for d in spec["shape"]) if spec else "—"
        dtype = spec["dtype"] if spec else ""
        click.echo(f"  {report.label:<14} {shape:<24} {dtype:<10} {report.elapsed_ms:6.2f} ms")

    click.echo(
        f"\n  Σ {result.total_params / 1e6:.2f}M params · "
        f"{len(result.nodes)} nodes · {result.elapsed_ms:.0f} ms"
    )
    if result.error:
        error = result.error
        click.secho(f"  {error['kind']} @ {error['node_id']}: {error['message']}", fg="red")
        sys.exit(1)


@main.command()
@click.option("--out", type=click.Path(dir_okay=False), default=".torchflow/registry.json")
def registry(out: str) -> None:
    """블록 레지스트리를 추출한다 (L0 리플렉션)."""
    from .kernel.registry import write

    path = write(out)
    blocks = json.loads(path.read_text(encoding="utf-8"))["blocks"]
    click.echo(f"{len(blocks)} blocks → {path}")


@main.command()
@click.option("--out", type=click.Path(dir_okay=False), default="web/src/types.gen.ts",
              show_default=True)
def typegen(out: str) -> None:
    """Pydantic 모델에서 TS 타입을 생성한다 (§9)."""
    from pathlib import Path

    from .typegen import generate

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate(), encoding="utf-8")
    click.echo(f"types → {path}")


@main.command()
@click.argument("graph", type=click.Path(exists=True, dir_okay=False))
@click.option("--batch", default=4, show_default=True)
@click.option("--objective", default="random_target_ce",
              type=click.Choice(["sum_of_outputs", "random_target_ce", "mse_to_zero"]),
              show_default=True)
@click.option("--rt", multiple=True)
def probe(graph: str, batch: int, objective: str, rt: tuple[str, ...]) -> None:
    """L1 probe를 한 번 돌려 노드별 grad를 표로 찍는다."""
    import torch

    from .ir import load
    from .kernel.devices import pick_l1_device
    from .kernel.l1 import L1Pass, ProbeConfig

    choice = pick_l1_device(torch, batch=batch)
    click.echo(f"  device: {choice.device} ({choice.reason})")
    result = L1Pass(load(graph), rt=_kv(rt), device=choice.device,
                    probe=ProbeConfig(batch=batch, objective=objective,
                                      backward=not choice.forward_only)).probe_once()
    if not result.ok:
        click.secho(f"  {result.error['kind']} @ {result.error['node_id']}: "
                    f"{result.error['message']}", fg="red")
        sys.exit(1)

    for node in result.nodes:
        if node.grad_norm is None:
            continue
        colour = "yellow" if node.warn else None
        click.secho(
            f"  {node.label:<14} ‖g‖={node.grad_norm:.3e}  ‖g‖/‖w‖={node.grad_ratio:.2e}"
            f"  {node.warn or ''}", fg=colour)
    click.echo(f"\n  {result.objective} · loss={result.loss:.4f} · {result.elapsed_ms:.0f} ms")


@main.command()
@click.argument("graph", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", type=click.Path(dir_okay=False), default=None,
              help="쓸 파일. 없으면 표준출력으로 (model-only)")
@click.option("--unit", type=click.Choice(["model-only", "package"]), default="model-only",
              show_default=True, help="출력 단위 (§7.3)")
@click.option("--target", type=click.Path(file_okay=False), default=None,
              help="package 단위가 풀릴 폴더")
@click.option("--backend", type=click.Choice(list(PACKAGE_BACKENDS)),
              default="argparse-dataclass", show_default=True, help="설정 백엔드")
@click.option("--annotate", is_flag=True, help="L0를 돌려 문장 끝에 shape 주석을 단다 (torch 필요)")
@click.option("--rt", multiple=True)
def codegen(graph: str, out: str | None, unit: str, target: str | None, backend: str,
            annotate: bool, rt: tuple[str, ...]) -> None:
    """그래프를 PyTorch 코드로 옮긴다 (§7.2, §7.3)."""
    from pathlib import Path

    from . import codegen as generator
    from . import package as packager
    from .ir import load

    ir = load(graph)
    specs = None
    if annotate:
        from .kernel.l0 import run_pass

        result = run_pass(ir, rt=_kv(rt))
        specs = {report.node_id: {"spec": report.spec} for report in result.nodes}

    if unit == "package":
        if not target:
            raise click.UsageError("package 단위에는 --target 폴더가 필요합니다")
        written = packager.build(ir, target, version=__version__, source=graph, specs=specs,
                                 backend=backend, graph_path=graph)
        for path in written:
            click.echo(f"  {path}")
        click.echo(f"  {len(written)}개 파일 · ir {generator.ir_hash(ir)[:16]} → {target}")
        click.echo(f"  돌려 보기: cd {target} && ./reproduce.sh")
        return

    code = generator.generate(ir, version=__version__, source=graph, specs=specs)
    if out is None:
        click.echo(code)
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    # 소스맵은 프로젝트 뿌리(지금 폴더)에 남긴다 - CI의 torchflow check가 여기서 읽는다(§7.6.2).
    packager.write_sourcemap(Path.cwd(), graph=graph, ir=ir, unit="model-only", files=[path],
                             version=__version__, source=graph)
    click.echo(f"  {len(code.splitlines())} 줄 · ir {generator.ir_hash(ir)[:16]} → {path}")


@main.command("import")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", type=click.Path(dir_okay=False), default=None,
              help="쓸 그래프 파일 (.tfg.json). 없으면 요약만 찍는다")
@click.option("--class", "class_name", default=None, help="최상위 클래스 (없으면 짐작)")
@click.option("--example-input", "example", default="",
              help="Input 규격. 예: x=B,3,32,32:f32")
@click.option("--merge", type=click.Path(exists=True, dir_okay=False), default=None,
              help="재import: 이 그래프에서 라벨과 프로브를 물려받는다 (§7.5)")
def import_code(file: str, out: str | None, class_name: str | None, example: str,
                merge: str | None) -> None:
    """.py를 읽어(실행하지 않고) 편집 가능한 그래프로 만든다 (§7.4 경로 1)."""
    from . import astimport
    from .ir import load, save, validate
    from .pysource import parse_example_spec

    source = Path(file).read_text(encoding="utf-8")
    try:
        ir, report = astimport.import_source(source, filename=file, name=class_name,
                                             previous=load(merge) if merge else None)
    except astimport.AstImportError as exc:
        raise click.ClickException(str(exc)) from None

    if example:
        for node in ir.graph.nodes:
            if (node.type or "").split("@")[0] != "torchflow.Input" or not node.ports_out:
                continue
            specs = parse_example_spec(example)
            for index, port in enumerate(node.ports_out):
                spec = specs.get(port.name) or list(specs.values())[index:index + 1]
                spec = spec if isinstance(spec, dict) else (spec[0] if spec else None)
                if spec:
                    port.shape = list(spec.get("shape") or [])
                    port.dtype = spec.get("dtype") or "float32"

    click.echo(f"  클래스 {', '.join(report['classes'])}")
    click.echo(f"  structural {report['structural']} · cell {report['cell']} · "
               f"비율 {report['structural_ratio'] * 100:.0f}%")
    for problem in report["problems"]:
        click.secho(f"  ! {problem}", fg="yellow")
    for problem in validate(ir):
        click.secho(f"  {problem}", fg="red")
    if out:
        save(ir, out)
        click.echo(f"  → {out}")


@main.command("merge")
@click.argument("base", type=click.Path(exists=True, dir_okay=False))
@click.argument("ours", type=click.Path(exists=True, dir_okay=False))
@click.argument("theirs", type=click.Path(exists=True, dir_okay=False))
def merge_graphs(base: str, ours: str, theirs: str) -> None:
    """git 병합 드라이버 (§10.2). 결과를 OURS 자리에 쓰고, 충돌이 남으면 1로 끝난다.

    한 번만 설정해 두면 git이 알아서 부른다:

        git config merge.torchflow.name "TorchFlow graph merge"
        git config merge.torchflow.driver "torchflow merge %O %A %B"
    """
    from . import merge as merger
    from .ir import load, save

    merged, conflicts = merger.merge(load(base), load(ours), load(theirs))
    save(merged, ours)
    for conflict in conflicts:
        click.secho(f"  충돌 {conflict}", fg="red")
    if conflicts:
        click.echo(f"  {len(conflicts)}군데는 손으로 정해야 합니다 - 나머지는 합쳐 두었습니다")
        sys.exit(1)
    click.secho("  자동으로 합쳤습니다", fg="green")


@main.command()
@click.argument("graph", type=click.Path(exists=True, dir_okay=False))
def show(graph: str) -> None:
    """그래프를 사람이 읽는 줄로 편다 (§10.2의 git textconv).

        git config diff.torchflow.textconv "torchflow show"
    """
    from . import merge as merger
    from .ir import load

    click.echo(merger.summary(load(graph)), nl=False)


@main.command()
@click.option("--state-dir", default=".torchflow", show_default=True)
@click.option("--axis", default="variant", show_default=True,
              help="행을 가르는 축. manifest의 점 경로 (variant, job.lr, job.scheduler ...)")
@click.option("--metric", "metrics", multiple=True, default=("val_acc",), show_default=True,
              help="열이 될 지표. 여러 번 줄 수 있다")
@click.option("--format", "shape", type=click.Choice(["latex", "markdown", "csv"]),
              default="latex", show_default=True)
@click.option("--all", "include_exploratory", is_flag=True,
              help="exploratory run도 넣는다 (표에 단검으로 표시)")
@click.option("--out", type=click.Path(dir_okay=False), default=None)
def table(state_dir: str, axis: str, metrics: tuple[str, ...], shape: str,
          include_exploratory: bool, out: str | None) -> None:
    """기록된 run에서 ablation 표를 만든다 (§6.4). 시드는 mean±std로 접는다."""
    from . import table as tables
    from .hub.tracker import Tracker

    tracker = Tracker(Path(state_dir) / "runs.db")
    runs = tracker.runs(limit=500)
    final = {}
    for run in runs:
        values = {}
        for metric in metrics:
            curve = tracker.curve(run.id, metric)
            if curve:
                values[metric] = curve[-1][1]
        final[run.id] = values
    built = tables.build(runs, axis=axis, metrics=list(metrics), final=final,
                         include_exploratory=include_exploratory)
    if not built.rows:
        raise click.ClickException(
            "표에 넣을 run이 없습니다 - reported로 표시한 run이 있어야 합니다 (--all로 전부 넣기)")
    text = {"latex": tables.to_latex, "markdown": tables.to_markdown,
            "csv": tables.to_csv}[shape](built)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        click.echo(f"  {len(built.rows)}행 · {len(metrics)}열 → {out}")
    else:
        click.echo(text)


@main.command()
@click.argument("project", type=click.Path(exists=True, file_okay=False), default=".")
def check(project: str) -> None:
    """리뷰 계약을 검사한다 (§10.2): 코드가 그래프와 같은가, 손으로 고쳐지지 않았는가."""
    from . import package as packager

    problems = packager.check(project)
    for problem in problems:
        click.secho(f"  {problem}", fg="red")
    if problems:
        sys.exit(1)
    click.secho("  통과 - 생성 코드가 그래프와 일치합니다", fg="green")


@main.command()
@click.argument("revisions", default="HEAD~1..HEAD")
@click.option("--graph", "graph_path", type=click.Path(), default=None,
              help="비교할 그래프 파일 (없으면 .sourcemap.json에서 찾는다)")
@click.option("--markdown", "as_markdown", is_flag=True, help="PR 본문에 붙일 형태로")
def diff(revisions: str, graph_path: str | None, as_markdown: bool) -> None:
    """두 커밋 사이의 그래프 변경을 요약한다 (§10.2). 예: torchflow diff main..HEAD"""
    import subprocess

    from . import graphdiff
    from .ir import ModuleGraph

    base, _, head = revisions.partition("..")
    if not head:
        raise click.UsageError("base..head 형태로 주세요 (예: main..HEAD)")
    if graph_path is None:
        from . import package as packager

        try:
            recorded = json.loads(Path(packager.SOURCEMAP).read_text(encoding="utf-8"))
            graph_path = recorded["graph"]
        except (OSError, KeyError, json.JSONDecodeError):
            raise click.UsageError("--graph 로 그래프 파일을 지정하세요") from None

    def at(revision: str) -> ModuleGraph:
        try:
            blob = subprocess.run(["git", "show", f"{revision}:{graph_path}"],
                                  capture_output=True, check=True).stdout
        except subprocess.CalledProcessError as exc:
            raise click.ClickException(
                f"{revision}:{graph_path} 를 읽지 못했습니다: "
                f"{exc.stderr.decode(errors='replace').strip()}") from None
        return ModuleGraph.model_validate_json(blob)

    before, after = at(base), at(head)
    if as_markdown:
        click.echo(graphdiff.markdown(before, after))
        return
    lines = graphdiff.summarize(before, after)
    if not lines:
        click.echo("  그래프는 그대로입니다")
    for line in lines:
        click.echo(f"  {line}")


@main.command()
@click.argument("graph", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", type=click.Path(dir_okay=False), default="paper/figure.pdf",
              show_default=True)
@click.option("--preset", type=click.Choice(sorted(PRESETS)), default="neurips",
              show_default=True, help="본문 단 폭 프리셋")
@click.option("--anonymous", is_flag=True, help="도구 이름과 버전을 지운다 (이중 블라인드)")
def export(graph: str, out: str, preset: str, anonymous: bool) -> None:
    """아키텍처 다이어그램을 SVG 또는 PDF로 내보낸다 (§6.4)."""
    from pathlib import Path

    from . import paper
    from .ir import load

    figure = paper.build(load(graph), preset=preset, anonymous=anonymous,
                         version=__version__)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".svg":
        path.write_text(paper.to_svg(figure), encoding="utf-8")
    else:
        path.write_bytes(paper.to_pdf(figure))
    width, height = figure.page
    click.echo(f"  {len(figure.boxes)} nodes · {width:.0f}×{height:.0f} pt · "
               f"ir {figure.ir_hash} → {path}")


@main.group()
def cache() -> None:
    """디스크 캐시 관리."""


@cache.command("gc")
@click.option("--state-dir", default=".torchflow", show_default=True)
@click.option("--quota-gb", default=20.0, show_default=True)
@click.option("--dry-run", is_flag=True)
def cache_gc(state_dir: str, quota_gb: float, dry_run: bool) -> None:
    """쿼터를 넘긴 만큼 참조 없는 항목부터 지운다 (§5.2)."""
    from pathlib import Path

    from .hub.cache import DiskCAS

    cas = DiskCAS(Path(state_dir) / "cache", quota_gb=quota_gb)
    before = cas.size_bytes()
    report = cas.gc(dry_run=dry_run)
    click.echo(f"  {before / 1024**3:.2f} GB → {(before - report.freed_bytes) / 1024**3:.2f} GB"
               f"  ({report.removed}/{report.scanned} 항목"
               f"{', dry-run' if dry_run else ''})")


@main.command()
@click.option("--state-dir", default=".torchflow", show_default=True)
@click.option("--out", default="runs/tb", show_default=True)
@click.option("--run", "run_ids", multiple=True, help="특정 run만 (기본: 전부)")
def tb(state_dir: str, out: str, run_ids: tuple[str, ...]) -> None:
    """트래커의 곡선을 TensorBoard가 읽는 tfevents로 내보낸다 (§11)."""
    from .hub.tensorboard import export
    from .hub.tracker import Tracker

    tracker = Tracker(Path(state_dir) / "runs.db")
    written = export(tracker, out, list(run_ids) or None)
    tracker.close()
    for path in written:
        click.echo(f"  {path}")
    click.secho(f"\n  {len(written)}개 run → tensorboard --logdir {out}\n", fg="green")


@main.command()
@click.argument("graph", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--port", default=8765, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--state-dir", default=".torchflow", show_default=True)
@click.option("--rt", multiple=True, help="런타임 상수 (num_classes=10)")
@click.option("--browser/--no-browser", default=True, show_default=True)
@click.option("--token", default=None, help="토큰을 직접 지정 (Attach 모드가 쓴다)")
def view(graph: str | None, port: int, host: str, state_dir: str, rt: tuple[str, ...],
         browser: bool, token: str | None) -> None:
    """hub를 띄운다. 그래프를 주지 않으면 첫 화면이 뜬다."""
    import uvicorn

    from .hub.app import create_app
    from .hub.auth import new_token

    # 사람이 직접 띄운 hub는 프로젝트에 하나다. 이미 떠 있으면 새로 띄우지 않고 그 창을 연다 -
    # 둘이 같은 runs/를 나눠 쓰면 run이 섞인다. 토큰을 직접 준 경우(Attach, 테스트)는 예외다.
    lock = Path(state_dir) / "hub.json"
    if token is None and (running := _running_hub(lock)) is not None:
        click.secho(f"\n  TorchFlow가 이미 열려 있습니다 → {running}\n", fg="yellow")
        if browser:
            webbrowser.open(running)
        return
    owns_lock = token is None
    token = token or new_token()
    app = create_app(graph, state_dir=state_dir, token=token, rt=_kv(rt))
    # 지난번 hub가 죽는 동안에도 워커는 돌았다. runs/를 훑어 다시 집는다(§5.5.3).
    if recovered := app.state.hub.recover():
        click.echo(f"  이어받은 run {len(recovered)}개")
    url = f"http://{host}:{port}/?token={token}"
    click.secho(f"\n  TorchFlow hub → {url}\n", fg="green")
    click.echo(f"  원격이면: ssh -L {port}:127.0.0.1:{port} <host>\n")
    if browser:
        webbrowser.open(url)
    if owns_lock:
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"pid": os.getpid(), "host": host, "port": port, "token": token}),
                        encoding="utf-8")
        # uvicorn은 다 내린 뒤 받았던 시그널을 기본 동작으로 다시 던진다 - 그러면 finally가 돌지
        # 않아 잠금이 남는다. SIGTERM은 우리가 받아 정상 종료로 바꾼다.
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        # Ctrl+C 뒤에 브라우저 탭이 붙들고 있는 연결(2초 폴링, WebSocket)을 기다리느라 hub가 안 내려간
        # 적이 있다 - 3초 뒤에는 남은 연결을 끊고 내려간다.
        uvicorn.run(app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=3)
    finally:
        if owns_lock:
            lock.unlink(missing_ok=True)


def _running_hub(lock: Path) -> str | None:
    """지난 hub가 남긴 hub.json이 가리키는 hub가 지금도 응답하면 그 URL. 아니면 None."""
    try:
        info = json.loads(lock.read_text(encoding="utf-8"))
        pid, host, port, token = info["pid"], info["host"], info["port"], info["token"]
    except (OSError, ValueError, KeyError):
        return None
    try:
        os.kill(int(pid), 0)
    except OSError:
        return None                     # 죽은 hub의 흔적이다. 새로 띄운다.
    request = urllib.request.Request(f"http://{host}:{port}/api/health",
                                     headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=1.0) as response:
            if response.status != 200:
                return None
    except (urllib.error.URLError, OSError):
        click.secho(f"  이전 hub(PID {pid})가 살아 있지만 응답하지 않습니다. 끝낸 뒤 다시 실행하세요",
                    fg="red")
        raise SystemExit(1)
    return f"http://{host}:{port}/?token={token}"


if __name__ == "__main__":
    main()
