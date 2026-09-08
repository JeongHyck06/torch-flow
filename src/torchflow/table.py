"""ablation 표 (기획서 §6.4, §13.1 M8의 "LaTeX 표(수동 축 1개)").

Switch와 Variant Set이 구조로 기록되므로, 논문 표는 새로 만드는 것이 아니라
**기록에서 뽑는 것**이다. 축 하나를 사람이 고르면(``--axis``) 나머지는 자동이다:
시드는 mean +- std로 접고, 열마다 가장 좋은 값을 굵게 한다.

``reported`` run만 센다. exploratory는 이름 옆에 단검을 달아 옵션으로 넣는다(§6.4).

torch를 import하지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

# 값이 클수록 좋은 지표. 나머지는 작을수록 좋다고 본다(loss 계열).
HIGHER_IS_BETTER = ("acc", "accuracy", "top1", "top5", "f1", "auc", "bleu", "map")


@dataclass
class Cell:
    values: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float | None:
        return sum(self.values) / len(self.values) if self.values else None

    @property
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        mean = self.mean or 0.0
        return math.sqrt(sum((value - mean) ** 2 for value in self.values) / (len(self.values) - 1))


@dataclass
class Table:
    axis: str
    metrics: list[str]
    rows: list[str]
    cells: dict[tuple[str, str], Cell]
    exploratory: set[str] = field(default_factory=set)

    def best(self, metric: str) -> str | None:
        """이 지표에서 가장 좋은 행. 값이 없으면 None."""
        scored = [(row, self.cells[(row, metric)].mean) for row in self.rows
                  if self.cells.get((row, metric)) and self.cells[(row, metric)].mean is not None]
        if not scored:
            return None
        higher = any(word in metric.lower() for word in HIGHER_IS_BETTER)
        return (max if higher else min)(scored, key=lambda pair: pair[1])[0]


def dig(source: Any, path: str) -> Any:
    """``job.lr`` 처럼 점으로 이어진 경로를 따라간다. 없으면 None."""
    for part in path.split("."):
        if not isinstance(source, dict):
            return None
        source = source.get(part)
    return source


def build(runs: Iterable[Any], *, axis: str, metrics: list[str],
          final: dict[str, dict[str, float]], include_exploratory: bool = False) -> Table:
    """run 목록을 표로 접는다.

    ``final``은 run id -> {지표: 마지막 값}이다. 곡선 전체가 아니라 마지막 값만 쓰므로
    트래커에 의존하지 않는다 - 그래서 테스트가 쉽다.
    """
    cells: dict[tuple[str, str], Cell] = {}
    rows: list[str] = []
    exploratory: set[str] = set()
    for run in runs:
        kind = getattr(run, "kind", "exploratory")
        if kind != "reported" and not include_exploratory:
            continue
        manifest = getattr(run, "manifest", {}) or {}
        label = dig(manifest, axis)
        if label is None:
            label = getattr(run, "name", None) or getattr(run, "id", "?")
        label = str(label)
        if label not in rows:
            rows.append(label)
        if kind != "reported":
            exploratory.add(label)
        for metric in metrics:
            value = (final.get(getattr(run, "id", ""), {}) or {}).get(metric)
            if value is None:
                continue
            cells.setdefault((label, metric), Cell()).values.append(float(value))
    return Table(axis=axis, metrics=metrics, rows=rows, cells=cells, exploratory=exploratory)


def _format(cell: Cell | None, digits: int = 2) -> str:
    if cell is None or cell.mean is None:
        return "--"
    if len(cell.values) < 2:
        return f"{cell.mean:.{digits}f}"
    return f"{cell.mean:.{digits}f} ± {cell.std:.{digits}f}"


def to_latex(table: Table, *, caption: str = "Ablation", label: str = "tab:ablation") -> str:
    """booktabs 표. 최고값은 굵게, exploratory 행에는 단검을 단다."""
    columns = "l" + "r" * len(table.metrics)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join([_escape(table.axis), *(_escape(one) for one in table.metrics)]) + " \\\\",
        "\\midrule",
    ]
    best = {metric: table.best(metric) for metric in table.metrics}
    for row in table.rows:
        cells = []
        for metric in table.metrics:
            text = _format(table.cells.get((row, metric)))
            cells.append(f"\\textbf{{{text}}}" if best[metric] == row and text != "--" else text)
        name = _escape(row) + ("$^\\dagger$" if row in table.exploratory else "")
        lines.append(" & ".join([name, *cells]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def to_markdown(table: Table) -> str:
    header = f"| {table.axis} | " + " | ".join(table.metrics) + " |"
    rule = "| --- " * (len(table.metrics) + 1) + "|"
    lines = [header, rule]
    best = {metric: table.best(metric) for metric in table.metrics}
    for row in table.rows:
        cells = []
        for metric in table.metrics:
            text = _format(table.cells.get((row, metric)))
            cells.append(f"**{text}**" if best[metric] == row and text != "--" else text)
        name = row + ("†" if row in table.exploratory else "")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_csv(table: Table) -> str:
    lines = [",".join([table.axis, *table.metrics])]
    for row in table.rows:
        lines.append(",".join([row, *(_format(table.cells.get((row, metric)), 4)
                                      for metric in table.metrics)]))
    return "\n".join(lines) + "\n"


def _escape(text: str) -> str:
    for character in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(character, f"\\{character}")
    return text
