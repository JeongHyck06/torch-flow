"""Attach Mode - 기존 코드에 붙는 관찰층 (기획서 §3.5)."""

from .session import Session, active, hook_step, log, watch
from .trace import trace_module, trace_report

__all__ = ["Session", "active", "hook_step", "log", "watch", "trace_module", "trace_report"]
