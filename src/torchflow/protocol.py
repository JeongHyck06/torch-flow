"""와이어 프로토콜 (기획서 §8.2, 예시 §17.5).

두 개의 링크가 있다.

* ``hub <-> kernel``: pyzmq, msgpack 헤더 + 멀티파트 raw 프레임.
* ``browser <-> hub``: WebSocket, JSON 텍스트 프레임 + 바이너리 프레임.

두 링크 모두 Pydantic 모델이 단일 진실원이다. TS 타입은 여기서 생성한다.

지금 정의된 것은 M1–M2가 쓰는 메시지뿐이다. L1 프로브(``RunClosure``,
``SwapModules``, ``SetProbeBudget``)와 L2 학습(``TrainCmd``, ``ScaleBaseLR``,
``Scalar``)은 해당 마일스톤에서 같은 방식으로 추가한다.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

import msgpack
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# 실행 축 - 노드 상태가 축마다 따로 있다(§5.3).
Level = Literal["L0", "L1fwd", "L1bwd", "L2"]
# 커널 프로세스의 정체. 축과 다르다: L1 커널 하나가 L1fwd와 L1bwd를 모두 돈다.
KernelLevel = Literal["L0", "L1", "L2"]


class _Msg(BaseModel):
    model_config = ConfigDict(extra="allow")


# hub -> kernel


class NodeRequest(_Msg):
    node_id: str
    version: int
    inputs: list[str] = Field(default_factory=list)


class Ping(_Msg):
    type: Literal["Ping"] = "Ping"
    req_id: str


class RunNodes(_Msg):
    """위상 순 배치 실행. §17.5의 정본 예시 형태."""

    type: Literal["RunNodes"] = "RunNodes"
    req_id: str
    level: Level = "L0"
    pass_: Literal["static", "real"] = Field(default="static", alias="pass")
    probe_batch: int | None = None
    graph: dict[str, Any] | None = None  # 첫 요청 또는 구조 변경 시 IR 스냅샷
    batch: list[NodeRequest] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Cancel(_Msg):
    type: Literal["Cancel"] = "Cancel"
    req_id: str


class ReloadBlocks(_Msg):
    type: Literal["ReloadBlocks"] = "ReloadBlocks"
    req_id: str
    paths: list[str] = Field(default_factory=list)


class RunClosure(_Msg):
    """L1-bwd - 목적함수의 조상 폐쇄를 autograd 포함 1회 forward+backward(§5.1.2)."""

    type: Literal["RunClosure"] = "RunClosure"
    req_id: str
    level: Level = "L1bwd"
    graph: dict[str, Any] | None = None
    node_ids: list[str] = Field(default_factory=list)
    probe_cfg: dict[str, Any] = Field(default_factory=dict)
    # L2 학습 워커가 잡고 있는 디바이스. L1은 여기를 피해 배정한다(§5.1.5).
    occupied: list[str] = Field(default_factory=list)


class ImportTrace(_Msg):
    """인스턴스 import (§7.4 경로 3). 커널이 사용자 모델을 만들어 트레이스한다."""

    type: Literal["ImportTrace"] = "ImportTrace"
    req_id: str
    file: str
    factory: str
    example_inputs: dict[str, Any] = Field(default_factory=dict)


class EstimateMemory(_Msg):
    """메모리 밴드 추정 요청 (§6.2). 배치를 구체값으로 묶어 실행한다."""

    type: Literal["EstimateMemory"] = "EstimateMemory"
    req_id: str
    graph: dict[str, Any] | None = None
    batch: int = 64
    optimizer: str = "adam"
    amp: bool = False


class NodeDetail(_Msg):
    """블록 상세(§6.3). 마지막 probe의 입력·출력·가중치로 그림을 만든다."""

    type: Literal["NodeDetail"] = "NodeDetail"
    req_id: str
    node_id: str
    path: str = ""


class Eval(_Msg):
    """Debug Console (§5.6.1). 선택 노드의 마지막 probe 값을 표현식으로 조회한다."""

    type: Literal["Eval"] = "Eval"
    req_id: str
    expr: str
    node_id: str = ""
    path: str = ""


class Shutdown(_Msg):
    type: Literal["Shutdown"] = "Shutdown"
    req_id: str = "shutdown"


ToKernel = Annotated[
    Union[Ping, RunNodes, RunClosure, Cancel, ReloadBlocks, ImportTrace, EstimateMemory, NodeDetail,
          Eval, Shutdown],
    Field(discriminator="type"),
]


# kernel -> hub


class Ready(_Msg):
    """커널이 기동을 마쳤다. registry는 L0만 채운다."""

    type: Literal["Ready"] = "Ready"
    req_id: str = "ready"
    level: KernelLevel = "L0"
    pid: int = 0
    torch_version: str | None = None
    device: str = "cpu"
    # 이 컴퓨터에서 쓸 수 있는 가속기 전부(cuda:0, cuda:1, mps ...). 학습 장치를 고르는 데 쓴다.
    devices: list[str] = Field(default_factory=list)
    registry_path: str | None = None


class Pong(_Msg):
    type: Literal["Pong"] = "Pong"
    req_id: str
    level: KernelLevel = "L0"


class TensorSpec(_Msg):
    shape: list[str | int] = Field(default_factory=list)
    dtype: str | None = None
    device: str | None = None


class Done(_Msg):
    type: Literal["Done"] = "Done"
    req_id: str
    node_id: str
    # 호출 경로 - 같은 컴포지트가 여러 번 인스턴스화될 때 결과를 가른다.
    path: str = ""
    version: int = 0
    output_key: str | None = None
    digest: str | None = None
    spec: dict[str, Any] | None = None
    elapsed_ms: float = 0.0
    cache_hit: bool = False
    params: int | None = None
    flops: int | None = None
    numeric: dict[str, Any] | None = None
    # L1 probe 결과(§17.5의 Done 예시).
    grad: dict[str, Any] | None = None
    histogram: list[int] | None = None
    feature: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = Field(default_factory=list)


# input: 이어지지 않은 포트, args: 비어 있는 필수 인자 - 편집 중 가장 흔한 둘이라 이름을 준다.
ErrorKind = Literal["shape", "exception", "oom", "export", "kernel", "timeout", "input", "args"]


class Error(_Msg):
    type: Literal["Error"] = "Error"
    req_id: str
    kind: ErrorKind
    node_id: str | None = None
    message: str = ""
    mapping: dict[str, Any] | None = None
    traceback: str | None = None


class Busy(_Msg):
    type: Literal["Busy"] = "Busy"
    req_id: str = "busy"
    level: KernelLevel = "L0"
    reason: str = ""
    expected_ms: int | None = None


class Log(_Msg):
    type: Literal["Log"] = "Log"
    req_id: str = "log"
    level_name: str = "info"
    node_id: str | None = None
    text: str = ""


class Imported(_Msg):
    type: Literal["Imported"] = "Imported"
    req_id: str
    graph: dict[str, Any] | None = None
    report: dict[str, Any] = Field(default_factory=dict)


class MemoryEstimate(_Msg):
    type: Literal["MemoryEstimate"] = "MemoryEstimate"
    req_id: str
    ok: bool = True
    batch: int = 64
    optimizer: str = "adam"
    breakdown_bytes: dict[str, int] = Field(default_factory=dict)
    total_bytes: int = 0
    band_gb: list[float] = Field(default_factory=list)
    error: dict[str, Any] | None = None


class NodeDetailResult(_Msg):
    type: Literal["NodeDetailResult"] = "NodeDetailResult"
    req_id: str
    ok: bool = True
    error: str | None = None
    node: str = ""
    label: str = ""
    kind: str = ""
    params: int = 0
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    explain: str = ""
    panels: list[dict[str, Any]] = Field(default_factory=list)


class EvalResult(_Msg):
    type: Literal["EvalResult"] = "EvalResult"
    req_id: str
    ok: bool = True
    text: str = ""
    spec: dict[str, Any] | None = None
    error: str | None = None


class Progress(_Msg):
    """요청 하나의 종결 메시지. 패스 총계도 여기 실린다."""

    type: Literal["Progress"] = "Progress"
    req_id: str
    done: int = 0
    total: int = 0
    total_params: int | None = None
    rerun_ratio: float | None = None


FromKernel = Annotated[
    Union[Ready, Pong, Done, Error, Busy, Log, Progress, MemoryEstimate, Imported, EvalResult,
          NodeDetailResult],
    Field(discriminator="type"),
]


# browser <-> hub

OpKind = Literal[
    "add_node", "remove_node", "set_param", "set_ports", "rename", "connect", "disconnect",
    "move",
    "set_code", "set_switch_active", "promote_hp", "save_variant", "add_probe",
    "remove_probe", "batch",
]


class Op(_Msg):
    type: Literal["Op"] = "Op"
    client_id: str
    tmp_seq: int
    kind: OpKind
    payload: dict[str, Any] = Field(default_factory=dict)
    inverse_of: int | None = None
    ops: list[dict[str, Any]] | None = None  # kind == "batch"


class OpAck(_Msg):
    type: Literal["OpAck"] = "OpAck"
    tmp_seq: int
    seq: int


class OpBroadcast(_Msg):
    type: Literal["OpBroadcast"] = "OpBroadcast"
    seq: int
    op: dict[str, Any]


class NodeState(_Msg):
    type: Literal["NodeState"] = "NodeState"
    seq: int = 0
    node: str
    path: str = ""
    axis: Level = "L0"
    state: str = "ok"
    spec: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    badges: dict[str, Any] = Field(default_factory=dict)


class KernelStatus(_Msg):
    type: Literal["KernelStatus"] = "KernelStatus"
    level: KernelLevel = "L0"
    alive: bool = False
    responsive: bool = False
    device: str = "cpu"
    busy: dict[str, Any] | None = None
    mem: dict[str, Any] | None = None


class Auth(_Msg):
    type: Literal["Auth"] = "Auth"
    token: str


class Subscribe(_Msg):
    type: Literal["Subscribe"] = "Subscribe"
    node_ids: list[str] = Field(default_factory=list)


class Resync(_Msg):
    type: Literal["Resync"] = "Resync"
    from_seq: int = 0
    to_seq: int = 0
    ops: list[dict[str, Any]] = Field(default_factory=list)
    snapshot: dict[str, Any] | None = None


FromBrowser = Annotated[Union[Auth, Op, Subscribe, Resync], Field(discriminator="type")]
ToBrowser = Annotated[
    Union[OpAck, OpBroadcast, NodeState, KernelStatus, Resync, Error, Log],
    Field(discriminator="type"),
]

_to_kernel = TypeAdapter(ToKernel)
_from_kernel = TypeAdapter(FromKernel)
_from_browser = TypeAdapter(FromBrowser)


# 인코딩


def encode(message: BaseModel, frames: list[bytes] | None = None) -> list[bytes]:
    """멀티파트 zmq 프레임으로 직렬화한다: ``[헤더, *raw]``."""
    header = message.model_dump(mode="json", by_alias=True, exclude_none=True)
    return [msgpack.packb(header, use_bin_type=True), *(frames or [])]


def decode_to_kernel(parts: list[bytes]) -> tuple[Any, list[bytes]]:
    return _to_kernel.validate_python(msgpack.unpackb(parts[0], raw=False)), list(parts[1:])


def decode_from_kernel(parts: list[bytes]) -> tuple[Any, list[bytes]]:
    return _from_kernel.validate_python(msgpack.unpackb(parts[0], raw=False)), list(parts[1:])


def decode_from_browser(raw: str | bytes) -> Any:
    return _from_browser.validate_json(raw)
