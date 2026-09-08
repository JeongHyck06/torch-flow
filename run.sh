#!/usr/bin/env bash
# TorchFlow hub를 띄운다.
#
#   ./run.sh                              첫 화면(프로젝트·템플릿 고르기)에서 시작
#   ./run.sh --port 8799                  옵션은 그대로 넘어간다
#   ./run.sh graph/my.tfg.json --rt num_classes=100   그래프 파일을 바로 연다
#   ./run.sh --minivit                    예제 그래프(MiniViT)로 바로 시작
#
# 템플릿은 첫 화면에서 열면 num_classes 같은 런타임 상수를 스스로 채운다. 파일을 직접 열 때만
# --rt로 넘긴다.
set -euo pipefail
cd "$(dirname "$0")"

python=".venv/bin/python"
[ -x "$python" ] || python="$(command -v python3)"

if [ "${1-}" = "--new" ]; then
    shift
elif [ "${1-}" = "--minivit" ]; then
    shift
    set -- examples/minivit.tfg.json --rt num_classes=10 "$@"
fi

exec "$python" -m torchflow.cli view "$@"
