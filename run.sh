#!/usr/bin/env bash
# TorchFlow hub를 띄운다.
#
#   ./run.sh                              예제 그래프(MiniViT)로 시작
#   ./run.sh --port 8799                  옵션은 그대로 넘어간다
#   ./run.sh graph/my.tfg.json --rt num_classes=100
#   ./run.sh --new                        빈 첫 화면에서 시작
#
# 예제 그래프는 num_classes를 런타임 상수로 받는다. IR이 런타임 상수의 기본값을
# 들고 있지 않아서(스키마에 자리가 없다) 여기서 대신 채워 준다.
set -euo pipefail
cd "$(dirname "$0")"

python=".venv/bin/python"
[ -x "$python" ] || python="$(command -v python3)"

if [ "${1-}" = "--new" ]; then
    shift
elif [ $# -eq 0 ] || [ "${1#-}" != "$1" ]; then
    set -- examples/minivit.tfg.json --rt num_classes=10 "$@"
fi

exec "$python" -m torchflow.cli view "$@"
