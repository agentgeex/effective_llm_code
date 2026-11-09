#!/usr/bin/env bash
# 安装 spaCy 官方中文模型（zh_core_web_sm / zh_core_web_lg）
#
# 为什么不在 pyproject.toml 里：
#   这两个模型是 GitHub Release 的 wheel（spaCy 官方模型不在 PyPI），而 `uv lock`
#   会把**所有** optional-dependencies 分组一起解析 —— 只要写进任何一组，读者即使
#   只想 `uv sync --extra core`（numpy / pandas / openai，全在 PyPI），也必须先能
#   访问 GitHub 才拿得到这两个 wheel 的元数据。国内网络这一步常常直接断。
#
# 代价（说清楚，别粉饰）：
#   `uv sync` 会把不在 pyproject.toml 里的包当「多余」删掉。顺序是
#   「先 uv sync --extra core 建好 .venv，再跑本脚本」；之后又 sync 就再跑一次。
#
# 用法（在仓库根目录）：
#   bash tools/install_spacy_models.sh sm     # zh_core_web_sm，约 12MB
#   bash tools/install_spacy_models.sh lg     # zh_core_web_lg，约 575MB
#   bash tools/install_spacy_models.sh both   # 两个都装
#
# 校验：
#   .venv/bin/python -c "import spacy; print(spacy.load('zh_core_web_sm').meta['version'])"

set -euo pipefail

SM="https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl"
LG="https://github.com/explosion/spacy-models/releases/download/zh_core_web_lg-3.8.0/zh_core_web_lg-3.8.0-py3-none-any.whl"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "${1:-}" in
    sm)   urls=("$SM") ;;
    lg)   urls=("$LG") ;;
    both) urls=("$SM" "$LG") ;;
    *)
        echo "用法: bash tools/install_spacy_models.sh {sm|lg|both}" >&2
        exit 2
        ;;
esac

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "找不到 .venv/bin/python —— 请先在本目录执行：uv sync --extra core" >&2
    exit 2
fi

# 模型只是 spacy 的数据包：没有 spacy 装上也用不了，所以先保证它在。
echo "==> uv pip install spacy"
uv pip install --python "$PY" spacy

for u in "${urls[@]}"; do
    echo "==> uv pip install ${u##*/}"
    uv pip install --python "$PY" "$u"
done

echo
echo "装好了。校验："
echo "  $PY -c \"import spacy; print(spacy.load('zh_core_web_sm').meta['version'])\""
echo "注意：之后任何一次 'uv sync' 都会把它们删掉，需要时重跑本脚本。"
